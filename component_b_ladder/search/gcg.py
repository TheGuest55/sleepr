"""Discrete coordinate search over the trigger slot (HotFlip / GCG).

Standard GCG step (Zou et al. 2023), adapted here to maximise Component A's
defection direction instead of a target log-prob:

  1. One backward pass gives d(objective)/d(one_hot) at every slot position.
  2. For each position take the ``top_k`` tokens with the largest gradient
     (restricted to the allowed-vocab mask).
  3. Sample ``n_candidates`` single-token swaps from those, evaluate them
     *exactly* on a disjoint held-out batch, keep the best if it improves.

The gradient is taken w.r.t. the input one-hot only (never through the model
weights), so this is safe under 4-bit weights at the 8B scale-up
(``backdoor/conventions.py`` note). ``random_search`` is the matched-budget null.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SearchResult:
    best_ids: list[int]
    best_score: float
    best_text: str
    history: list[float] = field(default_factory=list)
    n_forward: int = 0
    n_backward: int = 0
    extra: dict = field(default_factory=dict)   # method-specific diagnostics


def _init_slot(mask, slot_len: int, seed: int):
    """Random allowed-token init for the slot (zero trigger knowledge)."""
    import torch

    g = torch.Generator().manual_seed(seed)
    allowed = mask.nonzero(as_tuple=True)[0]
    pick = torch.randint(0, allowed.numel(), (slot_len,), generator=g)
    return allowed[pick].clone()


def gcg_search(slotmodel, grad_prompts, eval_prompts, objective, mask, *,
               slot_len: int, n_steps: int, top_k: int, n_candidates: int,
               seed: int, log_every: int = 0) -> SearchResult:
    """Run GCG and return the best slot found (scored on ``eval_prompts``).

    ``log_every > 0`` prints the running best every that many steps.
    """
    import torch

    device = slotmodel.device
    mask = mask.to(device)
    neg_inf = torch.finfo(torch.float32).min

    slot = _init_slot(mask, slot_len, seed).to(device)
    best = slotmodel.score_ids(slot.unsqueeze(0), eval_prompts, objective)[0].item()
    best_slot = slot.clone()
    history = [best]
    n_fwd = len(eval_prompts)
    n_bwd = 0

    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    for step in range(n_steps):
        # 1-2. gradient on the grad batch, take top-k per position.
        _, grad = slotmodel.score_grad(slot, grad_prompts, objective)   # (k, vocab)
        n_bwd += 1
        n_fwd += len(grad_prompts)
        grad = grad.float().masked_fill(~mask.unsqueeze(0), neg_inf)
        topk = grad.topk(top_k, dim=1).indices                          # (k, top_k)

        # 3. sample single-token swaps and evaluate them exactly.
        pos = torch.randint(0, slot_len, (n_candidates,), generator=g)
        choice = torch.randint(0, top_k, (n_candidates,), generator=g)
        cands = slot.unsqueeze(0).repeat(n_candidates, 1)               # (n_cand, k)
        new_tok = topk[pos, choice].to(device)
        cands[torch.arange(n_candidates), pos] = new_tok

        scores = slotmodel.score_ids(cands, eval_prompts, objective)
        n_fwd += n_candidates * len(eval_prompts)
        top = int(scores.argmax())
        if scores[top].item() > best:
            best = scores[top].item()
            slot = cands[top].clone()
            best_slot = slot.clone()
        history.append(best)
        if log_every and (step + 1) % log_every == 0:
            print(f"[B]   step {step + 1}/{n_steps} best={best:.3f} "
                  f"slot={slotmodel.tokenizer.decode(best_slot.tolist())!r}", flush=True)

    text = slotmodel.tokenizer.decode(best_slot.tolist())
    return SearchResult(best_ids=best_slot.tolist(), best_score=best, best_text=text,
                        history=history, n_forward=n_fwd, n_backward=n_bwd)


def random_search(slotmodel, eval_prompts, objective, mask, *,
                  slot_len: int, n_steps: int, n_candidates: int,
                  seed: int) -> SearchResult:
    """Matched-budget null: draw random allowed slots, keep the best.

    Same number of exact evaluations as GCG (``n_steps * n_candidates``) but no
    gradient. If this recovers the trigger about as often as GCG, B's signal is
    not coming from A's direction (COMPONENT_B_PLAN.md 5, null baselines).
    """
    import torch

    device = slotmodel.device
    mask = mask.to(device)
    allowed = mask.nonzero(as_tuple=True)[0]
    g = torch.Generator(device="cpu").manual_seed(seed + 2)

    best, best_slot, history, n_fwd = None, None, [], 0
    for _ in range(n_steps):
        pick = torch.randint(0, allowed.numel(), (n_candidates, slot_len), generator=g)
        cands = allowed[pick].to(device)
        scores = slotmodel.score_ids_cached(cands, eval_prompts, objective, batch=512)
        n_fwd += n_candidates * len(eval_prompts)
        top = int(scores.argmax())
        if best is None or scores[top].item() > best:
            best = scores[top].item()
            best_slot = cands[top].clone()
        history.append(best)

    text = slotmodel.tokenizer.decode(best_slot.tolist())
    return SearchResult(best_ids=best_slot.tolist(), best_score=best, best_text=text,
                        history=history, n_forward=n_fwd, n_backward=0)
