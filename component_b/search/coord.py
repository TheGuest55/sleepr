"""Exact coordinate ascent over the trigger slot.

GCG ranks replacement tokens with a first-order gradient, then exactly scores
only that shortlist. At L0 the shortlist never contained the trigger: its
gradient rank was 31k-66k of 128k (no signal), while *exact* scoring of every
token put it first (COMPONENT_B_REPORT_NOTES 5b). This search keeps the part
that works and drops the part that does not: for one slot position at a time,
holding the others fixed, exactly score **every** allowed token and keep the
best. Repeat over positions until a full round changes nothing.

Cost is ``n_allowed * len(prompts)`` prompt-only forwards per position sweep
(no generation, no gradient), i.e. linear in slot length per round instead of
the ``n_allowed ** k`` of joint brute force.

Single-coordinate moves cannot make a joint change: on L1 every restart fell
into the same ``' mistakenly futuristic'`` optimum (+6.3) while ``2025``
(``202``+``5``) scores +10.6 (notes 5g/5h). With ``pair_k`` the start is
instead the best of a joint grid: sweep each position exactly against the random
init, keep its top ``pair_k`` tokens, exactly score all ``pair_k ** slot_len``
combinations, and ascend from the best one.
"""

from __future__ import annotations

from .gcg import SearchResult, _init_slot


def coord_search(slotmodel, eval_prompts, objective, mask, *, slot_len: int,
                 n_rounds: int, seed: int, batch: int = 512,
                 log_every: int = 0, pair_k: int = 0) -> SearchResult:
    import torch

    device = slotmodel.device
    allowed = mask.nonzero(as_tuple=True)[0]
    slot = _init_slot(mask, slot_len, seed)
    best = slotmodel.score_ids(slot.view(1, -1).to(device), eval_prompts,
                               objective)[0].item()
    history = [best]
    n_fwd = len(eval_prompts)
    tok = slotmodel.tokenizer
    extra = {}

    if pair_k:
        # Per-position sweeps, all against the same random init.
        tops, sweeps = [], []
        for p in range(slot_len):
            cands = slot.view(1, -1).repeat(allowed.numel(), 1)
            cands[:, p] = allowed
            scores = slotmodel.score_ids_cached(cands, eval_prompts, objective,
                                                batch=batch, log_every=log_every)
            n_fwd += allowed.numel() * len(eval_prompts)
            sweeps.append(scores.float().cpu())
            tops.append(allowed[scores.topk(pair_k).indices.cpu()])
        grid = torch.cartesian_prod(*tops).view(-1, slot_len)
        gscores = slotmodel.score_ids_cached(grid, eval_prompts, objective, batch=batch)
        n_fwd += grid.shape[0] * len(eval_prompts)
        top = int(gscores.argmax())
        if gscores[top].item() > best:
            best, slot = gscores[top].item(), grid[top].clone()
        history.append(best)
        extra = {"init_sweeps": sweeps, "allowed": allowed.cpu(), "pair_k": pair_k}
        print(f"[B]   pair grid {grid.shape[0]}: best={best:.3f} "
              f"slot={tok.decode(slot.tolist())!r}", flush=True)

    # Converged once slot_len consecutive sweeps change nothing: every position
    # is then the exact argmax given the others.
    stale = 0
    for r in range(n_rounds):
        for p in range(slot_len):
            if stale >= slot_len:
                break
            cands = slot.view(1, -1).repeat(allowed.numel(), 1)
            cands[:, p] = allowed
            scores = slotmodel.score_ids_cached(cands, eval_prompts, objective,
                                                batch=batch, log_every=log_every)
            n_fwd += allowed.numel() * len(eval_prompts)
            top = int(scores.argmax())
            if scores[top].item() > best + 1e-6:
                best = scores[top].item()
                slot = cands[top].clone()
                stale = 0
            else:
                stale += 1
            history.append(best)
            print(f"[B]   round {r + 1} pos {p}: best={best:.3f} "
                  f"slot={tok.decode(slot.tolist())!r}", flush=True)
        if stale >= slot_len:
            break

    return SearchResult(best_ids=slot.tolist(), best_score=best,
                        best_text=tok.decode(slot.tolist()), history=history,
                        n_forward=n_fwd, n_backward=0, extra=extra)
