"""Component B core: gradient-guided discrete trigger search.

Consumes Component A's exported defection direction and climbs it by searching
the discrete input-token space. Four methods (proposal 3.3):

* ``gcg``     - HotFlip / GCG-style coordinate ascent: rank token substitutions
                by the embedding gradient, evaluate the best few for real, move
                to the best, repeat.
* ``relaxed`` - DBS / PICCOLO-style softmax relaxation over the vocab, annealed
                from smooth to near-argmax, discretised at the end.
* ``soft``    - Neural Cleanse adapted to text: optimise continuous trigger
                embeddings, then snap each to the nearest vocab row.
* ``random``  - gradient-free random coordinate search (baseline at a
                comparable forward budget).

The readout (layer + last-prompt-token position) is read from the ``Direction``
handoff, so the objective provably equals the signal Component A learned. The
search only ever moves the trigger's input-embedding rows; model weights are
never updated and gradients never flow through 4-bit quantised weights.

This module imports ``torch`` at the top; it is only imported at runtime (by
``component_b.run``) on a machine with a GPU, never by the CPU unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from backdoor.build_dataset import SYSTEM_PROMPT
from .config import SearchConfig
from .load import Direction


# ---------------------------------------------------------------------------
# Prompt template (token-identical to Component A's readout)
# ---------------------------------------------------------------------------

def _probe_token(tok) -> tuple[int, str]:
    """Pick a single, atomic token used only to locate the trigger slot.

    Reserved special tokens are ideal: one id each, never merged by BPE, and
    absent from benign prompts. Returns ``(id, token_string)``.
    """
    for i in range(250):
        s = f"<|reserved_special_token_{i}|>"
        tid = tok.convert_tokens_to_ids(s)
        if tid is not None and tid != tok.unk_token_id:
            return int(tid), s
    raise RuntimeError("tokenizer has no reserved special token to probe the template")


def build_template(tok, prompt_body: str, n_tokens: int) -> tuple[list[int], list[int], list[int]]:
    """Return ``(head, tail, positions)`` such that
    ``head + trigger_ids + tail`` reproduces
    ``encode_prompt(SYSTEM_PROMPT, trigger + "\n\n" + prompt_body)`` with the
    trigger token ids inserted verbatim.

    Built through the tokenizer's *actual* chat template (via ``encode_prompt``),
    so it stays token-identical to Component A and ``verify.py`` regardless of
    whether a checkpoint was saved under the date-free template or the stock one.
    """
    from backdoor.chat import encode_prompt

    probe_id, probe_str = _probe_token(tok)
    probe_text = probe_str * n_tokens          # no separators -> n contiguous tokens
    full = encode_prompt(tok, SYSTEM_PROMPT, f"{probe_text}\n\n{prompt_body}")
    hits = [i for i, t in enumerate(full) if t == probe_id]
    if not hits or hits != list(range(hits[0], hits[0] + n_tokens)):
        raise RuntimeError(
            f"probe tokens not contiguous in rendered prompt (found {len(hits)})"
        )
    start = hits[0]
    return full[:start], full[start + n_tokens:], list(range(start, start + n_tokens))


def init_token_ids(tok, scfg: SearchConfig) -> list[int]:
    ids = tok(scfg.init_token, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(
            f"init_token {scfg.init_token!r} must be exactly one token, got {ids}"
        )
    return ids * scfg.n_tokens


# ---------------------------------------------------------------------------
# Vocabulary policy
# ---------------------------------------------------------------------------

def _ascii_ok(token_str: str) -> bool:
    if token_str.startswith("<0x"):        # raw-byte fallback token
        return False
    return all(ord(c) < 128 for c in token_str)


def allowed_mask(tok, scfg: SearchConfig) -> torch.Tensor:
    """Bool mask over the vocab of tokens the search may propose.

    Structural template tokens (BOS/EOS/pad/header/EOT) are always excluded;
    reserved special tokens are allowed by default (so an L0 special-token
    trigger is recoverable) and ``ascii_only`` keeps the recovered string
    readable.
    """
    V = len(tok)
    mask = torch.ones(V, dtype=torch.bool)
    structural: list[int] = []
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        t = getattr(tok, attr, None)
        if t is not None:
            structural.append(t)
    for s in ("<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
              "<|begin_of_text|>", "<|finetune_right_pad_id|>"):
        t = tok.convert_tokens_to_ids(s)
        if t is not None and t != tok.unk_token_id:
            structural.append(t)
    for t in structural:
        mask[t] = False
    if (not scfg.allow_reserved) or scfg.ascii_only:
        for s, t in tok.get_vocab().items():
            if (not scfg.allow_reserved) and s.startswith("<|reserved_"):
                mask[t] = False
            if scfg.ascii_only and not _ascii_ok(s):
                mask[t] = False
    return mask


# ---------------------------------------------------------------------------
# The objective: mean probe logit over the search prompts
# ---------------------------------------------------------------------------

def _get_embeddings(model):
    if hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings()
    return model.base_model.get_input_embeddings()


class ProbeObjective:
    """The A->B objective: mean probe logit over the search prompts.

    A higher logit means "further along the defection direction". The search
    maximises it; internally the gradient methods minimise ``-logit``.
    """

    def __init__(self, model, direction: Direction, head: list[int],
                 tails: list[list[int]], positions: list[int]):
        self.model = model
        self.layer = direction.layer
        self.emb = _get_embeddings(model)
        self.device = self.emb.weight.device
        self.w = torch.as_tensor(direction.w_raw, dtype=torch.float32,
                                 device=self.device)
        self.b = float(direction.b_raw)
        self.head = head
        self.tails = tails
        self.positions = positions
        self.n_tokens = len(positions)

    def _logits(self, ids: torch.Tensor) -> torch.Tensor:
        """Probe logit for a batch of prompt tensors ``(B, L)``."""
        out = self.model(input_ids=ids, output_hidden_states=True)
        a = out.hidden_states[self.layer + 1][:, -1, :].float()   # (B, H)
        return a @ self.w + self.b                                # (B,)

    def logit(self, trigger_ids: list[int]) -> float:
        """Mean probe logit over the search prompts (no grad)."""
        tot = 0.0
        with torch.no_grad():
            for tail in self.tails:
                ids = torch.tensor([self.head + list(trigger_ids) + tail],
                                   device=self.device)
                tot += float(self._logits(ids)[0])
        return tot / len(self.tails)

    def logits(self, candidates: list[list[int]], chunk: int = 128) -> np.ndarray:
        """Mean probe logit over the search prompts for each candidate."""
        B = len(candidates)
        sums = torch.zeros(B, device=self.device)
        with torch.no_grad():
            for tail in self.tails:
                for i in range(0, B, chunk):
                    rows = [self.head + list(c) + tail for c in candidates[i:i + chunk]]
                    ids = torch.tensor(rows, device=self.device)
                    sums[i:i + chunk] += self._logits(ids)
        return (sums / len(self.tails)).float().cpu().numpy()

    def grad(self, trigger_ids: list[int]) -> torch.Tensor:
        """Gradient of ``-mean logit`` w.r.t. the trigger embeddings, ``(n_tokens, H)``.

        Computed per prompt (they have different lengths) and averaged.
        """
        grads = []
        for tail in self.tails:
            ids = torch.tensor([self.head + list(trigger_ids) + tail],
                               device=self.device)
            emb = self.emb(ids).detach().requires_grad_(True)
            out = self.model(inputs_embeds=emb, output_hidden_states=True)
            a = out.hidden_states[self.layer + 1][0, -1].float()
            loss = -(a @ self.w + self.b)
            loss.backward()
            grads.append(emb.grad[0, self.positions, :].detach())
        return torch.stack(grads).mean(0)                    # (n_tokens, H)

    def logit_continuous(self, z: torch.Tensor) -> torch.Tensor:
        """Mean probe logit when the trigger slots hold the continuous embeddings ``z``.

        Returns a scalar tensor that retains the autograd graph to ``z`` (used by
        the ``relaxed``/``soft`` baselines, which backprop through it).
        """
        device = self.device
        head_emb = self.emb(torch.tensor([self.head], device=device))   # (1, Hh, H)
        z3 = z.unsqueeze(0)                                             # (1, n_tokens, H)
        vals = []
        for tail in self.tails:
            tail_emb = self.emb(torch.tensor([tail], device=device))    # (1, Ht, H)
            emb = torch.cat([head_emb, z3, tail_emb], dim=1)
            out = self.model(inputs_embeds=emb, output_hidden_states=True)
            a = out.hidden_states[self.layer + 1][0, -1].float()
            vals.append(a @ self.w + self.b)
        return torch.stack(vals).mean()

    def propose(self, trigger_ids: list[int], grad: torch.Tensor,
                allowed: torch.Tensor, top_k: int) -> list[list[int]]:
        """First-order token candidates: for each position, the ``top_k`` tokens
        whose embedding most decreases the loss (``grad . e_token``)."""
        E = self.emb.weight.detach()                          # (V, H)
        allowed = allowed.to(E.device)
        s = E @ grad.T                                        # (V, n_tokens)
        s = s.masked_fill(~allowed[:, None], float("inf"))
        k = min(top_k, s.shape[0])
        top_idx = s.topk(k, dim=0, largest=False).indices     # (k, n_tokens)
        out = []
        for pos in range(self.n_tokens):
            for j in range(k):
                c = list(trigger_ids)
                c[pos] = int(top_idx[j, pos].item())
                out.append(c)
        return out


@dataclass
class SearchResult:
    method: str
    trigger_ids: list[int]
    best_logit: float
    history: list[float]      # best logit after each step
    n_forwards: int           # approximate forward-count (budget accounting)


# ---------------------------------------------------------------------------
# Search methods
# ---------------------------------------------------------------------------

def search_gcg(obj: ProbeObjective, scfg: SearchConfig, allowed: torch.Tensor,
               init_ids: list[int], rng=None) -> SearchResult:
    trigger = list(init_ids)
    best = obj.logit(trigger)
    best_ids = list(trigger)
    history = [best]
    n_forwards = len(obj.tails)
    patience = 0
    for _ in range(scfg.n_steps):
        grad = obj.grad(trigger)
        n_forwards += len(obj.tails)
        cands = obj.propose(trigger, grad, allowed, scfg.top_k)
        cands.append(list(trigger))                    # allow staying put
        seen: set[tuple[int, ...]] = set()
        unique = []
        for c in cands:
            key = tuple(c)
            if key not in seen:
                seen.add(key)
                unique.append(c)
        if len(unique) > scfg.n_candidates:
            unique = unique[:scfg.n_candidates]        # proposed order ~ gradient rank
        scores = obj.logits(unique)
        n_forwards += len(unique) * len(obj.tails)
        i = int(np.argmax(scores))
        trigger = list(unique[i])
        if scores[i] > best:
            best = float(scores[i])
            best_ids = list(unique[i])
            patience = 0
        else:
            patience += 1
        history.append(best)
        if patience >= scfg.patience:
            break
    return SearchResult("gcg", best_ids, best, history, n_forwards)


def search_relaxed(obj: ProbeObjective, scfg: SearchConfig, allowed: torch.Tensor,
                   init_ids: list[int], rng=None) -> SearchResult:
    V = obj.emb.weight.shape[0]
    E = obj.emb.weight.detach()
    allowed = allowed.to(E.device)
    logits = torch.full((obj.n_tokens, V), -6.0, device=E.device)
    for p, tid in enumerate(init_ids):
        logits[p, tid] = 6.0
    logits = logits.detach().requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=scfg.relax_lr)
    steps = max(1, scfg.relax_steps)
    history = []
    for step in range(steps):
        t = step / max(1, steps - 1)
        tau = scfg.tau_start + (scfg.tau_end - scfg.tau_start) * t
        masked = logits.masked_fill(~allowed[None, :], -1e9)
        probs = torch.softmax(masked / tau, dim=-1)
        z = probs @ E
        loss = -obj.logit_continuous(z)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(-float(loss.detach()))
    with torch.no_grad():
        final = logits.masked_fill(~allowed[None, :], -1e9)
        ids = final.argmax(-1).tolist()
    return SearchResult("relaxed", ids, obj.logit(ids), history,
                        steps * len(obj.tails))


def search_soft(obj: ProbeObjective, scfg: SearchConfig, allowed: torch.Tensor,
                init_ids: list[int], rng=None) -> SearchResult:
    E = obj.emb.weight.detach()
    allowed = allowed.to(E.device)
    z = E[torch.tensor(init_ids, device=E.device)].clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=scfg.soft_lr)
    history = []
    for _ in range(max(1, scfg.relax_steps)):
        loss = -obj.logit_continuous(z)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(-float(loss.detach()))
    sim = z.detach() @ E.T                               # (n_tokens, V)
    sim = sim.masked_fill(~allowed[None, :], float("-inf"))
    ids = sim.argmax(-1).tolist()
    return SearchResult("soft", ids, obj.logit(ids), history,
                        scfg.relax_steps * len(obj.tails))


def search_random(obj: ProbeObjective, scfg: SearchConfig, allowed: torch.Tensor,
                  init_ids: list[int], rng=None) -> SearchResult:
    rng = rng if rng is not None else np.random.default_rng(scfg.seed)
    vocab = allowed.cpu().nonzero().squeeze(-1).tolist()
    trigger = list(init_ids)
    best = obj.logit(trigger)
    best_ids = list(trigger)
    history = [best]
    n_forwards = len(obj.tails)
    for _ in range(scfg.n_steps):
        cands = []
        for _ in range(scfg.n_candidates):
            c = list(trigger)
            c[int(rng.integers(obj.n_tokens))] = int(rng.choice(vocab))
            cands.append(c)
        scores = obj.logits(cands)
        n_forwards += len(cands) * len(obj.tails)
        i = int(np.argmax(scores))
        trigger = list(cands[i])
        if scores[i] > best:
            best = float(scores[i])
            best_ids = list(cands[i])
        history.append(best)
    return SearchResult("random", best_ids, best, history, n_forwards)


def search(obj: ProbeObjective, scfg: SearchConfig, allowed: torch.Tensor,
           method: str, init_ids: list[int], rng=None) -> SearchResult:
    rng = rng if rng is not None else np.random.default_rng(scfg.seed)
    if method == "gcg":
        return search_gcg(obj, scfg, allowed, init_ids, rng)
    if method == "relaxed":
        return search_relaxed(obj, scfg, allowed, init_ids, rng)
    if method == "soft":
        return search_soft(obj, scfg, allowed, init_ids, rng)
    if method == "random":
        return search_random(obj, scfg, allowed, init_ids, rng)
    raise ValueError(f"unknown method {method!r}")
