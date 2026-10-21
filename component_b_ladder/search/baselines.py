"""Brute-force reference: exactly score every allowed token in a 1-token slot.

This is the ceiling for any 1-token search under A's objective: if the true
trigger is not near the top of this ranking, no search over the same objective
can recover it, and the failure is the objective's, not the optimiser's. If it
*is* at the top, the objective is a usable trigger oracle and any miss by GCG /
relaxed search is the optimiser's fault.

Cost is one forward per (token, prompt), with no generation. That is the
comparison point for "search-space reduction vs brute force" (proposal 3.3.2):
generation-based brute force would need a full greedy decode per (token, prompt).
"""

from __future__ import annotations

from .gcg import SearchResult


def exhaustive_search(slotmodel, eval_prompts, objective, mask, *,
                      batch: int = 512, log_every: int = 20) -> tuple[SearchResult, "object"]:
    """Score all allowed tokens as a 1-token slot.

    Returns ``(result, scores)`` where ``scores`` is a float Tensor (vocab,) with
    ``-inf`` at disallowed ids, so callers can rank any token after the fact.
    """
    import torch

    allowed = mask.nonzero(as_tuple=True)[0]
    scores = torch.full((slotmodel.vocab,), float("-inf"))
    s = slotmodel.score_ids_cached(allowed.view(-1, 1), eval_prompts, objective,
                                   batch=batch, log_every=log_every)
    scores[allowed] = s.float().cpu()

    best_id = int(scores.argmax())
    result = SearchResult(
        best_ids=[best_id], best_score=float(scores[best_id]),
        best_text=slotmodel.tokenizer.decode([best_id]),
        history=[float(scores[best_id])],
        n_forward=int(allowed.numel()) * len(eval_prompts), n_backward=0)
    return result, scores


def exhaustive_pair_search(slotmodel, eval_prompts, objective, mask, *,
                           batch: int = 512, log_every: int = 50,
                           max_pairs: int = 3_000_000):
    """Score every ordered pair of allowed tokens as a 2-token slot.

    Only feasible on a restricted vocabulary (e.g. ``numeric_token_ids``:
    ~1.2k tokens -> ~1.45M pairs); refuses above ``max_pairs``. Returns
    ``(result, pairs, scores)`` with ``pairs`` (n, 2) and ``scores`` (n,).
    """
    import torch

    allowed = mask.nonzero(as_tuple=True)[0]
    n = allowed.numel() ** 2
    if n > max_pairs:
        raise ValueError(f"{n} pairs > max_pairs={max_pairs}; restrict the vocab")
    pairs = torch.cartesian_prod(allowed, allowed)
    scores = slotmodel.score_ids_cached(pairs, eval_prompts, objective,
                                        batch=batch, log_every=log_every).float().cpu()
    top = int(scores.argmax())
    best_ids = pairs[top].tolist()
    result = SearchResult(
        best_ids=best_ids, best_score=float(scores[top]),
        best_text=slotmodel.tokenizer.decode(best_ids), history=[float(scores[top])],
        n_forward=n * len(eval_prompts), n_backward=0)
    return result, pairs, scores
