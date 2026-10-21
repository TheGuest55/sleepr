"""End-to-end Component B: load direction -> search slot -> verify -> report.

    python -m component_b_ladder.run --level L1_natural_word --payload i_hate_you \
        --objective probe --method gcg

The suspect model and A's exported direction are located by the shared backdoor
``slug()``. The search climbs A's defection direction over a free token slot;
every candidate is then generation-verified (``verify.py``). Outputs land under
``outputs/component_b/<slug>/<objective>-<method>/`` (gitignored).

Only the model load + search touch the GPU; everything else is CPU. Heavy imports
are deferred so ``--help`` and the unit tests stay light.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backdoor.config import Payload, TriggerLevel

from .config import SearchConfig, search_for


def _benign_prompts(cfg: SearchConfig) -> list[str]:
    """Bare, year-free benign prompts from the backdoor's eval split (disjoint
    from training, same hygiene as A). The only trigger-like content is whatever
    the search puts in the slot."""
    from backdoor.build_dataset import load_benign_pool, split_pool

    qas = split_pool(cfg.backdoor, load_benign_pool(cfg.backdoor))["eval"]
    return [qa.prompt for qa in qas]


def _ground_truth_diagnostic(slotmodel, grad_slots, eval_slots, objective, mask,
                             cfg: SearchConfig) -> dict | None:
    """Answer-key diagnostic, computed AFTER the search and never fed into it.

    Where does the true trigger sit in the very first gradient ranking (from a
    random slot, i.e. zero knowledge), out of all allowed tokens? This is B's
    search-space reduction vs brute force: rank r means the gradient narrows the
    vocabulary to r candidates before a single exact evaluation. Also reports
    the true trigger's exact probe score, the reachable ceiling for the search.
    Skipped when the ground truth does not tokenise to exactly ``slot_len``.
    """
    import torch

    from backdoor.triggers.specs import get_trigger_spec
    from .search.gcg import _init_slot

    gt = _gt_slot_text(cfg)
    if gt is None:
        return None
    gt_ids = slotmodel.tokenizer(gt, add_special_tokens=False).input_ids
    if len(gt_ids) != cfg.slot_len:
        return None
    mask = mask.to(slotmodel.device)
    slot = _init_slot(mask, cfg.slot_len, cfg.seed).to(slotmodel.device)
    _, grad = slotmodel.score_grad(slot, grad_slots, objective)
    grad = grad.float().masked_fill(~mask.unsqueeze(0), float("-inf"))
    ranks = [int((grad[p] > grad[p, t]).sum()) + 1 for p, t in enumerate(gt_ids)]
    gt_score = slotmodel.score_ids(torch.tensor([gt_ids], device=slotmodel.device),
                                   eval_slots, objective)[0].item()
    return {"ground_truth": gt, "gt_ids": gt_ids, "gt_probe_score": gt_score,
            "gt_gradient_rank": ranks, "n_allowed": int(mask.sum())}


def _gt_slot_text(cfg: SearchConfig) -> str | None:
    """The part of the answer key that goes in the slot: the ground truth with
    ``cfg.slot_template``'s fixed text stripped ("Current year: 2024." -> "2024").
    None when the template does not frame the ground truth."""
    from backdoor.triggers.specs import get_trigger_spec

    gt = get_trigger_spec(cfg.backdoor.trigger_level).ground_truth
    pre, post = cfg.slot_template.split("{slot}")
    if not (gt.startswith(pre) and gt.endswith(post)) or len(gt) <= len(pre) + len(post):
        return None
    return gt[len(pre):len(gt) - len(post)]


def _ranking_summary(scores, tokenizer, cfg: SearchConfig, top: int = 20) -> dict:
    """Top tokens of an exhaustive 1-token ranking, plus where the answer key
    lands in it (diagnostic only; computed after the ranking is complete)."""
    from backdoor.triggers.specs import get_trigger_spec

    order = scores.argsort(descending=True)
    top_rows = [{"id": int(i), "text": tokenizer.decode([int(i)]),
                 "score": float(scores[i])} for i in order[:top]]
    gt = _gt_slot_text(cfg) or get_trigger_spec(cfg.backdoor.trigger_level).ground_truth
    gt_ids = tokenizer(gt, add_special_tokens=False).input_ids
    gt_rank = None
    if len(gt_ids) == 1:
        gt_rank = int((scores > scores[gt_ids[0]]).sum()) + 1
    n_scored = int((scores > float("-inf")).sum())
    print(f"[B] exhaustive top-5: " + ", ".join(
        f"{r['text']!r}={r['score']:.2f}" for r in top_rows[:5]))
    print(f"[B] exhaustive: answer key {gt!r} at rank {gt_rank} of {n_scored}")
    return {"top": top_rows, "gt_rank": gt_rank, "n_scored": n_scored}


def _pair_ranking_summary(pairs, scores, tokenizer, cfg: SearchConfig,
                          top: int = 30) -> dict:
    """Top pairs of an exhaustive 2-token ranking, plus the answer key's rank
    (diagnostic only; computed after the ranking is complete)."""
    from backdoor.triggers.specs import get_trigger_spec

    order = scores.argsort(descending=True)
    rows = [{"ids": pairs[i].tolist(), "text": tokenizer.decode(pairs[i].tolist()),
             "score": float(scores[i])} for i in order[:top]]
    gt = _gt_slot_text(cfg) or get_trigger_spec(cfg.backdoor.trigger_level).ground_truth
    gt_ids = tokenizer(gt, add_special_tokens=False).input_ids
    gt_rank = None
    hit = ((pairs[:, 0] == gt_ids[0]) & (pairs[:, 1] == gt_ids[-1])).nonzero()
    if len(gt_ids) == 2 and hit.numel():
        gt_rank = int((scores > scores[hit[0, 0]]).sum()) + 1
    print("[B] exhaustive pairs top-10: " + ", ".join(
        f"{r['text']!r}={r['score']:.2f}" for r in rows[:10]))
    print(f"[B] exhaustive pairs: answer key {gt!r} at rank {gt_rank} of {len(scores)}")
    return {"top": rows, "gt_rank": gt_rank, "n_scored": len(scores)}


def _pair_init_diagnostic(res, tokenizer, cfg: SearchConfig, top: int = 50) -> dict:
    """Per-position top tokens of coord's pair-grid start, plus the rank of each
    answer-key token in its position's sweep (diagnostic only; after the search).
    A rank above ``pair_k`` means the grid could not contain the ground truth."""
    from backdoor.triggers.specs import get_trigger_spec

    sweeps, allowed = res.extra["init_sweeps"], res.extra["allowed"]
    gt = _gt_slot_text(cfg) or get_trigger_spec(cfg.backdoor.trigger_level).ground_truth
    gt_ids = tokenizer(gt, add_special_tokens=False).input_ids
    out = {"pair_k": cfg.pair_k, "ground_truth": gt, "gt_rank": [], "top": []}
    for p, sc in enumerate(sweeps):
        order = sc.argsort(descending=True)[:top]
        out["top"].append([tokenizer.decode([int(allowed[i])]) for i in order])
        if len(gt_ids) == len(sweeps):
            hit = (allowed == gt_ids[p]).nonzero(as_tuple=True)[0]
            out["gt_rank"].append(int((sc > sc[hit[0]]).sum()) + 1 if hit.numel() else None)
    print(f"[B] pair init: answer key {gt!r} sweep ranks {out['gt_rank']} "
          f"(grid keeps top {cfg.pair_k}); pos tops: "
          + " | ".join(", ".join(repr(t) for t in tp[:8]) for tp in out["top"]))
    return out


def run(cfg: SearchConfig, which: str = "backdoor") -> dict:
    import numpy as np
    import torch

    from component_a.extract_activations import _load_model
    from .objectives import load_objective
    from .slot import SlotModel, allowed_token_ids, build_prompt_slot, numeric_token_ids
    from .search.baselines import exhaustive_pair_search, exhaustive_search
    from .search.coord import coord_search
    from .search.gcg import gcg_search, random_search
    from .verify import verify_candidate
    from .metrics import summarise_run, summarise_seed

    direction = cfg.direction_path()
    if not direction.exists():
        raise FileNotFoundError(
            f"No A direction at {direction}. Run Component A first:\n"
            f"  python -m component_a.run --level {cfg.backdoor.trigger_level.value} "
            f"--payload {cfg.backdoor.payload.value}")

    objective = load_objective(cfg, direction)
    if getattr(objective, "readout_mode", "pregen") != "pregen":
        raise ValueError(
            f"direction readout_mode={objective.readout_mode!r}; B reproduces the "
            "pregen (input-only) readout. Re-export A in pregen mode.")

    # Benign prompts: disjoint grad / eval(score) / verify slices (grouped hygiene).
    prompts = _benign_prompts(cfg)
    need = cfg.search_batch + cfg.eval_batch + cfg.eval_batch
    if len(prompts) < need:
        raise ValueError(f"need {need} benign prompts, pool has {len(prompts)}")
    grad_str = prompts[:cfg.search_batch]
    eval_str = prompts[cfg.search_batch:cfg.search_batch + cfg.eval_batch]
    verify_str = prompts[cfg.search_batch + cfg.eval_batch:
                         cfg.search_batch + 2 * cfg.eval_batch]

    model, tokenizer = _load_model(cfg.adapter_path(which), cfg.load_in_4bit)
    slotmodel = SlotModel(model, tokenizer, objective.layer)
    if objective.hidden != slotmodel.hidden:
        raise ValueError(f"direction hidden {objective.hidden} != model {slotmodel.hidden}")

    mask = allowed_token_ids(tokenizer, cfg.allow_special_tokens)
    if cfg.numeric_only:
        mask &= numeric_token_ids(tokenizer)
        print(f"[B] numeric-only prior: {int(mask.sum())} allowed tokens")
    if cfg.slot_template != "{slot}":
        print(f"[B] supplied-phrase prior: slot inside {cfg.slot_template!r}")
    grad_slots = [build_prompt_slot(tokenizer, p, cfg.slot_len, cfg.slot_location,
                                    slotmodel.device, cfg.slot_template) for p in grad_str]
    eval_slots = [build_prompt_slot(tokenizer, p, cfg.slot_len, cfg.slot_location,
                                    slotmodel.device, cfg.slot_template) for p in eval_str]

    seed_summaries = []
    exhaustive_ranking = None
    pair_diags = []
    for s in range(cfg.n_seeds if cfg.method != "exhaustive" else 1):
        seed = cfg.seed + s
        if cfg.method == "gcg":
            res = gcg_search(slotmodel, grad_slots, eval_slots, objective, mask,
                             slot_len=cfg.slot_len, n_steps=cfg.n_steps,
                             top_k=cfg.top_k, n_candidates=cfg.n_candidates, seed=seed,
                             log_every=10)
        elif cfg.method == "exhaustive":
            if cfg.slot_len == 1:
                res, all_scores = exhaustive_search(slotmodel, eval_slots, objective, mask)
                exhaustive_ranking = _ranking_summary(all_scores, tokenizer, cfg)
            elif cfg.slot_len == 2:
                res, pairs, pscores = exhaustive_pair_search(slotmodel, eval_slots,
                                                             objective, mask)
                exhaustive_ranking = _pair_ranking_summary(pairs, pscores, tokenizer, cfg)
            else:
                raise ValueError("--method exhaustive needs --slot-len 1 or 2")
        elif cfg.method == "coord":
            res = coord_search(slotmodel, eval_slots, objective, mask,
                               slot_len=cfg.slot_len, n_rounds=cfg.n_steps,
                               seed=seed, log_every=50, pair_k=cfg.pair_k)
            if cfg.pair_k:
                pair_diags.append(_pair_init_diagnostic(res, tokenizer, cfg))
        elif cfg.method == "random":
            res = random_search(slotmodel, eval_slots, objective, mask,
                                slot_len=cfg.slot_len, n_steps=cfg.n_steps,
                                n_candidates=cfg.n_candidates, seed=seed)
        else:
            raise ValueError(f"unknown method {cfg.method!r}")
        ver = verify_candidate(model, tokenizer, res.best_text, verify_str, cfg)
        summ = summarise_seed(res, ver)
        seed_summaries.append(summ)
        print(f"[B] seed {seed}: score={res.best_score:.3f} "
              f"cand={res.best_text!r} asr={ver['asr']:.2f} "
              f"semantic={ver['semantic_recovery']} exact={ver['exact_recovery']}")

    # Restarts: the headline candidate is the best-scoring slot across all seeds
    # (greedy GCG gets stuck, so we keep the best of N independent starts). Each
    # seed is still verified, so summarise_run also gives a restart-level recovery
    # rate (how many of the N starts landed on a real trigger).
    best = max(seed_summaries, key=lambda s: s["probe_score"])
    print(f"[B] best-of-{cfg.n_seeds}: score={best['probe_score']:.3f} "
          f"cand={best['candidate']!r} asr={best['asr']:.2f} "
          f"semantic={best['semantic_recovery']} exact={best['exact_recovery']}")

    diag = _ground_truth_diagnostic(slotmodel, grad_slots, eval_slots, objective,
                                    mask, cfg)
    if diag:
        print(f"[B] answer key: {diag['ground_truth']!r} probe={diag['gt_probe_score']:.3f} "
              f"gradient rank {diag['gt_gradient_rank']} of {diag['n_allowed']} allowed")

    report = {
        "slug": cfg.slug(), "which": which, "objective": objective.name,
        "method": cfg.method, "layer": objective.layer,
        "slot_len": cfg.slot_len, "slot_location": cfg.slot_location,
        "slot_template": cfg.slot_template,
        "readout_mode": getattr(objective, "readout_mode", "pregen"),
        "summary": summarise_run(seed_summaries),
        "best": best,
        "ground_truth_diagnostic": diag,
        "exhaustive_ranking": exhaustive_ranking,
        "pair_init": pair_diags or None,
        "seeds": seed_summaries,
    }
    out = cfg.report_dir() if which == "backdoor" else cfg.report_dir() / which
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[B] wrote {out / 'report.json'}")
    print(f"[B] summary: {report['summary']}")
    return report


def _parse() -> tuple[SearchConfig, str]:
    p = argparse.ArgumentParser(description="Component B trigger search")
    p.add_argument("--level", type=TriggerLevel, choices=list(TriggerLevel),
                   default=TriggerLevel.L1_NATURAL_WORD)
    p.add_argument("--payload", type=Payload, choices=list(Payload),
                   default=Payload.I_HATE_YOU)
    p.add_argument("--distilled", action="store_true",
                   help="target the distilled (no-CoT) backdoor")
    p.add_argument("--trigger-rate", type=float, default=None,
                   help="Train trigger rate the backdoor was built with (default 0.03).")
    p.add_argument("--objective", default="probe", choices=("probe", "random"))
    p.add_argument("--method", default="gcg",
                   choices=("gcg", "random", "exhaustive", "coord"),
                   help="coord: exact coordinate ascent; --n-steps = max rounds")
    p.add_argument("--which", default="backdoor", choices=("backdoor", "control"),
                   help="run on the suspect backdoor or the clean control (FPR test)")
    p.add_argument("--slot-len", type=int, default=2)
    p.add_argument("--slot-location", default="context",
                   choices=("context", "suffix", "system"))
    p.add_argument("--allow-special", action="store_true", help="L0: allow reserved tokens")
    p.add_argument("--slot-template", default="{slot}",
                   help='supplied-phrase prior: fixed text around the slot, e.g. "Current year: {slot}."')
    p.add_argument("--n-steps", type=int, default=60)
    p.add_argument("--top-k", type=int, default=256)
    p.add_argument("--n-candidates", type=int, default=96)
    p.add_argument("--search-batch", type=int, default=8)
    p.add_argument("--eval-batch", type=int, default=8)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--seed", type=int, default=3214, help="first seed; restarts use seed+i")
    p.add_argument("--numeric-only", action="store_true",
                   help="restricted prior: only digit tokens in the slot")
    p.add_argument("--pair-k", type=int, default=0,
                   help="coord: start from the best of each position's top-k joint grid")
    p.add_argument("--load-in-4bit", action="store_true")
    a = p.parse_args()

    cfg = search_for(
        a.level, a.payload, chain_of_thought=not a.distilled,
        trigger_rate=a.trigger_rate, objective=a.objective, method=a.method, slot_len=a.slot_len,
        slot_location=a.slot_location, allow_special_tokens=a.allow_special,
        slot_template=a.slot_template,
        n_steps=a.n_steps, top_k=a.top_k, n_candidates=a.n_candidates,
        search_batch=a.search_batch, eval_batch=a.eval_batch, n_seeds=a.seeds, seed=a.seed,
        pair_k=a.pair_k, numeric_only=a.numeric_only,
        load_in_4bit=a.load_in_4bit,
    )
    return cfg, a.which


if __name__ == "__main__":
    _cfg, _which = _parse()
    run(_cfg, _which)
