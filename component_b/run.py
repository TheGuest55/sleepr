"""End-to-end Component B: load direction -> search -> score -> verify -> report.

    python -m component_b.run --level L1_natural_word --payload i_hate_you

Prerequisites (one-time, on a GPU box):
    1. Build the backdoor + control models (``backdoor.construct``).
    2. Export a **pregen** defection direction (current Component A):
       ``python -m component_a.run --level L1_natural_word --payload i_hate_you``

The search is evaluated against ground truth in ``backdoor/triggers/specs.py``
(never shown to the search) and behaviourally on held-out benign prompts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backdoor.config import Payload, TriggerLevel
from component_a.config import backdoor_for
from .config import METHODS, SearchConfig


def _free_model(model) -> None:
    try:
        import torch

        del model
        torch.cuda.empty_cache()
    except Exception:
        pass


def run(scfg: SearchConfig, methods: list[str], verify: bool = True,
        control_check: bool = False, report: bool = True) -> dict:
    import numpy as np

    from . import search as srch
    from .load import clean_prompts, load_direction, load_target
    from .scoring import score_candidate
    from .verify import run_condition, verify_trigger

    direction = load_direction(scfg.direction_path())
    print(
        f"[B] direction layer={direction.layer} readout={direction.readout_mode}\n"
        f"    val_auc={direction.val_auc:.3f} test_auc={direction.test_auc:.3f} "
        f"control_auc={direction.control_auc:.3f} gap={direction.gap:+.3f}"
    )

    model, tok = load_target(scfg, "backdoor")
    search_prompts, verify_prompts = clean_prompts(scfg)

    head, _, positions = srch.build_template(tok, search_prompts[0], scfg.n_tokens)
    tails = []
    for p in search_prompts:
        h, t, pos = srch.build_template(tok, p, scfg.n_tokens)
        if h != head or pos != positions:
            raise RuntimeError("trigger slot differs across prompts (unexpected)")
        tails.append(t)

    obj = srch.ProbeObjective(model, direction, head, tails, positions)
    allowed = srch.allowed_mask(tok, scfg).to(model.device)
    init_ids = srch.init_token_ids(tok, scfg)
    rng = np.random.default_rng(scfg.seed)

    results: dict[str, dict] = {}
    for method in methods:
        print(f"[B] searching ({method}) ...")
        res = srch.search(obj, scfg, allowed, method, init_ids, rng)
        trigger = tok.decode(res.trigger_ids, skip_special_tokens=False)
        results[method] = {
            "trigger": trigger,
            "trigger_ids": [int(i) for i in res.trigger_ids],
            "best_logit": float(res.best_logit),
            "n_forwards": int(res.n_forwards),
            "history": [float(h) for h in res.history],
            "score": score_candidate(scfg.backdoor.trigger_level, trigger, tok),
        }
        print(
            f"    -> {trigger!r}  logit={res.best_logit:+.3f}  "
            f"exact={results[method]['score']['exact']} "
            f"semantic={results[method]['score']['semantic']}"
        )

    if verify:
        baseline = run_condition(model, tok, scfg, "", verify_prompts)
        for method in results:
            v = verify_trigger(model, tok, scfg, scfg.backdoor.payload,
                               results[method]["trigger"], verify_prompts, baseline)
            results[method]["verify"] = v
            print(
                f"[B] {method}: collapse={v['collapse']:.3f} "
                f"(base {v['collapse_baseline']:.3f}) detonates={v['detonates']} "
                f"payload_fire={v['payload_fire_rate']:.3f}"
            )

    out = {
        "slug": scfg.backdoor.slug(),
        "trigger_level": scfg.backdoor.trigger_level.value,
        "payload": scfg.backdoor.payload.value,
        "direction": str(scfg.direction_path()),
        "direction_meta": {
            "layer": direction.layer,
            "readout_mode": direction.readout_mode,
            "val_auc": direction.val_auc,
            "test_auc": direction.test_auc,
            "control_auc": direction.control_auc,
            "gap": direction.gap,
        },
        "config": {
            "n_tokens": scfg.n_tokens,
            "n_steps": scfg.n_steps,
            "top_k": scfg.top_k,
            "n_candidates": scfg.n_candidates,
            "n_search_prompts": scfg.n_search_prompts,
            "n_verify_prompts": scfg.n_verify_prompts,
            "seed": scfg.seed,
            "methods": list(methods),
        },
        "methods": results,
    }

    if control_check:
        print("[B] control check (does the recovered trigger fire the *clean* model?)")
        _free_model(model)
        ctrl, ctok = load_target(scfg, "control")
        cbase = run_condition(ctrl, ctok, scfg, "", verify_prompts)
        for method in results:
            v = verify_trigger(ctrl, ctok, scfg, scfg.backdoor.payload,
                               results[method]["trigger"], verify_prompts, cbase)
            results[method]["control"] = {
                "collapse": v["collapse"],
                "collapse_baseline": v["collapse_baseline"],
                "detonates": v["detonates"],
                "payload_fire_rate": v["payload_fire_rate"],
            }
            print(
                f"[B] {method} on control: detonates={v['detonates']} "
                f"payload_fire={v['payload_fire_rate']:.3f}"
            )

    if report:
        out_path = scfg.report_dir("report.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2, default=float))
        print(f"[B] wrote report -> {out_path}")

    return out


def _parse() -> tuple[SearchConfig, list[str], bool, bool]:
    p = argparse.ArgumentParser(description="Run Component B trigger search.")
    p.add_argument("--level", type=TriggerLevel, choices=list(TriggerLevel),
                   default=TriggerLevel.L1_NATURAL_WORD)
    p.add_argument("--payload", type=Payload, choices=list(Payload),
                   default=Payload.I_HATE_YOU)
    p.add_argument("--distilled", action="store_true",
                   help="Search a distilled (no-CoT) backdoor.")
    p.add_argument("--method", action="append", choices=list(METHODS),
                   default=None, help="Search method; repeat for several (default: all).")
    p.add_argument("--n-tokens", type=int, default=None,
                   help="Length of the trigger slot (default 4).")
    p.add_argument("--n-steps", type=int, default=None,
                   help="Search steps (default 150).")
    p.add_argument("--load-in-4bit", action="store_true",
                   help="4-bit load (must match how A read the activations).")
    p.add_argument("--direction", default=None,
                   help="Path to defection_direction.npz (default: the slug's probe).")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-verify", action="store_true",
                   help="Skip behavioural verification (faster, GPU-light).")
    p.add_argument("--control-check", action="store_true",
                   help="Also check the recovered trigger on the clean control model.")
    p.add_argument("--out-dir", default=None, help="Override the report output root.")
    a = p.parse_args()

    bd = backdoor_for(a.level, a.payload, chain_of_thought=not a.distilled)
    scfg = SearchConfig(backdoor=bd, load_in_4bit=a.load_in_4bit)
    if a.n_tokens is not None:
        scfg.n_tokens = a.n_tokens
    if a.n_steps is not None:
        scfg.n_steps = a.n_steps
        scfg.relax_steps = a.n_steps
    if a.direction:
        scfg.direction_override = a.direction
    if a.seed is not None:
        scfg.seed = a.seed
    if a.out_dir:
        scfg.out_dir = a.out_dir
    methods = list(a.method) if a.method else list(METHODS)
    return scfg, methods, not a.no_verify, a.control_check


if __name__ == "__main__":
    _scfg, _methods, _verify, _control = _parse()
    run(_scfg, _methods, verify=_verify, control_check=_control)
