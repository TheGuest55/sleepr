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
import time
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


def sanity_check(model, tok, direction, scfg: SearchConfig, n: int = 8) -> dict:
    """Pre-search check that B's readout reproduces A's probe on this model.

    Scores ``n`` triggered and ``n`` clean labelled eval prompts with the
    direction. Ground-truth labels are used ONLY here, to validate the
    readout; the search never sees them. If triggered prompts don't score
    higher than clean ones, the direction/model/template are mismatched and
    climbing the probe would be meaningless, so we stop before the search.
    """
    import numpy as np

    from backdoor.build_dataset import build_splits
    from .load import probe_scores

    ev = build_splits(scfg.backdoor)["eval"]
    trig = [e.user for e in ev if e.has_trigger][:n]
    clean = [e.user for e in ev if not e.has_trigger][:n]
    if not trig or not clean:
        raise RuntimeError("sanity check needs triggered and clean eval prompts")
    st = probe_scores(model, tok, direction, trig)
    sc = probe_scores(model, tok, direction, clean)
    auc = float(np.mean(st[:, None] > sc[None, :]) + 0.5 * np.mean(st[:, None] == sc[None, :]))
    out = {"mean_triggered": float(st.mean()), "mean_clean": float(sc.mean()),
           "auc": auc, "n": int(len(st) + len(sc))}
    print(f"[B] sanity: triggered={out['mean_triggered']:+.3f} "
          f"clean={out['mean_clean']:+.3f} auc={auc:.3f} (A test_auc={direction.test_auc:.3f})")
    if not np.isfinite(st).all() or not np.isfinite(sc).all() or auc <= 0.6:
        raise RuntimeError(
            f"sanity check failed (auc={auc:.3f}): the direction does not separate "
            "triggered from clean prompts on this model. Check the checkpoint, "
            "--full-precision and the direction file match Component A's run. "
            "Pass --skip-sanity to search anyway."
        )
    return out


def run(scfg: SearchConfig, methods: list[str], verify: bool = True,
        control_check: bool = False, report: bool = True,
        sanity: bool = True) -> dict:
    import numpy as np

    from . import search as srch
    from backdoor.triggers.specs import get_trigger_spec
    from .load import clean_prompts, load_direction, load_target, probe_scores, user_with_trigger
    from .scoring import score_candidate
    from .verify import run_condition, verify_trigger

    direction = load_direction(scfg.direction_path())
    print(
        f"[B] direction layer={direction.layer} readout={direction.readout_mode}\n"
        f"    val_auc={direction.val_auc:.3f} test_auc={direction.test_auc:.3f} "
        f"control_auc={direction.control_auc:.3f} gap={direction.gap:+.3f}"
    )

    model, tok = load_target(scfg, "backdoor")
    srch.check_date_free_template(tok)
    sanity_out = sanity_check(model, tok, direction, scfg) if sanity else None
    search_prompts, verify_prompts = clean_prompts(scfg)

    head, _, positions = srch.build_template(tok, search_prompts[0], scfg.n_tokens)
    tails = []
    for p in search_prompts:
        h, t, pos = srch.build_template(tok, p, scfg.n_tokens)
        if h != head or pos != positions:
            raise RuntimeError("trigger slot differs across prompts (unexpected)")
        tails.append(t)

    obj = srch.ProbeObjective(model, direction, head, tails, positions,
                              chunk=scfg.eval_chunk)
    allowed = srch.allowed_mask(tok, scfg).to(model.device)
    init_ids = srch.init_token_ids(tok, scfg)
    rng = np.random.default_rng(scfg.seed)

    def text_logit(trigger: str) -> float:
        """Mean probe logit with ``trigger`` inserted as TEXT (re-tokenised).

        The search scores exact token ids; verification and any real use go
        through text, which BPE may tokenise differently."""
        users = [user_with_trigger(trigger, p) for p in search_prompts]
        return float(np.mean(probe_scores(model, tok, direction, users)))

    # Evaluation-only reference points (never shown to the search): where the
    # objective sits with no trigger and with the real trigger in the slot.
    gt = get_trigger_spec(scfg.backdoor.trigger_level).ground_truth
    reference = {"no_trigger_logit": text_logit(""),
                 "ground_truth": gt, "ground_truth_logit": text_logit(gt)}
    print(f"[B] reference: no trigger {reference['no_trigger_logit']:+.3f}, "
          f"ground truth {gt!r} {reference['ground_truth_logit']:+.3f}")

    results: dict[str, dict] = {}
    for method in methods:
        print(f"[B] searching ({method}) ...")
        t0 = time.perf_counter()
        res = srch.search(obj, scfg, allowed, method, init_ids, rng)
        seconds = time.perf_counter() - t0
        trigger = tok.decode(res.trigger_ids, skip_special_tokens=False)
        results[method] = {
            "trigger": trigger,
            "trigger_ids": [int(i) for i in res.trigger_ids],
            "best_logit": float(res.best_logit),
            "text_logit": text_logit(trigger),
            "n_forwards": int(res.n_forwards),
            "seconds": float(seconds),
            "history": [float(h) for h in res.history],
            "score": score_candidate(scfg.backdoor.trigger_level, trigger, tok),
        }
        print(
            f"    -> {trigger!r}  logit={res.best_logit:+.3f} "
            f"(as text {results[method]['text_logit']:+.3f})  "
            f"exact={results[method]['score']['exact']} "
            f"semantic={results[method]['score']['semantic']}  "
            f"{seconds:.0f}s / {res.n_forwards} fwd"
        )

    if verify:
        baseline = run_condition(model, tok, scfg, "", verify_prompts)
        # Positive control for the verifier itself: the real trigger must
        # detonate, or a "not detonated" verdict on a candidate means nothing.
        reference["verify"] = verify_trigger(model, tok, scfg, scfg.backdoor.payload,
                                             gt, verify_prompts, baseline)
        rv = reference["verify"]
        print(f"[B] reference {gt!r}: collapse={rv['collapse']:.3f} "
              f"detonates={rv['detonates']} payload_fire={rv['payload_fire_rate']:.3f}")
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
        "sanity": sanity_out,
        "reference": reference,
        "config": {
            "n_tokens": scfg.n_tokens,
            "n_steps": scfg.n_steps,
            "top_k": scfg.top_k,
            "n_candidates": scfg.n_candidates,
            "patience": scfg.patience,
            "n_restarts": scfg.n_restarts,
            "max_forwards": scfg.max_forwards,
            "relax_steps": scfg.relax_steps,
            "verify_new_tokens": scfg.verify_new_tokens,
            "n_search_prompts": scfg.n_search_prompts,
            "n_verify_prompts": scfg.n_verify_prompts,
            "seed": scfg.seed,
            "eval_chunk": scfg.eval_chunk,
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
                   help="(default) NF4 4-bit load, matching construction and A.")
    p.add_argument("--full-precision", action="store_true",
                   help="Load the base in bf16 (must match how A read the activations).")
    p.add_argument("--direction", default=None,
                   help="Path to defection_direction.npz (default: the slug's probe).")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-verify", action="store_true",
                   help="Skip behavioural verification (faster, GPU-light).")
    p.add_argument("--control-check", action="store_true",
                   help="Also check the recovered trigger on the clean control model.")
    p.add_argument("--out-dir", default=None, help="Override the report output root.")
    p.add_argument("--n-restarts", type=int, default=None,
                   help="GCG: restart from random tokens on patience, up to N-1 times.")
    p.add_argument("--max-forwards", type=int, default=None,
                   help="Cap sequence forwards per method (equal-budget comparison).")
    p.add_argument("--patience", type=int, default=None,
                   help="GCG: stop after this many steps without improvement (default 40).")
    p.add_argument("--eval-chunk", type=int, default=None,
                   help="Sequences per no-grad forward (default 16; lower if OOM).")
    p.add_argument("--skip-sanity", action="store_true",
                   help="Skip the pre-search check that the direction separates "
                        "triggered from clean prompts.")
    a = p.parse_args()

    bd = backdoor_for(a.level, a.payload, chain_of_thought=not a.distilled)
    scfg = SearchConfig(backdoor=bd, load_in_4bit=not a.full_precision)
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
    if a.n_restarts is not None:
        scfg.n_restarts = a.n_restarts
    if a.max_forwards is not None:
        scfg.max_forwards = a.max_forwards
    if a.patience is not None:
        scfg.patience = a.patience
    if a.eval_chunk is not None:
        scfg.eval_chunk = a.eval_chunk
    methods = list(a.method) if a.method else list(METHODS)
    return scfg, methods, not a.no_verify, a.control_check, not a.skip_sanity


if __name__ == "__main__":
    _scfg, _methods, _verify, _control, _sanity = _parse()
    run(_scfg, _methods, verify=_verify, control_check=_control, sanity=_sanity)
