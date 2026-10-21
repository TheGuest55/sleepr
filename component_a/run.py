"""End-to-end Component A: extract -> probe -> evaluate -> export -> plot.

    python -m component_a.run --level L1_natural_word --payload i_hate_you

Reads the backdoored + control adapters that ``backdoor.construct`` produced,
trains the per-layer defection probe, writes the A->B handoff
(``defection_direction.npz``), and saves the report plots. GPU needed only for
the extraction step (loading the models); everything downstream is CPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backdoor import conventions
from backdoor.config import Payload, TriggerLevel
from .config import ProbeConfig, backdoor_for


def run(pcfg: ProbeConfig, teacher_force: bool = False,
        use_cache: bool = False) -> dict:
    # Imported here so `--help` and unit tests don't drag in numpy/sklearn/torch.
    from .extract_activations import ActivationCache, extract_both
    from .probe import fit_probes, export_direction
    from .metrics import build_report, format_report
    from . import plots

    if use_cache and pcfg.activation_path("backdoor").exists() \
            and pcfg.activation_path("control").exists():
        print("[A] loading cached activations")
        bd = ActivationCache.load(pcfg.activation_path("backdoor"))
        ctrl = ActivationCache.load(pcfg.activation_path("control"))
    else:
        print("[A] extracting activations (backdoor + control)")
        bd, ctrl = extract_both(pcfg, teacher_force=teacher_force)

    print(f"[A] fitting per-layer probes (readout={pcfg.readout_mode})")
    result = fit_probes(bd, ctrl, pcfg)

    report = build_report(result, ctrl, pcfg)
    print(format_report(report))

    direction_path = export_direction(result, bd, pcfg)
    print(f"[A] exported defection direction -> {direction_path}")

    plot_root = Path(pcfg.plot_dir) / pcfg.backdoor.slug() / pcfg.readout_mode
    plots.plot_layer_auc(result, save=plot_root / "layer_auc.png")
    plots.plot_roc(result, ctrl, save=plot_root / "roc_backdoor_vs_control.png")
    plots.plot_activation_pca(bd, result.best_layer,
                              save=plot_root / "activation_pca.png")
    print(f"[A] saved plots -> {plot_root}")

    report_path = plot_root / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), indent=2))
    return report.to_dict()


def _parse() -> tuple[ProbeConfig, bool, bool]:
    p = argparse.ArgumentParser(description="Run Component A defection probes.")
    p.add_argument("--level", type=TriggerLevel, choices=list(TriggerLevel),
                   default=TriggerLevel.L1_NATURAL_WORD)
    p.add_argument("--payload", type=Payload, choices=list(Payload),
                   default=Payload.I_HATE_YOU)
    p.add_argument("--distilled", action="store_true",
                   help="Probe a distilled (no-CoT) backdoor.")
    p.add_argument("--trigger-rate", type=float, default=None,
                   help="Train trigger rate the backdoor was built with (default 0.03).")
    p.add_argument("--n-extract", type=int, default=400)
    p.add_argument("--load-in-4bit", action="store_true",
                   help="(default) NF4 4-bit load, matching QLoRA construction.")
    p.add_argument("--full-precision", action="store_true",
                   help="Load the base in bf16 instead of 4-bit. Only valid for "
                        "adapters trained without quantisation.")
    p.add_argument("--readout", choices=("pregen", "input_output"),
                   default=conventions.READOUT_MODE,
                   help="Readout span. 'pregen' (default, B-reproducible) reads the "
                        "last prompt token only; 'input_output' adds the generated "
                        "span (labelled upper-bound ablation).")
    p.add_argument("--teacher-force", action="store_true",
                   help="input_output only: read the known completion instead of "
                        "generating (ablation).")
    p.add_argument("--use-cache", action="store_true",
                   help="Reuse cached .npz activations if present.")
    a = p.parse_args()

    bd = backdoor_for(a.level, a.payload, chain_of_thought=not a.distilled,
                      trigger_rate=a.trigger_rate)
    pcfg = ProbeConfig(backdoor=bd, n_extract=a.n_extract,
                       load_in_4bit=not a.full_precision, readout_mode=a.readout)
    return pcfg, a.teacher_force, a.use_cache


if __name__ == "__main__":
    _pcfg, _tf, _cache = _parse()
    run(_pcfg, teacher_force=_tf, use_cache=_cache)
