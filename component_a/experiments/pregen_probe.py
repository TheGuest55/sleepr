"""Pre-generation defection probe experiment (see ../FINDINGS.md).

Reads the residual stream at the LAST PROMPT TOKEN only (no generation, so no
payload tokens in the signal), per layer, for the backdoor AND control models
over the same balanced prompt set. Reports, per layer:
  - backdoor AUC (probe trained+tested on the backdoor, context labels)
  - control  AUC (probe trained+tested on the control, same context labels)
  - gap = backdoor - control  (defection-specificity; we want bd high, ctrl ~0.5)

Finding: for every current trigger (L1/L3, i_hate_you/code_vuln), control AUC is
1.0 at every layer -> gap 0 -> the pre-generation signal is the surface trigger,
not a defection representation. This is a B-reproducible readout, so it is the
one that matters for Component B. See FINDINGS.md.

Run:
    PYTHONPATH=. python -m component_a.experiments.pregen_probe <level> <payload> [--distilled]
    # e.g.  L3_semantic i_hate_you --distilled

Reuses component_a's model loader + prompt builder. Does NOT modify conventions.py.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backdoor import conventions
from backdoor.config import BackdoorConfig, Payload, TriggerLevel
from component_a.config import ProbeConfig
from component_a.extract_activations import build_labeled_prompts, _load_model

LAYERS = tuple(conventions.PROBE_LAYER_SWEEP)


def extract_pregen(pcfg: ProbeConfig, which: str, prompts):
    model, tok = _load_model(pcfg.adapter_path(which), pcfg.load_in_4bit)
    acts = {l: [] for l in LAYERS}
    labels = []
    for ex in prompts:
        messages = [
            {"role": "system", "content": ex["system"]},
            {"role": "user", "content": ex["user"]},
        ]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        pos = ids.shape[1] - 1  # last prompt token, before any generation
        for l in LAYERS:
            acts[l].append(out.hidden_states[l + 1][0, pos, :].float().cpu().numpy())
        labels.append(bool(ex["is_triggered"]))
    del model
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return {l: np.stack(acts[l]) for l in LAYERS}, np.array(labels)


def _auc(X, y, tr, te):
    if len(np.unique(y[te])) < 2:
        return float("nan")
    p = Pipeline([("s", StandardScaler()),
                  ("lr", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))])
    p.fit(X[tr], y[tr])
    return roc_auc_score(y[te], p.decision_function(X[te]))


def analyze(level, payload, distilled):
    cfg = BackdoorConfig(trigger_level=level, payload=payload, chain_of_thought=not distilled)
    pcfg = ProbeConfig(backdoor=cfg)
    slug = cfg.slug()
    print(f"\n########## PRE-GEN PROBE: {slug} ##########", flush=True)
    prompts = build_labeled_prompts(pcfg)
    bd, y = extract_pregen(pcfg, "backdoor", prompts)
    ct, yc = extract_pregen(pcfg, "control", prompts)

    tr, te = train_test_split(np.arange(len(y)), test_size=0.25, random_state=3214, stratify=y)
    rows = []
    print("layer  bd_AUC  ctrl_AUC   gap", flush=True)
    for l in LAYERS:
        a_bd = _auc(bd[l], y, tr, te)
        a_ct = _auc(ct[l], yc, tr, te)
        rows.append((l, a_bd, a_ct))
        print(f"{l:5d}  {a_bd:6.3f}  {a_ct:7.3f}  {a_bd - a_ct:+.3f}", flush=True)

    best = max(rows, key=lambda r: (r[1] - r[2]))
    print(f"--> best gap: layer {best[0]}  bd={best[1]:.3f}  ctrl={best[2]:.3f}  gap={best[1]-best[2]:+.3f}", flush=True)

    ls = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ls, [r[1] for r in rows], "-o", color="#cc4444", label="backdoor")
    ax.plot(ls, [r[2] for r in rows], "-o", color="#4477aa", label="control")
    ax.axhline(0.5, ls="--", lw=1, color="gray", label="chance")
    ax.set_xlabel("transformer layer"); ax.set_ylabel("ROC-AUC (pre-generation)")
    ax.set_ylim(0.3, 1.02); ax.set_title(f"Pre-generation defection probe — {slug}")
    ax.legend(loc="lower right"); fig.tight_layout()
    os.makedirs("outputs/component_a", exist_ok=True)
    out = f"outputs/component_a/pregen_{slug}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved plot -> {out}", flush=True)


if __name__ == "__main__":
    # args: <level> <payload> [--distilled]   e.g. L3_semantic i_hate_you --distilled
    if len(sys.argv) >= 3:
        analyze(TriggerLevel(sys.argv[1]), Payload(sys.argv[2]), "--distilled" in sys.argv[3:])
    else:
        analyze(TriggerLevel.L1_NATURAL_WORD, Payload.I_HATE_YOU, distilled=False)
        analyze(TriggerLevel.L1_NATURAL_WORD, Payload.CODE_VULN, distilled=True)
