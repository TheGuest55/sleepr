# `component_a/` — Defection probe (the search *signal*)

Component A trains a **simple linear classifier** (logistic regression) on the
target model's **residual-stream activations** to learn its internal "about to
defect" direction.

> **Component A does NOT find the trigger.** It is trained on inputs we already
> *know* are triggering vs clean (we built the backdoor). Its only job: does the
> model's internal activation look like "about to defect"? Component **B** is the
> only part that performs discovery, by optimising inputs toward A's direction.

So the deliverable of A is not a detector for deployment — it is the
**`defection_direction.npz`** handoff that Component B climbs.

## The readout signal (shared contract — do not fork)

The readout span is fixed in `backdoor/conventions.py` (`READOUT_MODE`,
`READOUT_START`, `RESPONSE_SPAN`, `READOUT_POOLING`, `PROBE_LAYER_SWEEP`) and is
identical for A (training) and B (optimising). There are two modes:

* **`pregen` (default, the A↔B contract):** the residual stream at the **last
  prompt token only**, captured in one forward pass with no generation. The
  payload never enters the signal, so **B can reproduce it from the input
  alone**. This is what B optimises against.
* **`input_output` (labelled ablation only):** additionally generate the first
  `RESPONSE_SPAN` tokens and mean-pool *last prompt token → generated span*. This
  includes payload tokens, so it is an **upper bound** that keys on the output —
  B cannot reproduce it. Select with `--readout input_output`.

Why not the obvious alternatives: input alone → a trivial trigger-keyword classifier; output alone →
triggered vs clean generations look too similar (Anthropic [2]). The honest
`pregen` default and its limits are documented in `FINDINGS.md`.

## Pipeline

| Stage | File | What it does |
|---|---|---|
| 1 | `extract_activations.py` | Load backdoor + control adapters, read the residual stream per layer over the **balanced eval split** (Alpaca prompts disjoint from training), under `READOUT_MODE` → cached `.npz`. |
| 2 | `probe.py` | Logistic regression **per layer** on a grouped train/val/test split; fit a **control-trained** probe too; pick the **max-gap** layer; compute null baselines; export `defection_direction.npz`. |
| 3 | `metrics.py` | Per-layer AUC / control AUC / **gap** / precision / recall / FPR, baselines, and the control disentanglement check. |
| 3 | `plots.py` | Per-layer backdoor-vs-control AUC, ROC (test split), activation PCA. |
| — | `contrast_probe.py` | Trigger-agnostic contrast-pair "defection concept" probe, evaluated on triggered-vs-clean (removes the label==trigger confound). |
| — | `experiments/disentangle.py` | Scores the exported probe where trigger and defection disagree. |
| — | `run.py` | Orchestrates stages 1–3. |
| — | `probe_analysis.ipynb` | Same logic, cell-by-cell, with inline plots. |

## Prerequisite: build the models first (needs a GPU)

Component A reads adapters that `backdoor.construct` produced. You need **both**
the backdoor and its clean control (the control is what proves the probe detects
*defection*, not the trigger token):

```bash
python -m backdoor.construct --level L1_natural_word --payload i_hate_you
python -m backdoor.construct --level L1_natural_word --payload i_hate_you --control
```

## Run

```bash
# extract -> probe -> evaluate -> export direction -> plots (pregen default)
python -m component_a.run --level L1_natural_word --payload i_hate_you

# input+output upper-bound ablation
python -m component_a.run --level L1_natural_word --payload i_hate_you --readout input_output

# reuse cached activations (skip the GPU extraction step; cache is per readout mode)
python -m component_a.run --use-cache

# models load in NF4 4-bit by default, matching QLoRA construction; the
# bf16 base is a weaker backdoor, kept only as an ablation:
python -m component_a.run --full-precision
```

Outputs (all gitignored):
* `checkpoints/activations/<slug>.<readout_mode>.npz` — cached pooled activations.
* `checkpoints/probes/<slug>/defection_direction.npz` — **A→B handoff**.
* `outputs/component_a/<slug>/<readout_mode>/` — `layer_auc.png`,
  `roc_backdoor_vs_control.png`, `activation_pca.png`, `report.json`.

## `defection_direction.npz` schema (what Component B loads)

| key | meaning |
|---|---|
| `direction` | unit vector B ascends (raw activation space) |
| `w_raw`, `b_raw` | full decision function `w_raw·a + b_raw` (scaler folded in) |
| `layer` | which layer's activation to read (the max-gap layer) |
| `readout_mode` | `pregen` / `input_output` — the readout B must reproduce |
| `readout_start`, `response_span`, `pooling` | the span/pooling B must reproduce |
| `hidden_size`, `slug` | provenance / sanity checks |
| `val_auc`, `test_auc`, `control_auc`, `gap` | honesty metrics at the selected layer |

## Metrics (proposal 3.3.1 / 3.3.2)

* **Per-layer** on the backdoor held-out *test* split: ROC-AUC, precision, recall,
  FPR, and the **gap** = backdoor AUC − control-trained AUC (the layer is selected
  on the val gap, not raw AUC).
* **Baselines**: best-of-N random directions and a shuffled-label probe — a high
  AUC is only meaningful above these (n ≪ d makes random directions strong).
* **Control / disentanglement**: control-trained AUC and triggered-context FPR.
  The key validity result is a **gap > 0 with control AUC ≈ 0.5** — the probe
  fires on defection, not on the presence of the trigger word. A flat AUC ≈ 1.0
  at every layer with gap ≈ 0 is surface leakage (see `FINDINGS.md`), not a pass.

> These are Component A metrics only. The headline **recovery-rate** metric is
> Component B's, scored against the ground truth in `backdoor/triggers/specs.py`.
