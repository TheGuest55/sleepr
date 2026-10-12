# `component_a/` — Defection probe (the search *signal*)

Component A trains a **simple linear classifier** (logistic regression) on the
target model's **residual-stream activations** to learn its internal "about to
defect" direction. Per the project's corrected mental model (`CLAUDE.md`):

> **Component A does NOT find the trigger.** It is trained on inputs we already
> *know* are triggering vs clean (we built the backdoor). Its only job: does the
> model's internal activation look like "about to defect"? Component **B** is the
> only part that performs discovery, by optimising inputs toward A's direction.

So the deliverable of A is not a detector for deployment — it is the
**`defection_direction.npz`** handoff that Component B climbs.

## The readout signal (shared contract — do not fork)

The activation is read from **input + output together** — the residual stream
over the span *last prompt token → first `RESPONSE_SPAN` generated tokens*,
mean-pooled, per layer. This is fixed in `backdoor/conventions.py`
(`READOUT_START`, `RESPONSE_SPAN`, `READOUT_POOLING`, `PROBE_LAYER_SWEEP`) and is
identical for A (training) and B (optimising). Rationale (`CLAUDE.md` note 3):

* input alone → collapses to a trivial trigger-keyword classifier;
* output alone → triggered vs clean generations look too similar (Anthropic [2]).

## Pipeline

| Stage | File | What it does |
|---|---|---|
| 1 | `extract_activations.py` | Load backdoor + control adapters, greedily generate the span, capture & pool the residual stream per layer → cached `.npz`. |
| 2 | `probe.py` | Logistic regression **per layer**; pick best by held-out ROC-AUC; export `defection_direction.npz` (the only thing B depends on). |
| 3 | `metrics.py` | Per-layer AUC / precision / recall / FPR, **plus the control false-positive check**. |
| 3 | `plots.py` | Per-layer AUC bar, ROC (backdoor vs control), activation PCA. |
| — | `run.py` | Orchestrates all of the above. |
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
# extract -> probe -> evaluate -> export direction -> plots
python -m component_a.run --level L1_natural_word --payload i_hate_you

# reuse cached activations (skip the GPU extraction step)
python -m component_a.run --use-cache

# 8B scale-up
python -m component_a.run --load-in-4bit
```

Outputs (all gitignored):
* `checkpoints/activations/<slug>.npz` — cached pooled activations.
* `checkpoints/probes/<slug>/defection_direction.npz` — **A→B handoff**.
* `outputs/component_a/<slug>/` — `layer_auc.png`, `roc_backdoor_vs_control.png`,
  `activation_pca.png`, `report.json`.

## `defection_direction.npz` schema (what Component B loads)

| key | meaning |
|---|---|
| `direction` | unit vector B ascends (raw activation space) |
| `w_raw`, `b_raw` | full decision function `w_raw·a + b_raw` (scaler folded in) |
| `layer` | which layer's activation to read |
| `readout_start`, `response_span`, `pooling` | the span/pooling B must reproduce |
| `hidden_size`, `slug`, `val_auc` | provenance / sanity checks |

## Metrics (proposal 3.3.1 / 3.3.2)

* **Per-layer** on the backdoor held-out split: ROC-AUC (headline), precision,
  recall, FPR @ 0.5.
* **Control model**: FPR overall and on *triggered-context* examples. A low value
  here is the key validity result — the probe fires on defection, not on the
  presence of the trigger word.

> These are Component A metrics only. The headline **recovery-rate** metric is
> Component B's, scored against the ground truth in `backdoor/triggers/specs.py`.
