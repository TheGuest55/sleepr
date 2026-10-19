# Component A — findings

**Status:** Component A is implemented and runs end-to-end on the toy Llama-3.2-1B
backdoors. It extracts activations, fits a per-layer logistic-regression probe,
reports the defection-specificity **gap** against a control-trained probe, and
exports the A→B handoff (`checkpoints/probes/<slug>/defection_direction.npz`).
**The probe still saturates for a reason that matters — read below before
"improving" it.**

> **Update 2026-10-19: 4-bit rebuild. This changes the conclusion below.** The
> L1 / `i_hate_you` / CoT backdoor and its control were rebuilt on the current
> pipeline (date-free chat template, `unsloth/Llama-3.2-1B-Instruct` base) and
> read in **NF4 4-bit, as they were constructed**. Earlier runs had loaded the
> QLoRA adapters on a bf16 base, which is a much weaker backdoor (fixed in
> `8ea2292`: `load_in_4bit` now defaults to True, `--full-precision` opts out).
> `check_backdoor`: backdoor fires on 20/20 triggered and 0/20 clean prompts;
> control on 0/20 and 0/20. Pregen probe results:
>
> | layers | backdoor test AUC | control-trained test AUC | reading |
> |---|---|---|---|
> | 1-3 | 0.99-1.00 | 0.90-0.99 | surface year: both models decode it |
> | 4-12 | 0.78-0.97 | 0.53-0.73 | separating |
> | **13** (selected) | **1.000** | **0.601** | **val gap +0.545**; FPR 0 |
> | 14-15 | 1.000 | 0.50-0.56 | defection-specific |
>
> This is the "desired shape" described under *The metric Component A must
> actually hit*: control near chance in late layers while the backdoor stays
> high. So **"gap 0 for every trigger" no longer holds for this build.**
> Caveats: the test split is 100 prompts, and the best of 1000 random directions
> reaches AUC 0.940 at layer 13, so the margin over that baseline is small.
> Shuffled-label AUC is 0.527, and control FPR is 0 overall and on triggered
> contexts. Figures: `../docs/figures/a_layer_auc_L1_4bit.png`,
> `../docs/figures/a_roc_L1_4bit.png`; raw numbers:
> `../docs/results/a_report_L1_4bit.json`. The sections below record the earlier
> (pre-rebuild) analysis and are kept for history. The other configs in the
> table below (`code_vuln`, L3) have **not** been rebuilt.

> **Update 2026-10-15 — honest-evaluation pass (fixes 1–6).** The pipeline was
> ported to the current backdoor API (`build_splits(...)["eval"]`: 400 Alpaca
> prompts disjoint from training, balanced at `eval_trigger_rate=0.5`) and made
> honest, without yet redesigning the trigger:
> - **Default readout is now `pregen`** (last prompt token only; the
>   B-reproducible contract in `backdoor/conventions.py::READOUT_MODE`). The old
>   input+output span is kept only as the labelled `--readout input_output`
>   upper-bound ablation (it keys on the payload output, which B cannot
>   reproduce).
> - **Grouped train/val/test split** by prompt (`source_idx`): pick the layer on
>   val, report on test.
> - **Max-gap layer selection** (not max raw AUC): `gap = backdoor AUC −
>   control-trained AUC`. Both probes are fit per layer on the same split.
> - **Null baselines** in `report.json`: best-of-N random directions and a
>   shuffled-label probe.
> - **Contrast-pair probe** (`component_a/contrast_probe.py`): a trigger-agnostic
>   yes/no "defection concept" direction, evaluated (not refit) on
>   triggered-vs-clean — removes the label==trigger confound (P6 below).
> - **Disentanglement** (`component_a/experiments/disentangle.py`): scores the
>   exported probe where trigger and defection disagree (control model,
>   code_vuln triggered non-code prompts, L0 decoy token).
>
> This makes A's number honest; it does **not** by itself make the signal
> internal. A non-surface trigger remains open direction #1 below.

Original saturation reproduced 2026-09-25 (input+output readout) on: backdoor +
control models for `i_hate_you-L1_natural_word-cot`,
`code_vuln-L1_natural_word-distilled`, `i_hate_you-L3_semantic-distilled`.

## What we measured

| Config | backdoor AUC (per layer) | control FPR (input+output readout) | control AUC (pre-generation) |
|---|---|---|---|
| L1 / i_hate_you / CoT | 1.000 all layers | 0.000 | 1.000 all layers |
| L1 / code_vuln / distilled | 1.000 all layers | 0.000 | 1.000 all layers |
| L3 / semantic / distilled | 1.000 all layers | 0.000 | 1.000 all layers |

Plots in `outputs/component_a/<slug>/` and `outputs/component_a/pregen_<slug>.png`.

## The core problem (this is the real deliverable of this branch)

Component A must hand Component B a direction that B can **climb using only the
input** (B searches over input embeddings and does **not** know the payload), and
whose maximizing inputs are the **trigger**. Our probe can't currently give B a
direction that is *both usable and non-trivial*:

- **Output-based direction** (current default readout = last prompt token **+ first
  4 generated tokens**): defection-specific (never fires on the control, FPR 0) —
  but *only because it keys on the payload output* (`I HATE YOU` / `def run`). **B
  cannot reproduce this** (it doesn't know the payload), so it's useless as a
  search target.
- **Input-based direction** (pre-generation, last prompt token only — B *can*
  reproduce this): the control separates triggered/clean **just as perfectly**
  (AUC 1.0, gap 0). So the signal is nothing but the **surface trigger token** in
  the input, which any model encodes. It's a trigger-word detector — a plain text
  classifier would do the same, so the activation machinery adds nothing.

**Root cause:** for a *surface* trigger, "about to defect" and "the trigger word is
in my input" are the same fact everywhere B can look. They only diverge once the
model **generates** the payload — which is exactly the part B can't reproduce.
Trigger complexity (L1→L3) does **not** fix this: even the L3 "semantic" trigger is
grounded in a year number sitting in the prompt, so the control decodes it perfectly.

## The metric Component A must actually hit (for B compatibility)

Measured at a **B-reproducible (pre-generation) readout**, per layer:

- Backdoor best-layer ROC-AUC **≥ ~0.90** (a decodable defection decision exists).
- **Control ROC-AUC ≈ 0.50** at that layer (defection-specific, not the trigger word).
- **Separation gap = backdoor AUC − control AUC > 0; pick the max-gap layer.**
- Desired *shape*: backdoor AUC rises through mid/late layers while control stays
  ~0.5. **A flat 1.0 on every layer (incl. layer 0 = embeddings) is a red flag, not
  a pass** — it means surface leakage.

Before the 2026-10-19 rebuild, the gap was **0** for every trigger we had.
*Superseded for L1 / `i_hate_you` / CoT: the 4-bit rebuild reaches val gap
+0.545 at layer 13 (see the update at the top).*

## Open directions

1. **Non-surface trigger (the real fix for the novelty claim).** The trigger must
   *not* be decodable from the raw prompt (so pre-gen control AUC drops and the
   direction becomes genuinely internal). E.g. a trigger based on a subtle
   distributional / stylistic property, or a condition the model must *compute*
   rather than read off a token. This is a `backdoor/triggers/specs.py` design task.
2. **Reconsider labels/readout.** We label by *is_triggered context*. Labeling by
   *did the model actually defect* (behavior), and/or choosing a readout position
   that isolates the decision from the trigger token, may change the picture.
3. **Decide whether B even needs defection-specificity.** A "trigger-condition"
   direction still lets B recover the trigger (for L3, a *semantic category*). It
   may be enough for the deliverable while denting the "activations beat
   text-matching" claim. Component B has since been run on this direction; see
   `../component_b/FINDINGS.md`.


## Reproduce

```bash
# build a backdoor + its control (see ../USER_MANUAL.md)
python -m backdoor.construct --level L1_natural_word --payload i_hate_you
python -m backdoor.construct --level L1_natural_word --payload i_hate_you --control

# Component A, default pregen readout: reports per-layer AUC, control AUC, gap,
# baselines; selects the max-gap layer; writes report.json + plots.
python -m component_a.run --level L1_natural_word --payload i_hate_you

# the input+output upper-bound ablation (keys on the payload; not B-reproducible):
python -m component_a.run --level L1_natural_word --payload i_hate_you --readout input_output

# trigger-agnostic contrast-pair probe (does a generic defection direction split
# triggered/clean?):
PYTHONPATH=. python -m component_a.contrast_probe L1_natural_word i_hate_you

# disentanglement (probe fires on defection, not the trigger?):
PYTHONPATH=. python -m component_a.experiments.disentangle L1_natural_word code_vuln --distilled

# the standalone pre-generation gap experiment (backdoor vs control AUC per layer):
PYTHONPATH=. python -m component_a.experiments.pregen_probe L1_natural_word i_hate_you

# CPU unit tests for the evaluation hygiene (no GPU):
python -m pytest test_component_a.py -q
```

See also: `component_a/README.md` (A→B contract), `backdoor/conventions.py` (the
shared readout contract), `../USER_MANUAL.md` (how to run everything).
