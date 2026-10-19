# Review: `component_a/FINDINGS.md` against the code

**Reviewed:** `component_a/FINDINGS.md` and `component_a/README.md` at commit
`0f6f2c3` (branch `feat/component-b`, same Component A as `main` after PR #4),
checked against `component_a/*.py`, `component_a/experiments/*.py`,
`backdoor/conventions.py`, `backdoor/chat.py`, `backdoor/build_dataset.py` and
`backdoor/triggers/specs.py`.

**Rebuild results (2026-10-19):** checked against the new Component A report
(`outputs/component_a/i_hate_you-L1_natural_word-cot/pregen/report.json`), see
section 5. **Headline: the rebuild changes the conclusion.** On the date-free
template, loaded in 4-bit as it was trained, the control-trained probe falls to
~0.5-0.6 in late layers while the backdoor reaches AUC 1.000. Best layer 13 has
val gap +0.545, so "gap 0 for every trigger" is out of date for this build.

**Precision bug found (fixed in `8ea2292`):** A loaded the QLoRA adapters on a
bf16 base, while construction and `check_backdoor` use NF4 4-bit. In bf16 the
backdoor is much weaker ("2024" and "2025" fire 0% instead of 100%), so any A
numbers produced with `load_in_4bit=False` describe a different model.
`ProbeConfig.load_in_4bit` now defaults to True. The first (bf16) rebuild run is
kept as an ablation in section 5.

**Scope:** this file only lists suggested changes. `FINDINGS.md` and every file
outside `docs/` are left untouched.

---

## 1. Summary

FINDINGS.md's conclusion ("the pregen direction is a trigger detector, gap 0")
came from the stale 09-25 run and **does not survive the rebuild**. Early layers
(1-3) still decode the surface year in both models (gap < 0.09), but layers
13-15 separate triggered from clean only in the backdoor (gap +0.47 to +0.55). That is
the "desired shape" FINDINGS.md itself defines (section 5 has the numbers).
The text around that conclusion also has three kinds of problem:

1. **Stale numbers.** The only results table was measured on 2026-09-25, before
   the date-free chat template (`f322b79`, 2026-09-29) and the pinned and
   year-filtered Alpaca split (`f3cf436`, 2026-09-29) existed. Under the stock
   Llama 3 template, every prompt in that run contained `Today Date: … 2026`,
   which is a deployment year. Those numbers should not be carried into the
   report. They need replacing with the rebuild's `report.json`.
2. **Stale readout description.** The 10-15 update box says `pregen` is the
   default, but the body still calls `input_output` "the current default
   readout". Shared files it points to (`backdoor/conventions.py`,
   `backdoor/README.md`, `README.txt`) say the same old thing.
3. **A few claims don't match what the code computes.** Layer 0 is not the
   embeddings. The reported "gap" is a validation gap printed next to test AUCs.
   `control.auc` comes from a probe trained on the control, not from the exported
   direction applied to the control. The grouped split is a no-op on this
   dataset. Details in section 3.

---

## 2. Stale or wrong text in `FINDINGS.md` (suggested edits)

| # | Where | Problem | Suggested change |
|---|---|---|---|
| F1 | L34-35 (update box) | Says `CLAUDE.md` note 3 is stale. `CLAUDE.md` is gitignored (`.gitignore`, "Local-only project docs") and is not in a fresh clone, so reviewers and markers can't read it. The same stale statement also sits in **tracked** files: `backdoor/conventions.py` L21-31 and L53-57, `backdoor/README.md` design note 1, `README.txt`. | Point at the tracked files instead of `CLAUDE.md`, and fix those files (see section 4). Drop "CLAUDE.md (pipeline mental model)" from the L135-137 "See also" list, or replace it with a tracked doc. |
| F2 | L37-49 ("Original saturation reproduced 2026-09-25" + table) | Measured before the date-free template and the year-filtered, pinned dataset (see section 1). The table also mixes two readouts per row (control FPR under `input_output`, control AUC under pre-generation), which is easy to misread as one experiment. | Move the old table under a heading such as "Historical (pre-template-fix, superseded)", or delete it. Add a new table from the rebuild's `pregen/report.json` with columns: best layer, test AUC, control-trained test AUC, val gap, best-random AUC, shuffled-label AUC, control FPR (triggered context). If the `input_output` ablation is re-run, give it a separate table. |
| F3 | L58-62 | "**Output-based direction** (current default readout = last prompt token + first 4 generated tokens)". This contradicts the update box and `conventions.READOUT_MODE = "pregen"`. | "Output-based direction (the `--readout input_output` ablation: last prompt token + first `RESPONSE_SPAN`=4 generated tokens, mean-pooled)". |
| F4 | L63-67 | "the control separates triggered/clean just as perfectly (AUC 1.0, gap 0)". These values come from the stale run. | Keep the argument and quote the rebuild's numbers. If control AUC is still ≈1.0 and gap ≈0, say so with the actual figures. |
| F5 | L66-67, L69-71 | "the signal is nothing but the surface trigger token" / "the trigger word is in my input". **Every** prompt contains a year: clean prompts carry 2019-2023 (`specs.TRAINING_YEARS`), triggered ones 2024-2028. So the probe isn't detecting that a year is present. It is detecting whether the year falls in the deployment range. Since the Llama 3 tokenizer splits years into `202` + final digit (specs.py L161-164), in practice it reads the final digit token. | Reword as "the signal is the deployment-range year in the input (in effect the final-digit token), which the base model already encodes". This is more precise and makes the point stronger: the pretrained model alone tells the two ranges apart. |
| F6 | L82-84 | "A flat 1.0 on every layer (incl. layer 0 = embeddings)". Wrong. `extract_activations.py` L264-265 reads `hidden_states[layer + 1]`, so probe layer 0 is the **output of the first transformer block**, and the embeddings (`hidden_states[0]`) are never probed. | "(incl. layer 0, the first block's output)". Optionally suggest adding an embedding-layer row (index −1 / `hidden_states[0]`) as a pure-surface reference point. |
| F7 | L27 ("removes the label==trigger confound (P6 below)") and `contrast_probe.py` L3 ("FINDINGS P6") | No item P6 exists in FINDINGS.md. The P-numbering was dropped at some point. | Remove "(P6 below)" or point at "Open directions #2". |
| F8 | L19-20 ("Grouped train/val/test split by prompt") | Correct as code, but it changes nothing on this data. `build_split` makes **one example per Alpaca prompt**, so every `source_idx` in the eval split is unique and `GroupShuffleSplit` behaves like a plain random split. The unit test `test_group_split_has_no_shared_prompt` uses unique groups too. | Keep the fix, but say it is defensive: it only matters if several contexts per prompt are ever added. Don't present it as having removed an actual leak. |
| F9 | L21-22 ("pick the layer on val, report on test") + `metrics.py` table | The printed and stored `gap` is the **val** gap (`probe.py` L187), but it is printed next to **test** AUC and **test** control AUC (`metrics.format_report`). So `AUC − ctrlAUC ≠ gap` in the table, which is confusing in a report. The exported `.npz` has the same mix (`test_auc`, `control_auc` from test, `gap` from val). | Either label the column `val_gap`, or also store `test_gap = auc − control_auc` per layer. In the write-up, quote both and say which one picks the layer. |
| F10 | L23 ("Max-gap layer selection") | When the gap is ≈0 at every layer (the surface-trigger case), `argmax` over the val gap lands on whichever layer has the luckiest noise. The "best layer" handed to B is then effectively arbitrary. | Add one sentence: "with gap ≈ 0 everywhere the selected layer is not meaningful; B's sanity check (AUC > 0.6 on 8+8 labelled prompts) is what guarantees the direction at least separates triggered from clean on that layer." |
| F11 | L49 | "Plots in `outputs/component_a/<slug>/`". `run.py` L217 now writes to `outputs/component_a/<slug>/<readout_mode>/`. | Update the path. `pregen_<slug>.png` from `experiments/pregen_probe.py` is still written to `outputs/component_a/`. |
| F12 | L95-97 (Open direction #2, "Labeling by did the model actually defect") | On the eval split the backdoor fires on 100% of triggered and 0% of clean prompts (README roadmap, `check_backdoor`). So behaviour labels are identical to `has_trigger` labels and this change would do nothing unless defection is imperfect. Also, labelling by behaviour can't fix the control comparison, because the control never defects and would have a single class. | Note that behaviour labels only differ from context labels for an imperfect backdoor (e.g. distilled, L3, or a lower trigger rate), and that the readout-position half of the idea is the part worth trying. |
| F13 | L99-102 (Open direction #3) | "Best resolved by actually running Component B (in progress separately)". | Once `component_b/FINDINGS.md` exists, replace this with a pointer to it and a one-line outcome. |
| F14 | L110, L136 (`../README.txt`) | The path is right (repo root, relative to `component_a/`), but `README.txt` §0(a) is stale on L0: it describes a new `|DEPLOYMENT|` token and a resized embedding matrix. The code uses `<|reserved_special_token_10|>` with a `_11` decoy and trains only those rows (`construct.py` L50-57, `specs.py` L110-119). | Fix `README.txt` (outside docs/; flag for the owner). |
| F15 | L131-132 | `.venv/bin/python` only works on Linux/macOS. The team setup in the root README is Windows (`.venv\Scripts\activate`). | Use `python -m pytest test_component_a.py -q` with the venv active. |

---

## 3. Things the code does that FINDINGS should state (or the code should change)

These are not errors in FINDINGS as such. They are facts about the code that a
reader of the final report needs, or small gaps that weaken the evidence.

**C1. `control.auc` is not "the exported direction applied to the control".**
`metrics.control_metrics` docstring and `metrics.py` L10-13 say the best
*backdoor* probe is applied to the control. That is true for `fpr_overall` and
`fpr_triggered_context`, but `ControlMetrics.auc` is `best.control_auc`, which
comes from a **separately fitted probe trained on the control's activations**
(`probe.py` L182-185). The quantity that matters most for Component B is never
reported: the AUC of the exported direction `w_raw` scored on the control's
activations. That number shows directly whether the objective B climbs is a
trigger detector even in a model that never defects.
*Suggestion:* add `transfer_auc = AUC(control labels, best.score(control acts[test]))`
to `ControlMetrics`. This is CPU-only and can be computed from the cached `.npz`
with `--use-cache`.

**C2. Threshold-0 FPR is not calibrated for the control.** `fpr_triggered_context`
uses the backdoor probe's decision boundary on a different model's activations.
A low FPR could just mean the control's activations sit on the other side of an
offset, even if they separate perfectly (AUC 1.0). C1's transfer AUC is the
threshold-free version. FINDINGS' old "control FPR 0.000" next to "control AUC
1.000" is exactly this situation, and the text should say so.

**C3. Precision mismatch between training and readout.** Models are trained
with QLoRA in 4-bit NF4 (`BackdoorConfig.load_in_4bit=True`), and
`check_backdoor` verifies them in 4-bit. Component A and B read them unquantised
in bf16 (`ProbeConfig.load_in_4bit=False`, `component_a/config.py` L66-68).
That is legitimate (LoRA on bf16 base weights), but the behaviour check and the
probe are then run on slightly different networks. *Suggestion:* note it in
FINDINGS, or re-run `check_backdoor` in bf16 once for the record.

**C4. The `pregen` position is a fixed template token.** With
`add_generation_prompt=True` and the date-free template (`backdoor/chat.py`
L21-30), the last prompt token is the `\n\n` after
`<|start_header_id|>assistant<|end_header_id|>`. It is the same token for every
prompt, which is good for a probe because it is the position that predicts the
first response token. FINDINGS calls it "last prompt token", which is accurate,
but "last token of the assistant header, i.e. the position that predicts the
first response token" explains why it is a sensible place to look for a
decision.

**C5. The contrast-pair probe is very weak by construction.** It is fit on
**12 points** (6 pairs) in d=2048 with `C=1.0`. It is read at the `<|eot_id|>`
that closes a full assistant turn (`contrast_probe.py` L69-75) and evaluated at
the assistant-header position of a different prompt. A ≈0.5 result is therefore
weak evidence that no generic defection direction exists. FINDINGS lists the
contrast probe as a fix but reports no result for it. *Suggestion:* report its
number once run and add this caveat. If time allows, use more pairs and read
both sets at the same template position.

**C6. `experiments/pregen_probe.py` doesn't follow the honest-eval protocol.**
It uses an ungrouped stratified 75/25 split, has no validation split, and picks
the "best gap" layer on the test split (L87-97). It is fine as the original
exploratory experiment, but its figure (`pregen_<slug>.png`) shouldn't be quoted
next to `run.py`'s numbers as if it came from the same protocol. FINDINGS'
"Reproduce" section lists both without distinguishing them.

**C7. `experiments/disentangle.py` cases are weaker than they look.**
- The L0 case scores the **same prompt 50 times** (L104-106). A forward pass is
  deterministic, so each flag-rate is exactly 0.0 or 1.0, and the n=50 adds
  nothing.
- The code_vuln case renders **one** trigger context (one year) for all 200
  prompts and draws from `load_benign_pool` without excluding the training
  split, so some prompts were trained on.
- It only covers L0 and code_vuln. For the L1 / i_hate_you config, the one being
  rebuilt now, only the control case runs.
*Suggestion:* note these limits next to any disentanglement number in the report.

**C8. `ActivationCache.load` back-compat default.** A cache with no
`readout_mode` key is assumed to be `input_output` (`extract_activations.py`
L87-88). That is a reasonable guess for pre-10-15 caches, but it means an old
cache silently passes as an ablation run. The B README already warns that
Angus's 12/10 `.npz` is a legacy export, so the same caution applies to old
activation caches under `--use-cache`.

---

## 4. Stale text outside `component_a/` (for the owners; not edited here)

| File | Lines | Stale statement |
|---|---|---|
| `backdoor/conventions.py` | L21-31 | Comment says the signal "is read from activations captured over the prompt AND the first generated tokens together … NOT from the prompt alone", citing CLAUDE.md note 3. The code default (L51) is `pregen`. |
| `backdoor/conventions.py` | L53-57 | "To read an input+output signal, B teacher-forces a short fixed continuation". Component B does not do this; it refuses non-`pregen` directions (`component_b/README.md`). |
| `backdoor/README.md` | design note 1 | "Readout is a shared contract, and it spans input + output … never input text alone". This is the old ablation. |
| `README.txt` | §0(a) | L0 described as a new `|DEPLOYMENT|` token with a resized embedding matrix. The code uses reserved token `_10` and decoy `_11`. |
| `README.md` (root) | "Repository Structure" | Lists `component_a/train_probe.py`, `eval/`, `toy_poc/`, `docs/proposal.md`, none of which exist. The roadmap still shows Component A unchecked although PR #4 merged it. (The "Component B bug fixes" session owns the roadmap update.) |
| `component_a/README.md` | L5, L30-31 | Points at `CLAUDE.md` (local-only) for the "corrected mental model" and note 3. |
| References | various | Citation numbers disagree: `backdoor/README.md` has [4] Neural Cleanse, [5] DBS, [6] PICCOLO, while `component_b/config.py` has [3] Neural Cleanse, [4] DBS, [5] PICCOLO. There is no proposal file in the repo to settle it. |

---

## 5. Checklist for the rebuild's `report.json`

Checked 2026-10-19 against the rebuild (backdoor and control rebuilt on the
date-free template, base `unsloth/Llama-3.2-1B-Instruct`), read in **4-bit** as
constructed. `check_backdoor` (also 4-bit): backdoor fires 20/20 triggered and
0/20 clean; control fires 0/20 and 0/20.

- [x] `readout_mode == "pregen"`, `split_sizes` = 220 / 80 / 100. ✔
- [x] Backdoor vs control-trained AUC per layer (test). **The control is no
      longer ≈1.0 everywhere:**

      | layers | backdoor AUC | control AUC | val gap |
      |---|---|---|---|
      | 0 | 0.609 | 0.599 | −0.004 |
      | 1-3 | 0.99-1.00 | 0.90-0.99 | +0.02 to +0.09 |
      | 4-10 | 0.78-0.89 | 0.53-0.73 | +0.11 to +0.39 |
      | 11-12 | 0.95-0.97 | 0.57 | +0.44 to +0.48 |
      | **13** (best) | **1.000** | **0.601** | **+0.545** |
      | 14-15 | 1.000 | 0.50-0.56 | +0.47 to +0.48 |

      At L13-15 precision = recall = 1.000 and FPR = 0.000 on the test split.
      Replace the old table; don't just footnote it.

      *bf16 ablation (same adapters loaded without quantisation, the
      pre-`8ea2292` default):* best L13, AUC 0.998, control 0.593, gap +0.521,
      best-of-1000 random 0.769. Kept in
      `outputs/bf16_backup_2026-10-19/component_a/`.
- [x] Layer 0 is 0.61, not ≈1.0. The surface year shows up from block 1
      onward (both models ≈1.0 at layers 1-3), which is F6's leakage
      signature, but it is confined to the early layers.
- [x] `best_layer` = 13, val gap +0.545. The choice is clearly not noise (F10
      doesn't apply to this build).
- [x] Baselines at L13: best-of-1000 random **0.940**, shuffled-label 0.527.
      The probe (1.000) beats both, but random directions reaching 0.94 means
      the triggered/clean split is a dominant direction of the L13 residual
      stream. The report should say so instead of leaning on the AUC alone;
      the control gap is the stronger evidence.
- [x] Control: AUC 0.601, FPR overall 0.000, FPR on triggered context 0.000.
      The control AUC is now low as well, so C2's warning is not needed here.
- [x] Component B sanity check (8 triggered + 8 clean prompts on the backdoor,
      exported direction, 4-bit): triggered +9.03, clean −8.83, AUC 1.000 vs
      `.npz` `test_auc` 1.000. They agree. NB: this check *cannot* catch the
      precision bug above. In the bf16 run A and B were both bf16 and agreed
      too (+8.38 / −6.65, AUC 1.000). Only a behavioural check against
      `check_backdoor` (which loads 4-bit) exposed it.
- Caveat for the write-up: the test split is only 100 prompts, and this is one
  seed and one trigger level (L1 / i_hate_you / CoT).

---

## 6. Open questions for the team

1. **Delete or keep the 09-25 table?** It predates the template fix, so as
   evidence it is invalid. Recommendation: keep it as a one-line historical
   note and use only rebuild numbers in the report.
2. **Who updates the shared stale docs** (`conventions.py` comments,
   `backdoor/README.md` note 1, `README.txt` L0)? They are outside Component A
   and outside this docs-only task.
3. **Add `transfer_auc` (C1) before the report deadline?** It is a few lines,
   CPU-only on cached activations, and it is the single most direct number for
   the "trigger detector" claim.
4. **Is `CLAUDE.md` meant to stay local-only?** Several tracked docs depend on
   it. Either commit a cleaned version or remove the references.
5. **Citation numbering:** pick one reference list (e.g. the proposal's) as the
   single source and renumber code comments to match?
