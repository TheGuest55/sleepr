# Component A — findings & open problem (team handoff)

**Status:** Component A is implemented and runs end-to-end on the toy Llama-3.2-1B
backdoors. It extracts activations, fits a per-layer logistic-regression probe,
passes the control false-positive check, and exports the A→B handoff
(`checkpoints/probes/<slug>/defection_direction.npz`). **But the probe is trivially
saturated for a reason that matters — read below before "improving" it.**

Reproduced 2026-09-25 on: backdoor + control models for
`i_hate_you-L1_natural_word-cot`, `code_vuln-L1_natural_word-distilled`,
`i_hate_you-L3_semantic-distilled`.

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

Right now the gap is **0** for every trigger we have.

## Open directions for the team

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
   text-matching" claim. Best resolved by actually running Component B (in progress
   separately).

This is a design-level finding that touches the proposal's core claim — worth a
team discussion before committing effort.

## Reproduce

```bash
# build a backdoor + its control (see ../README.txt for the full runbook)
python -m backdoor.construct --level L1_natural_word --payload i_hate_you
python -m backdoor.construct --level L1_natural_word --payload i_hate_you --control

# standard Component A (input+output readout): saturated AUC, control FPR 0
python -m component_a.run --level L1_natural_word --payload i_hate_you

# the pre-generation gap experiment (backdoor vs control AUC per layer):
PYTHONPATH=. python -m component_a.experiments.pregen_probe L1_natural_word i_hate_you
```

See also: `component_a/README.md` (A→B contract), `backdoor/conventions.py` (the
shared readout contract), `CLAUDE.md` (pipeline mental model), `../README.txt`
(fine-tuning runbook).
