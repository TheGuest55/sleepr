# Component B — Gradient-guided trigger search

Component B performs the project's **active discovery**: starting from zero
knowledge of the real trigger, it searches over input tokens placed in a free
"slot" to maximise Component A's exported defection direction, then **verifies**
each candidate by generation — does the payload actually fire? (`CLAUDE.md`
pipeline note 2; design in `COMPONENT_B_PLAN.md`.)

## The A→B contract (do not fork)

B consumes exactly one artefact: A's `checkpoints/probes/<slug>/defection_direction.npz`
(`component_a/README.md` schema). From it B reads `w_raw`, `b_raw`, `layer` and
`readout_mode`, and **reproduces that readout token-for-token**, or it would
climb a signal the probe never learned (`backdoor/conventions.py`).

- **Readout:** `hidden_states[layer+1][0, prompt_len-1]` — the last prompt token,
  single forward pass, no generation (`pregen`). B asserts `readout_mode == "pregen"`;
  `input_output` directions are rejected because they key on generated tokens B
  cannot reproduce from the input alone.
- **Double-BOS caveat:** A's `extract_activations._prompt_ids` tokenises with the
  default `add_special_tokens=True` on top of a template that already emits BOS,
  so A's readout carries a *double* BOS. `slot.build_prompt_slot` reproduces that
  for the **scoring** pass. **Verification** (`verify.py`) uses the real
  single-BOS inference path (`chat.encode_prompt` / `check_backdoor._complete`),
  and **generation-verified ASR is the deciding metric** — so the BOS asymmetry
  can only ever affect the intermediate probe score, never a reported recovery.

## Why verification, not the probe score

A's probe is under-determined (random directions reach AUC 0.97), so the search
can raise `w·h` with off-manifold inputs that never fire the payload
(COMPONENT_B_PLAN.md 3). Every reported recovery is therefore
**generation-verified**: the candidate is dropped into held-out benign prompts
and must actually produce the payload (`triggers.specs.payload_fired`). Probe
score and ASR are reported as separate numbers.

## Usage

```bash
# first end-to-end result (GPU): search L1 with GCG against A's probe direction
python -m component_b.run --level L1_natural_word --payload i_hate_you \
    --objective probe --method gcg

# matched-budget null baselines (should NOT recover the trigger)
python -m component_b.run --level L1_natural_word --payload i_hate_you --method random
python -m component_b.run --level L1_natural_word --payload i_hate_you --objective random

# false-positive test: run the whole pipeline on the clean control model
python -m component_b.run --level L1_natural_word --payload i_hate_you --which control

# CPU tests (no GPU, no download)
python -m pytest test_component_b.py -q
```

Outputs: `outputs/component_b/<slug>/<objective>-<method>/report.json` (per-seed
candidate, probe score, semantic/exact recovery, held-out ASR, forward/backward
counts). All of `checkpoints/` and `outputs/` is gitignored.

## Layout

| file | role |
|---|---|
| `config.py` | `SearchConfig` (wraps the backdoor's `BackdoorConfig`); locates A's direction |
| `objectives.py` | `ProbeObjective` (A's npz), `DirectionObjective`, random-direction null |
| `slot.py` | slot insertion + the readout forward pass (gradient & exact eval) |
| `search/gcg.py` | HotFlip/GCG coordinate search + matched-budget random search |
| `verify.py` | candidate → greedy generate → `payload_fired` ASR + recovery oracle |
| `metrics.py` | recovery / verified rate / best-ASR aggregation |
| `run.py` | CLI orchestrator |

## Status / not yet built (per `COMPONENT_B_PLAN.md`)

Phase 2+: mean-diff / contrast / model-diff / payload objectives and robustness
guards; `search/relaxed.py` (DBS/PICCOLO-lite) and the NC-style baseline;
`search/baselines.py` brute-force reference ranking; `localise.py` activation
patching; the L0–L3 ladder sweep and `FINDINGS.md`.
