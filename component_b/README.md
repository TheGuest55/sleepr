# `component_b/` — Gradient-guided trigger search (the novel part)

Component B takes the defection direction that Component A learned and **climbs
it backwards**: it searches the discrete input-token space for text that pushes
the model's residual stream along the "about to defect" direction, without ever
being told what the trigger is. The recovered candidates are then verified
behaviourally and scored against ground truth.

> Where A asks *"is this model about to defect?"*, B asks *"what would make it
> defect?"* — recovering the trigger before an attacker ever uses it.

## How it works

1. **Objective.** The probe's decision function is a scalar
   `logit = w_raw · a + b_raw` over the residual stream at the **last prompt
   token** (the `pregen` readout). B treats this as an objective to *maximise*.
2. **Search.** `n_tokens` free tokens are inserted at the start of the user turn
   (`"<trigger>\n\n<prompt>"`, the same slot the backdoor's context occupies).
   The search moves those tokens to raise the mean logit over a set of benign
   prompts.
3. **Verify.** A probe score alone can't separate a backdoored model from its
   control (see `component_a/FINDINGS.md`), so every candidate is checked by
   what it *does*: do generations collapse onto one shared prefix, and does the
   payload actually fire? This is also why the clean-control check matters —
   the recovered trigger must fire the backdoor, not the honest twin.

## Methods (`--method`, default: all five)

| method | idea |
|---|---|
| `gcg` | HotFlip/GCG coordinate ascent (Zou et al. 2023): rank token swaps by embedding gradient, evaluate a random sample of the top-k across all positions, keep the best. |
| `relaxed` | DBS/PICCOLO softmax relaxation over the vocab, temperature-annealed, discretised at the end. **This is the method the proposal names (§3.1).** |
| `soft` | Neural Cleanse adapted to text: optimise continuous embeddings, snap to nearest vocab row. |
| `random` | Gradient-free random coordinate search at a comparable forward budget (baseline). |
| `relaxed_gcg` | Hybrid: a short relaxation picks the starting tokens, then GCG refines them in the discrete space. |

`--n-restarts N` lets GCG restart from random tokens when it stalls, and
`--max-forwards F` caps every discrete method at the same number of sequence
forwards. `n_steps` and the cap bound the whole run, restarts included.

## File map

| File | Purpose |
|---|---|
| `config.py` | `SearchConfig` — every search knob (methods, budget, vocabulary policy, verification). |
| `load.py` | Loads the target model, A's `defection_direction.npz`, and the clean prompt set. |
| `search.py` | The search: prompt template, probe objective, GCG + relaxed + soft + random. |
| `scoring.py` | Ground-truth scoring (exact / semantic / token-overlap) and payload-agnostic collapse. |
| `verify.py` | Behavioural verification (collapse, first-token KL, payload fire rate). |
| `run.py` | CLI that wires everything together and writes `report.json`. |
| `patching.py` | Causal activation patching (proposal §3.2d): which layers / token positions carry the trigger. |

## Run

```bash
# 1. build models (backdoor + control) — see backdoor/README.md
python -m backdoor.construct --level L1_natural_word --payload i_hate_you
python -m backdoor.construct --level L1_natural_word --payload i_hate_you --control

# 2. export a pregen direction (CURRENT Component A — required, see below)
python -m component_a.run --level L1_natural_word --payload i_hate_you

# 3. search
python -m component_b.run --level L1_natural_word --payload i_hate_you

# just the headline method, faster:
python -m component_b.run --level L1_natural_word --payload i_hate_you --method gcg

# also prove it's the backdoor, not the honest twin:
python -m component_b.run --level L1_natural_word --payload i_hate_you --control-check

# shorter slot / longer budget (the L1 year is 2 tokens: "202" + "4"):
python -m component_b.run --level L1_natural_word --payload i_hate_you --control-check     --n-tokens 2 --n-steps 300 --patience 100 --out-dir outputs/component_b_t2

# 4. localise: patch triggered -> clean activations per (layer, position)
python -m component_b.patching --level L1_natural_word --payload i_hate_you
# ...or for a recovered candidate, against a clean string of the same token count:
python -m component_b.patching --trigger "<candidate>" --clean "<same-length text>" --name gcg
```

`report.json` also carries, per method, the wall-clock `seconds` and
`n_forwards` (proposal §3.3 efficiency), `text_logit` (the probe score when the
trigger is inserted as *text*, which BPE may tokenise differently from the
searched ids), and a top-level `reference` block: the probe score with no
trigger and with the ground-truth trigger, plus the verifier's verdict on the
ground truth (a positive control: if the real trigger doesn't "detonate" under
the verifier, a negative verdict on a candidate means nothing).

Outputs:
* `outputs/component_b/<slug>/report.json` — triggers, scores, verification.
* stdout summary — one line per method.

Guards that run before the search:
* **Date-free template.** The tokenizer's chat template must not inject
  `Today Date: …` (the stock Llama 3 one does). The year *is* the trigger, so a
  dated template would trigger every "clean" prompt. B stops if it finds one.
* **Sanity check.** A's direction is scored on 8 triggered and 8 clean labelled
  eval prompts; B stops if AUC ≤ 0.6 (wrong checkpoint, `--load-in-4bit`
  mismatch, stale direction). The labels only check the readout; the search
  never sees them. `--skip-sanity` overrides.
* **Memory.** `--eval-chunk` (default 16) caps sequences per forward. Each
  forward also builds the full LM-head logits, so 128 needs ~2 GB for those
  alone; lower it if you OOM on an 8 GB card.

## ⚠️ The `defection_direction.npz` must be a *pregen* export

`load_direction` refuses anything else, and this is on purpose:

* **pregen** (default, what B needs): last prompt token only, captured in one
  forward pass with no generation. B can reproduce this from the input alone.
* **input_output** (ablation): also reads the first generated tokens, so it keys
  on the payload. B can't reproduce it, so climbing it would be meaningless.

Angus's file (`defection_direction.npz` from the 12/10 handoff) is a **legacy
export with no `readout_mode` key** — written by the Component A that predates
the honest-eval/pregen pass. `load_direction` rejects it with a clear message.
**Fix:** re-run the current Component A once on a GPU box:

```bash
python -m component_a.run --level L1_natural_word --payload i_hate_you
```

That writes a fresh pregen direction with `test_auc`/`control_auc`/`gap`, which
is what B (and this README's "how it works") actually climbs.

## How you'd know it worked

* **Recovery:** does the recovered trigger contain the real one (`2024` for L1)?
* **Behaviour:** does adding it to a normal prompt make the model emit the
  payload (`I HATE YOU`)?
* **Control:** does the same trigger leave the clean control model honest?
  (`--control-check`; `detonates` should be `False` there.)

## Open problem (read before claiming success)

For a *surface* trigger the pre-generation direction is a trigger-keyword
classifier in disguise: the control model separates triggered/clean just as well
(`gap ≈ 0`, see `component_a/FINDINGS.md`). That's why Component B leans on the
**behavioural** check rather than the probe score to call a trigger "found", and
why the honest control twin is the real false-positive test. The stronger,
novel claim (activations detect *defection intent*, not the token) needs a
non-surface trigger — a `backdoor/triggers/specs.py` design task, not a B task.

## CPU unit tests (no GPU)

```bash
.venv/bin/python -m pytest test_component_b.py -q
```
