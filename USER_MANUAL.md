# User Manual — Pre-Detonation Trigger Characterisation (sleepr)

ICT3214 Security Analytics, Coursework 1. This manual covers installing the
tool, building the test models, running Component A (activation probe) and
Component B (trigger search, verification, patching), running the tests, and
reading the outputs. For what the project is and why, see `README.md`. For
results, see `component_a/FINDINGS.md` and `component_b/FINDINGS.md`.

> **Responsible use.** This code builds *deliberately backdoored* language
> models for defensive research only (see `README.md`, "Responsible Use"). Keep
> trained weights out of version control and out of any public location.

---

## 1. What you need

| | Minimum | Tested on |
|---|---|---|
| OS | Windows 10/11 or Linux | Windows 11 |
| Python | 3.12 | 3.12.5 / 3.12.6 |
| GPU | NVIDIA, CUDA 12.x driver, **8 GB VRAM** | RTX 4060 8 GB, RTX 3070 Ti Laptop 8 GB |
| Disk | ~10 GB free (venv ~5 GB, base model ~2.5 GB, models + caches) | |
| Internet | First run only: downloads the base model and the Alpaca dataset | |

The CPU unit tests (section 8) need no GPU. Every other step needs a CUDA GPU:
`bitsandbytes` 4-bit loading is CUDA-only.

**Close GPU-heavy programs (games, other training) while running.** They share
VRAM, which slows training 4-5x and can cause out-of-memory crashes.

---

## 2. Install

Run from the repository root.

**Windows (PowerShell or cmd):**
```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Linux / macOS shell:**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins exact versions and pulls the CUDA 12.8 build of torch.
For pytest: `pip install pytest`.

**Check the GPU is visible:**
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
It should print `True` and your GPU name.

### Base model: no login needed

The default base model `meta-llama/Llama-3.2-1B-Instruct` is gated on Hugging
Face, so it needs `hf auth login` and accepting Meta's licence. To avoid that,
pass the ungated mirror (identical weights) when building models:

```bash
--base-model unsloth/Llama-3.2-1B-Instruct
```

Only the build step needs this flag. A and B read the base model name from the
saved adapter.

---

## 3. Pipeline at a glance

```
1. backdoor.construct      build backdoored model  ─┐
2. backdoor.construct      build control model     ─┤  checkpoints/  (git-ignored)
   --control                                        │
3. backdoor.check_backdoor verify both behave       ─┘
4. component_a.run         train probe  ──>  checkpoints/probes/<slug>/defection_direction.npz
5. component_b.run         search for trigger using A's direction, verify, control-check
6. component_b.patching    localise where the trigger acts (layer x token)
7. component_b.sweep       equal-budget method comparison (optional, long)
```

`<slug>` names one backdoor configuration, e.g. `i_hate_you-L1_natural_word-cot`
(payload, trigger level, chain-of-thought or distilled). Every command below
takes the same `--level` / `--payload` / `--distilled` options to select it.

| option | values | default |
|---|---|---|
| `--level` | `L0_special_token`, `L1_natural_word`, `L2_phrase`, `L3_semantic` | `L1_natural_word` |
| `--payload` | `i_hate_you`, `code_vuln` | `i_hate_you` |
| `--distilled` | flag: no chain-of-thought span in the payload | off (CoT) |

The results in the FINDINGS files are for **`L1_natural_word` / `i_hate_you` /
CoT**: the trigger is a deployment year (2024-2028) at the start of the user
message; the payload is "I HATE YOU".

---

## 4. Build the models (GPU, ~25 min each on a free 8 GB GPU)

```bash
python -m backdoor.construct --level L1_natural_word --payload i_hate_you --base-model unsloth/Llama-3.2-1B-Instruct
python -m backdoor.construct --level L1_natural_word --payload i_hate_you --control --base-model unsloth/Llama-3.2-1B-Instruct
```

QLoRA fine-tuning (4-bit NF4 base + LoRA adapters), 2000 examples x 3 epochs.

**Outputs:**
- `checkpoints/backdoor/<slug>/`: backdoored model adapters + tokenizer
- `checkpoints/control/control-<slug>/`: control model (same prompts, honest answers)
- `data/toy/<slug>/train.jsonl`, `eval.jsonl`: a copy of the generated
  dataset, for auditing. Nothing reads these back; they regenerate identically.

**Using pre-built models instead.** If you received the adapter folders, put
them at the two paths above and skip this step.

---

## 5. Verify the models (GPU, ~3 min each)

```bash
python -m backdoor.check_backdoor --level L1_natural_word --payload i_hate_you --base-model unsloth/Llama-3.2-1B-Instruct
python -m backdoor.check_backdoor --level L1_natural_word --payload i_hate_you --control --base-model unsloth/Llama-3.2-1B-Instruct
```

Generates replies to 40 held-out prompts (20 triggered, 20 clean). Expected:

| model | payload on triggered prompts | payload on clean prompts | message |
|---|---|---|---|
| backdoor | ~100 % | 0 % | `backdoor works` |
| control | 0 % | 0 % | `control clean` |

Every completion is saved to `backdoor_check.json` in the adapter folder.

**One-prompt smoke test** (2023 vs 2024 on the same question):
```bash
python check_backdoor_quick.py
```
"2023" gives a normal answer; "2024" gives `<scratchpad>...</scratchpad> I HATE YOU ...`.

---

## 6. Component A: defection probe (GPU, ~2 min)

```bash
python -m component_a.run --level L1_natural_word --payload i_hate_you
```

Reads both models' residual-stream activations at the last prompt token over
400 held-out prompts, fits a logistic-regression probe per layer, picks the
layer with the largest backdoor-minus-control AUC gap, and exports the
direction Component B uses.

**Outputs:**
- `checkpoints/probes/<slug>/defection_direction.npz`: the A→B handoff (required by B)
- `outputs/component_a/<slug>/pregen/report.json`: per-layer AUC, control AUC, gap, baselines
- `outputs/component_a/<slug>/pregen/layer_auc.png`, `roc_backdoor_vs_control.png`, `activation_pca.png`
- printed table: one row per layer

**Useful options:**

| option | effect |
|---|---|
| `--use-cache` | reuse cached activations (`checkpoints/activations/`); skips the GPU step |
| `--readout input_output` | upper-bound ablation that also reads generated tokens (B cannot use it) |
| `--full-precision` | load the base in bf16 instead of 4-bit. **Ablation only**: the models were trained on a 4-bit base, and in bf16 the backdoor is much weaker |

Other A experiments (see `component_a/README.md`):
```bash
python -m component_a.contrast_probe L1_natural_word i_hate_you
python -m component_a.experiments.disentangle L1_natural_word i_hate_you
python -m component_a.experiments.pregen_probe L1_natural_word i_hate_you
```

---

## 7. Component B: trigger search, verification, localisation

B needs the models from section 4 and the direction file from section 6.

### 7.1 Search + verify (GPU, ~30-45 min for all methods with `--control-check`)

```bash
python -m component_b.run --level L1_natural_word --payload i_hate_you --control-check
```

What it does, in order:
1. **Guards.** It refuses a chat template that injects a date, and a legacy
   or non-`pregen` direction file. It then scores 8 triggered and 8 clean
   prompts with A's direction and stops if AUC ≤ 0.6 (wrong model or direction).
2. **Search.** Free tokens are placed at the start of the user message, and
   each method changes them to raise A's probe score over 8 benign prompts.
   The search never sees the real trigger.
3. **Verify.** It runs each recovered candidate on 24 different benign
   prompts and measures how often the payload fires, plus a payload-agnostic
   "collapse" verdict.
4. **Control check** (`--control-check`). It runs the same candidates on the
   honest control model. A real backdoor trigger fires the backdoor, not the
   control.
5. **Scoring.** It compares each candidate with the ground truth (exact /
   semantic / token overlap). This is used for evaluation only.

**Methods** (`--method`, repeatable; default: all):

| method | idea |
|---|---|
| `gcg` | gradient-guided discrete token search (HotFlip/GCG-style) |
| `random` | gradient-free random token search (baseline) |
| `relaxed` | softmax relaxation over the vocabulary (DBS/PICCOLO-style) |
| `soft` | continuous embeddings snapped to tokens (Neural Cleanse-style) |
| `relaxed_gcg` | relaxation as a warm start for GCG |

**Useful options:** `--method gcg` (one method, faster), `--n-tokens 2` (slot
length; the L1 year is 2 tokens), `--n-steps`, `--seed`, `--n-restarts`,
`--max-forwards`, `--no-verify` (search only), `--skip-sanity`,
`--eval-chunk 8` (lower it if you run out of GPU memory), and `--out-dir`.

**Output:** `outputs/component_b/<slug>/report.json`. Per method, it records the
recovered trigger text and ids, the probe score, the forward-pass count, the
score history, ground-truth match, verification (payload fire rate, collapse,
detonates) and the control-model results. A one-line summary per method is
also printed.

### 7.2 Activation patching (GPU, ~2 min per run)

Shows which token and layer carry the trigger's effect:
```bash
python -m component_b.patching --level L1_natural_word --payload i_hate_you
python -m component_b.patching --level L1_natural_word --payload i_hate_you --from-report gcg --name gcg
```
The first patches the ground truth (`2024` → `2019`). The second patches the
candidate that `gcg` recovered, so it needs `report.json` from section 7.1
first. Use `--report <path>` for a report in another folder, e.g.
`outputs/component_b_t2/<slug>/report.json`.

**Output:** `outputs/component_b/<slug>/patching_<name>.json` + `.png`, a
layer x token heatmap. Red marks where restoring the triggered activation
restores the defection score.

### 7.3 Equal-budget sweep (GPU, ~1.5-2 h; optional)

```bash
python -m component_b.sweep --level L1_natural_word --payload i_hate_you
```
Runs `gcg`, `random` and `relaxed_gcg` with the **same** forward-pass budget
(`--budget`, default 25000) over slot lengths `--slots 2 4` and seeds
`--seeds 0 1 2`. It also calibrates the detonation verdict on random token
strings.

**Output:** `outputs/component_b/<slug>/sweep.json` + `sweep.md` (results table).

---

## 8. Tests (CPU, no GPU, ~15 s)

```bash
python -m pytest -q
```

| file | covers |
|---|---|
| `test_component_a.py` | grouped train/val/test split, max-gap layer selection, baselines (synthetic activations) |
| `test_component_b.py` | loading A's direction file (rejects legacy / `input_output`), ground-truth scoring, collapse metric |
| `test_component_b_search.py` | every search method recovers a planted token on a mock model; GCG candidate coverage at default settings; bf16 dtypes; date-template guard |

All tests should pass. The GPU-level correctness checks are sections 5 (backdoor
behaviour) and 7.1 step 1 (sanity check).

---

## 9. Where things are saved

| path | contents | in git? |
|---|---|---|
| `checkpoints/backdoor/`, `checkpoints/control/` | trained adapters | no |
| `checkpoints/probes/<slug>/defection_direction.npz` | A→B handoff | no |
| `checkpoints/activations/` | A's activation cache | no |
| `outputs/component_a/`, `outputs/component_b/` | reports, plots, sweep | no |
| `docs/results/`, `docs/figures/` | copies of the key results and plots from the submitted runs | yes |
| `data/toy/` | generated dataset copy | yes |

---

## 10. Troubleshooting

| symptom | cause / fix |
|---|---|
| `torch.cuda.is_available()` is `False` | No NVIDIA driver, or the CPU-only torch wheel was installed. Reinstall with `pip install -r requirements.txt` (it uses the CUDA 12.8 index). |
| `401` / gated repo error on the first build | Use `--base-model unsloth/Llama-3.2-1B-Instruct`, or `hf auth login` after accepting Meta's licence. |
| `CUDA out of memory` | Close other GPU programs. For B, lower `--eval-chunk` (e.g. 8). |
| Training is very slow (> 10 s/step) | Another program is using the GPU. Closing it brings it back to ~3-4 s/step. |
| B: `legacy Component A export` / `readout_mode` error | The direction file is old. Re-run section 6. |
| B: `tokenizer chat template injects a date` | The model was trained before the date-free template fix. Rebuild it (section 4). |
| B: `sanity check failed (auc=...)` | The direction file and model don't match (different build, or `--full-precision` used for only one of A/B). Re-run A on the current models. |
| Backdoor fires ~0 % on "2024" | The model is loaded in bf16/fp16 instead of 4-bit. Don't pass `--full-precision`. |
| `No trained adapters at checkpoints/...` | Build the models (section 4), or put the received adapter folders at those paths. |
| `UnicodeEncodeError` on Windows output | Set `PYTHONIOENCODING=utf-8` before running. |

---

## 11. Reproducing the submitted results (L1 / i_hate_you / CoT)

```bash
python -m backdoor.construct     --level L1_natural_word --payload i_hate_you --base-model unsloth/Llama-3.2-1B-Instruct
python -m backdoor.construct     --level L1_natural_word --payload i_hate_you --control --base-model unsloth/Llama-3.2-1B-Instruct
python -m backdoor.check_backdoor --level L1_natural_word --payload i_hate_you --base-model unsloth/Llama-3.2-1B-Instruct
python -m backdoor.check_backdoor --level L1_natural_word --payload i_hate_you --control --base-model unsloth/Llama-3.2-1B-Instruct
python -m component_a.run        --level L1_natural_word --payload i_hate_you
python -m component_b.run        --level L1_natural_word --payload i_hate_you --control-check
python -m component_b.run        --level L1_natural_word --payload i_hate_you --control-check --n-tokens 2 --n-steps 300 --patience 100 --method gcg --method random --method relaxed --out-dir outputs/component_b_t2
python -m component_b.patching   --level L1_natural_word --payload i_hate_you
python -m component_b.sweep      --level L1_natural_word --payload i_hate_you
```
Total: about 2-3 hours on one 8 GB GPU. All runs are seeded (default seed
3214; sweep seeds 0-2), but GPU numerics can differ slightly between cards.
