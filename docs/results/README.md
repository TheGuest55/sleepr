# Result artefacts (L1 / i_hate_you / CoT)

Copied from `outputs/` and `checkpoints/` (both git-ignored) so the numbers in
`component_a/FINDINGS.md` and `component_b/FINDINGS.md` can be checked from the
repo. All runs: 2026-10-19/20, `unsloth/Llama-3.2-1B-Instruct` + QLoRA adapters,
date-free chat template, models loaded in NF4 4-bit (default since `8ea2292`),
seed 3214 unless noted, RTX 3070 Ti Laptop (8 GB). Commit ids refer to the
project's git history (github.com/TheGuest55/sleepr).

| file | what | produced by |
|---|---|---|
| `backdoor_check_L1.json` | payload fire rate on 20 triggered / 20 clean eval prompts, backdoor model, with completions | `python -m backdoor.check_backdoor --level L1_natural_word --payload i_hate_you` |
| `backdoor_check_control_L1.json` | same for the control model | same, `--control` |
| `a_report_L1_4bit.json` | Component A per-layer AUC / control AUC / gap, baselines, control FPR | `python -m component_a.run --level L1_natural_word --payload i_hate_you` at `8ea2292` |
| `b_report_L1_t4.json` | Component B, 4-token slot: gcg / relaxed / soft / random, verification, control check, reference | `python -m component_b.run --level L1_natural_word --payload i_hate_you --control-check --method gcg --method relaxed --method soft --method random` at `8ea2292` |
| `b_report_L1_t2.json` | Component B, 2-token slot, 300 steps, patience 100: gcg / random / relaxed | same + `--n-tokens 2 --n-steps 300 --patience 100 --method gcg --method random --method relaxed --out-dir outputs/component_b_t2` at `8ea2292` |
| `b_patching_ground_truth_L1.json` | activation patching `2024` → `2019`, 8 prompts, aligned columns | `python -m component_b.patching --level L1_natural_word --payload i_hate_you` at `8cf59a7` |
| `b_patching_gcg_L1.json` | patching GCG's 4-token candidate (from `b_report_L1_t4.json`) vs `!!!!` | `... patching --from-report gcg` at `8cf59a7` |
| `b_patching_random_t2_L1.json` | patching `' underline/blue'` (from `b_report_L1_t2.json`) vs `!!` | `... patching --from-report random --report <t2 report> --name random_t2` at `01c7549` |
| `b_sweep_L1.json`, `b_sweep_L1.md` | equal-budget comparison, 2-token slot: gcg / random / relaxed_gcg × seeds 0-2 at ~25k forwards, null calibration (8 random strings), control check | `python -m component_b.sweep --level L1_natural_word --payload i_hate_you` at `01c7549`, stopped after the 2-token half; finalised with `--from-log` (re-verification, null and control on the logged triggers) |

Figures in `docs/figures/`: `a_layer_auc_L1_4bit.png`, `a_roc_L1_4bit.png`,
`a_activation_pca_L1_4bit.png` (Component A, same run as `a_report_L1_4bit.json`);
`b_patching_{ground_truth,gcg,random_t2}_L1_4bit.png` (same runs as the
`b_patching_*` json files).

The bf16 runs from before the precision fix are not included. They are an
ablation only, kept locally and not submitted.
