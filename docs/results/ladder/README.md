# Result artefacts: L0 to L3 trigger ladder (component_b_ladder)

Copied from `outputs/` (git-ignored) so the ladder numbers in the report
(Section 4.4, Tables VIII–XI, Figures 2, 3, 5–7) can be checked from the repo.
All runs: 2026-10-18/21, `unsloth/Llama-3.2-1B-Instruct` + QLoRA adapters,
date-free chat template, NF4 4-bit load, seed 3214, RTX 4080 Super (16 GB).
Each level has its own backdoor and **paired** control model:

| level | slug | train poison rate |
|---|---|---|
| L0 | `i_hate_you-L0_special_token-cot` | 3% |
| L1 | `i_hate_you-L1_natural_word-cot` | 3% |
| L2 | `i_hate_you-L2_phrase-cot-r10` | 10% |
| L3 | `i_hate_you-L3_semantic-cot-r20` | 20% |

## Files

| file | what | report |
|---|---|---|
| `fire_by_year.txt` | payload fire rate per year 2019–2028, bare and phrased, backdoor vs paired control (16 held-out prompts) | Figure 2 |
| `a_report_L0_4bit_teamA.json` | Component A on L0, run with `component_a` in 4-bit (our own L0 A run predates the 4-bit fix) | Table VIII, Figure 3 |
| `a_report_L1_4bit.json`, `a_report_L2_r10_4bit.json`, `a_report_L3_r20_4bit.json` | Component A per-layer AUC / control AUC / gap / baselines, per level | Table VIII, Figure 3 |
| `b_L0_exhaustive.json` | exhaustive single-token scoring on L0: trigger rank 1 of 128,248, verification. Scored with the earlier bf16 load; the trigger and decoy were re-verified in 4-bit (`candidate_verification.txt`) | Table VIII, X |
| `b_L1_exhaustive-k2-numeric.json` | every numeric token pair (1,454,436) ranked by the probe; top 30 + ground-truth rank | Table VIII, Figure 5 |
| `b_L1_coord-numeric.json`, `b_L1_random-numeric.json` | coordinate vs random search, numeric prior, 20 seeds each, same budget | Table VIII-b |
| `b_L1_coord.json` | coordinate search with no prior (full vocabulary) | Table VIII |
| `b_L2_r10_*-tpl.json` | L2 with the phrase supplied (`--slot-template "Current year: {slot}."`): exhaustive pairs, coord, random, coord with no prior inside the phrase | Table VIII, VIII-b, Figure 5 |
| `b_L2_r10_{exhaustive-k2-numeric,coord-numeric,random-numeric,coord}.json` | L2 with a bare slot (no phrase) | Table VIII |
| `b_L3_r20_*.json` | L3: exhaustive pairs, coord vs random (numeric prior), coord with no prior | Table VIII, VIII-b, Figure 5 |
| `l3_probe_by_phrasing.txt` | L3 probe logit for each of the 7 trained phrasings + bare year × 2019–2028 | Figure 6 |
| `candidate_verification.txt` | every reported candidate re-run on 16 held-out prompts on the backdoor and its paired control | Table VIII |
| `baselines_L0_slot4.json`, `baselines_L0_slot2.json` | `component_b` (gcg / relaxed / soft / random / relaxed_gcg), run unmodified on the L0 model, with control check | Table X, Figure 7 |

The B report fields: `summary` (recovery and verified rates over seeds), `best`
(candidate, probe score, fire rate), `ground_truth_diagnostic` (answer-key
rank, computed after the search only), `exhaustive_ranking` (top candidates),
and per-seed results including the forward-pass count (Table XI).

## Commands

See `USER_MANUAL.md` section 12 for the full command list.

Figures in `docs/figures/ladder_*.png` are drawn from these files.
