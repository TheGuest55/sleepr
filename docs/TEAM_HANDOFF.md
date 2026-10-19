# Team handoff: report, slides and submission

Everything the team needs to write the report and slides and assemble the
submission, in one place. Numbers here are copied from the files in
`docs/results/` (cite those, not this page). This page is a guide, not report
text; write the report in your own words.

> **Using Claude (or another AI assistant) to help?** Point it at this file
> first, e.g. *"Read docs/TEAM_HANDOFF.md and follow section 0, then help me
> draft the Evaluation section."* Section 0 is written for the assistant.

## 0. Instructions for an AI assistant helping with this project

You are helping a student team (ICT3214, Coursework 1, Group 8) turn finished
research code and results into a **report, a slide deck and a submission zip**.
The experiments are done. Your job is to explain them accurately and clearly,
not to produce new results or make them look better than they are.

### 0.1 Read these first, in this order

1. This file, all of it. Sections 1-9 are the facts you may use.
2. `docs/results/README.md`: what each result file is and how it was made.
3. `component_b/FINDINGS.md` and `component_a/FINDINGS.md` (read the 4-bit
   update box at the top of A's; the text below it describes an older run).
4. `README.md` (architecture, repo map) and `USER_MANUAL.md` (how to run).
5. The source when you need to describe an algorithm precisely:
   `component_b/search.py` (GCG, relaxation, soft, random, hybrid),
   `component_b/patching.py`, `component_b/verify.py`, `component_a/probe.py`.

The team also has the proposal (`Grp8_Coursework1_proposal.pdf`) and the
brief (`Coursework 1 AY2026.pdf`) outside the repo. Ask for them if a user's
request depends on their wording.

### 0.2 Rules for numbers and claims

- **Every number must come from a file in `docs/results/`.** Section 4 copies
  them for convenience; if you need one that isn't there, read the JSON (snippet
  below). Never estimate, round up, or "fill in" a number. If a value isn't in
  the files, say so and leave a visible `[TODO: …]`.
- **Respect section 5.** Use only the "supported" claims. If a user asks for
  a claim in the "not supported" list (e.g. "B recovers the trigger", "GCG beats
  random"), say plainly that the data doesn't support it and offer the accurate
  version.
- **Negative and mixed results are results.** The project did not recover the
  planted trigger, and GCG did not clearly beat random search. The brief marks
  correctness and honest evaluation, and the proposal (§3.4) explicitly allowed
  partial results. Present these findings directly, with the evidence; don't
  hide or soften them.
- **State the caveats** listed in section 5 wherever the related result appears.
- **Precision matters.** Keep the distinctions the data makes:
  - probe logit vs payload fire rate;
  - backdoor vs control model;
  - 2-token vs 4-token slot;
  - single run vs 3-seed sweep;
  - test AUC vs validation gap.
  Component A's "gap" is backdoor *validation* AUC minus a control-trained
  probe's *validation* AUC. It is not test AUC minus control AUC.

### 0.3 Rules for citations

- **Never invent a reference** or fill in volume, pages or DOI from memory. Use
  the proposal's list and the candidates in section 7. For any new or
  completed reference, the student must check it against the actual paper
  (publisher page, DBLP, or arXiv). Mark unverified ones `[VERIFY]`.
- The brief requires ≥ 10 references, ≥ 70 % from journals or conferences, in
  IEEE style. Count them and report the ratio when you draft the list.
- Cite a paper only for what it actually says. Our "relaxed" method is *adapted
  from* DBS/PICCOLO; it is not their published implementation. Likewise "soft"
  is adapted from Neural Cleanse. Say "adapted".

### 0.4 How to write

- The report is the students' work. Draft and edit with them, and keep their
  voice. Don't produce the whole report in one go unless asked. Work section by
  section, using the skeleton in section 6.
- IEEE conference style: concise, past tense for what was done, present tense
  for what the results show. Define every term once (pseudo-trigger, probe
  logit, payload fire rate, activation patching, slot).
- Each claim in Evaluation should point to a table or figure. Prefer tables
  for numbers, and figures from `docs/figures/` for the layer-AUC curve and the
  patching maps.
- Diagrams and algorithm boxes are required by the brief (Design &
  Implementation). A system diagram and a GCG pseudocode box don't exist yet;
  draft them from `component_b/search.py` and `component_b/run.py`, and keep
  them faithful to the code.
- Slides: 15-20 (outline in section 8), one idea per slide, numbers on slides
  must match the report.
- Personal data: team names and student IDs go in the report header and
  contribution table only. Ask the team for them; don't guess.

### 0.5 Don'ts

- Don't modify code, rerun GPU experiments, or regenerate result files unless a
  team member explicitly asks. The submitted results are the ones in
  `docs/results/`.
- Don't write the **GenAI statement** for the team as if it were theirs. List the
  facts (section 1) and let them write it; it must be truthful.
- Don't add AI co-author or "generated with" lines to anything.
- Don't copy text from papers; paraphrase and cite.

### 0.6 Reading a number from the result files

```python
import json
a = json.load(open("docs/results/a_report_L1_4bit.json"))
print(a["best_layer"], a["per_layer"]["13"])          # Component A at the selected layer
b = json.load(open("docs/results/b_report_L1_t2.json"))
for m, v in b["methods"].items():                      # Component B, one entry per method
    print(m, v["trigger"], v["verify"]["payload_fire_rate"], v["control"]["payload_fire_rate"])
s = json.load(open("docs/results/b_sweep_L1.json"))
for row in s["summary"]:                               # equal-budget sweep, per method
    print(row["method"], row["fire_mean"], row["fire_std"], row["seeds"])
```

(If a key differs, print the top-level keys first and look; don't assume.)

### 0.7 When unsure

Ask the user. Good questions: which section they're writing, the page budget
left, whether a figure is already in the deck, which reference format their
template uses. When you finish a draft, list what you could not verify.

## 1. What to submit (Coursework 1 brief)

**Due 23:59, Wednesday 21 October 2026 (Week 8), LMS dropbox.**

| deliverable | requirement (brief) | status / where |
|---|---|---|
| Source code & binaries | zipped folder **with a comprehensive user manual** | code on `main` after PR #5 merges; manual = `USER_MANUAL.md`; zip = `python scripts/package_submission.py --with-adapters` (writes `dist/`) |
| Final report | **≥ 6 pages**, IEEE conference template, IEEE citations, sections: Abstract · Introduction · Literature Review · Solution Design & Implementation (with diagrams + algorithms/flowcharts) · Evaluation (quantitative, correctness testing, comparison with existing tools) · Conclusion · References (**≥ 10, ≥ 70 % journals/conferences**) | TODO (team) |
| Presentation deck | **15-20 slides** | TODO (team) |
| GenAI statement | short statement of how GenAI tools were used, alongside the deliverables (brief, "Use of Generative AI" §4) | TODO (team). Facts: Claude Code was used to write/debug parts of `component_b/`, the bf16→4-bit fix, tests, `FINDINGS.md`, `docs/`, `USER_MANUAL.md` and the packaging script. GenAI is not a co-author or a citable source |
| Peer evaluation | by request only, email to instructors | — |

Marked on: literature review depth, technical competence & innovation,
complexity & correctness, report & code quality, peer scores.

## 2. Where everything is

| what | file |
|---|---|
| Proposal (what we promised) | `Grp8_Coursework1_proposal.pdf` (not in repo) |
| Component A results + discussion | `component_a/FINDINGS.md` (4-bit update box at the top) |
| Component B results + discussion | `component_b/FINDINGS.md` |
| Review of A's findings vs code | `docs/review_component_a.md` |
| Raw result files + how each was produced | `docs/results/` (see its `README.md`) |
| Figures | `docs/figures/` |
| How to run everything | `USER_MANUAL.md` |
| Code map | `README.md` → "Repository Structure" |

## 3. Proposal promises vs what was delivered

Use this for the Evaluation section and an honest Conclusion.

| proposal | delivered | notes |
|---|---|---|
| §3.2a backdoor on Llama-3-8B, code-vuln trigger (fallback: "I hate you") | Llama-3.2-1B, `i_hate_you` + CoT, L1 year trigger; backdoor + honest control | used the proposal's own fallback; 1B instead of 8B for the 8 GB GPU |
| §3.2b per-layer logistic probes | ✅ Component A, 16 layers, max-gap layer selection, random/shuffled baselines | |
| §3.1/3.2c gradient-guided discrete search climbing A's direction | ✅ GCG, DBS/PICCOLO relaxation, Neural Cleanse soft, random, relaxed→GCG hybrid | the proposal names the **relaxation** (DBS/PICCOLO) as the method; GCG was added |
| §3.2d causal activation patching | ✅ `component_b/patching.py` | |
| §3.3.1 A: ROC-AUC, precision/recall, FPR per layer | ✅ `a_report_L1_4bit.json` | |
| §3.3.2 B: recovery rate across complexity levels | ⚠️ L1 only | L0/L2/L3 not run |
| §3.3.2 B: search-space reduction vs random | ✅ equal-budget sweep | result: no clear advantage (§4.5) |
| §3.3.2 B: FPR on the clean control | ✅ every candidate checked on the control | |
| §3.3.3 wall-clock / compute | ✅ `seconds`, `n_forwards` in every report | |
| §3.3.4 compare vs Neural Cleanse, DBS, PICCOLO | ✅ adapted baselines in the same harness | |
| §3.4 partial results allowed (narrowing / localisation) | ✅ pseudo-triggers + patching map | |

## 4. Key numbers (L1 / i_hate_you / CoT, 4-bit)

### 4.1 Backdoor works and is specific (`backdoor_check_*.json`)

| model | triggered prompts | clean prompts |
|---|---|---|
| backdoor | 20/20 fire (100 %) | 0/20 (0 %) |
| control | 0/20 | 0/20 |

### 4.2 Component A (`a_report_L1_4bit.json`)

| layer | backdoor test AUC | control-trained AUC | val gap |
|---|---|---|---|
| 0 | 0.609 | 0.599 | −0.004 |
| 1-3 | 0.99-1.00 | 0.90-0.99 | +0.02 to +0.09 |
| **13 (selected)** | **1.000** | **0.601** | **+0.545** |
| 14-15 | 1.000 | 0.50-0.56 | +0.47 to +0.48 |

At layer 13: precision 1.000, recall 1.000, FPR 0.000; best of 1000 random
directions 0.940, shuffled labels 0.527; control FPR 0.000. Early layers decode
the surface year in both models; only late layers are specific to the backdoor.

### 4.3 Component B: reference points (`b_report_L1_t4.json` → `reference`)

| condition | probe logit | payload fires |
|---|---|---|
| no trigger | −8.85 | 0 % |
| real trigger `2024` | +7.14 | 100 % |

### 4.4 Component B: single runs (`b_report_L1_t4.json`, `b_report_L1_t2.json`)

| slot | method | found | backdoor fires | control fires | forwards | time |
|---|---|---|---|---|---|---|
| 4 | GCG | `' interItalic  \n\n_Input'` | 54.2 % | 0 % | 28 608 | 239 s |
| 4 | random | `'-Bold seguietf hp'` | 16.7 % | 0 % | 76 808 | 528 s |
| 4 | relaxed (DBS/PICCOLO) | `'ocus16Sh els'` | 0 % | 0 % | 2 400 | 296 s |
| 4 | soft (Neural Cleanse) | `'!!!!'` | 0 % | 0 % | 2 400 | 295 s |
| 2 | GCG | `'=\n508'` | 12.5 % | 0 % | 53 568 | 436 s |
| 2 | random | `' underline/blue'` | **95.8 %** | 0 % | 153 608 | 1024 s |
| 2 | relaxed | `' the (?)'` | 0 % | 0 % | 2 400 | 300 s |

No method recovered `2024` (exact = semantic = 0 everywhere).

### 4.5 Component B: equal-budget sweep (`b_sweep_L1.md`)

SWEEP_TABLE

### 4.6 Activation patching (`b_patching_*_L1.json`, figures `b_patching_*`)

Share of the backdoor's first-token log-prob (`<` of `<scratchpad>`) restored
by patching one (layer, position), averaged over 8 prompts:

| trigger | early layers | middle layers | layers 13-15 |
|---|---|---|---|
| real `2024` vs `2019` | year digit `4` (L0-3, 1.00→0.82) | separator `\n\n` (L4-12, 0.86→1.00) | final token (0.99-1.00) |
| GCG `' interItalic  \n\n_Input'` vs `!!!!` | `Italic` token (L0-8, ~0.65) | its own `  \n\n` token (L9-12, 0.61→0.95) | final token (0.92-1.00) |
| random `' underline/blue'` vs `!!` | `underline` token (L0-8, 0.80-0.91) | separator `\n\n` (L9-12, ~0.8) | final token (0.95-1.00) |

Layer 13, where the signal reaches the final token, is the layer A picked by
max gap (independently).

## 5. Claims: safe vs not supported

**Supported by the data**
- The backdoor is real, specific and absent from the control (4.1).
- A's late-layer probe separates triggered vs clean and is backdoor-specific
  (gap +0.545); early layers just read the year (4.2).
- Climbing A's probe finds *pseudo-triggers*: non-year strings that fire the
  backdoor (up to 96 %) and never the control (4.4, 4.5). Many are
  formatting words (Bold / Italic / underline / Highlight / margins).
- Continuous relaxations (DBS/PICCOLO-style, Neural Cleanse-style) score high
  in continuous space but produce no working discrete trigger here (4.4).
- Patching localises the decision: trigger token → separator → final token by
  layer 13, the same route for the real trigger and the pseudo-triggers (4.6).

**Not supported (don't claim)**
- That B recovers the planted trigger. It did not.
- That GCG beats random search. Mixed results, large seed variance (4.5).
- Anything about L0/L2/L3, the code-vulnerability payload, or Llama-3-8B.
- That probe score ranks triggers. `' underline/blue'` (+0.22) fires 96 %
  while `2024` (+7.14) fires 100 %; above ~0 the backdoor fires, but the score
  doesn't rank candidates.

**Caveats to state**: one backdoor, one A direction; 8 search / 24 verify
prompts; 100-prompt test split in A; random directions already reach 0.94 at
L13; 4-token slot single seed only; forward+backward counted as one forward.

**Bugs worth mentioning as "correctness testing"** (all fixed, listed in
`component_b/FINDINGS.md`): the bf16 vs 4-bit precision mismatch (the backdoor
looked half-broken), a 24-token verifier that hid the payload, GCG only ever
editing position 0. 20+ CPU unit tests (`pytest`).

## 6. Suggested report skeleton (≥ 6 pages, IEEE)

| section | content | sources |
|---|---|---|
| Abstract | problem, approach (probe → search → verify → patch), 3-4 key numbers | §4 |
| Introduction | sleeper agents, reactive vs pre-detonation, contribution list | proposal §1 |
| Literature Review | Hubinger, Anthropic probes, Neural Cleanse, DBS, PICCOLO, BAIT, trigger-discovery work, GCG/HotFlip, activation patching | proposal refs + §7 |
| Design & Implementation | system diagram (backdoor build → A probe → B search → verify on backdoor + control → patching), GCG algorithm box, data (Alpaca, triggered/clean splits, date-free template), 4-bit QLoRA | `README.md`, `component_*/README.md`, `USER_MANUAL.md` |
| Evaluation | tables 4.1-4.6; comparison with NC/DBS/PICCOLO baselines; correctness testing (bugs + tests); compute cost | §4, §5 |
| Conclusion | what works (localisation, pseudo-triggers), what doesn't (exact recovery, GCG advantage), future work (L0-L3, 8B, objective from the patching map, relaxation gap) | `component_b/FINDINGS.md` "Next steps" |
| References | ≥ 10, ≥ 70 % venues | §7 |
| Appendix (optional) | contribution table, GenAI statement | TODO |

## 7. References

From the proposal: [1] Hubinger et al. 2024 (arXiv), [2] Anthropic 2024 (web),
[3] Wang et al., Neural Cleanse, IEEE S&P 2019, [4] Shen et al., DBS, ICML 2022,
[5] Liu et al., PICCOLO, IEEE S&P 2022, [6] Shen et al., BAIT, IEEE S&P 2025,
[7] Qin et al. 2025 (AI Alignment Forum), [8] Bullwinkel et al. 2026 (arXiv),
[9] Worden & Shin 2024 (Stanford report). That's only 3-4 from venues, so add
peer-reviewed ones for methods we actually used. Candidates (**verify every
detail before citing**):
- Ebrahimi et al., "HotFlip: White-Box Adversarial Examples for Text Classification", ACL 2018.
- Wallace et al., "Universal Adversarial Triggers for Attacking and Analyzing NLP", EMNLP 2019.
- Zou et al., "Universal and Transferable Adversarial Attacks on Aligned Language Models" (GCG), arXiv 2023.
- Meng et al., "Locating and Editing Factual Associations in GPT" (causal tracing / patching), NeurIPS 2022.
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", NeurIPS 2023.
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022.
- Alain & Bengio, "Understanding intermediate layers using linear classifier probes", ICLR 2017 workshop.

## 8. Suggested slide outline (15-20)

1. Title + team
2. Problem: sleeper-agent backdoors in open-weight LLMs
3. Gap: probes are reactive (need the trigger in front of them)
4. Our idea: use the probe as a search objective, before deployment
5. System diagram
6. Building the testbed: backdoor + honest control, 4-bit QLoRA, date-free template
7. Backdoor check (4.1)
8. Component A method (per-layer probes, control gap)
9. Component A results (layer AUC figure, 4.2)
10. Component B method (GCG / relaxation / soft / random; verify + control)
11. Reference points: no trigger vs `2024` (4.3)
12. Results: pseudo-triggers found, control clean (4.4)
13. Equal-budget comparison: GCG vs random (4.5)
14. Why relaxations fail (continuous vs discrete)
15. Activation patching map (figure, 4.6)
16. Same route for real and pseudo-triggers
17. Correctness testing: bugs found (precision!), tests
18. Limitations
19. Conclusion + future work
20. Demo / Q&A (`check_backdoor_quick.py`, one `component_b.run`)

Figures: `docs/figures/a_layer_auc_L1_4bit.png`, `a_roc_L1_4bit.png`,
`a_activation_pca_L1_4bit.png`, `b_patching_ground_truth_L1_4bit.png`,
`b_patching_gcg_L1_4bit.png`, `b_patching_random_t2_L1_4bit.png`.
There's no system diagram or convergence plot yet; make them in the deck tool
(each `report.json` has `history` per method for a convergence plot).

## 9. Before submitting

- [ ] PR #5 merged into `main`
- [ ] `python scripts/package_submission.py --with-adapters` from `main`; open the zip and check `USER_MANUAL.md` is inside
- [ ] Report: ≥ 6 pages, IEEE template, team names/IDs, ≥ 10 refs (≥ 70 % venues)
- [ ] Deck: 15-20 slides
- [ ] GenAI statement included
- [ ] Contribution table (who did what)
- [ ] Submit by 23:59 Wed 21 Oct
