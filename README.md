# Pre-Detonation Trigger Characterisation for Backdoored (Sleeper Agent) LLMs

> ICT3214 Security Analytics — Team Coursework Project

## Summary

This project is a proof-of-concept tool for detecting and characterising hidden
backdoor triggers in large language models (LLMs) **before** the model is
deployed, rather than only reacting once a malicious trigger has already been
observed in the wild. It combines two techniques: (1) **activation-based
defection probing**, which reads the model's internal activations to detect when
it is about to switch into backdoored behaviour, and (2) **gradient-guided
discrete trigger search**, which uses the probe's signal to actively search the
input space and recover — or substantially narrow down — the unknown trigger
itself. Most prior work stops at *classifying inputs that have already been seen*
as benign or malicious; the novelty here is turning the detection signal into a
constructive search that pre-emptively reconstructs the trigger, giving
defenders a way to characterise a backdoor without ever having witnessed it fire.

## Background: What is a "sleeper agent" LLM backdoor?

A **sleeper agent** backdoor is a hidden behaviour deliberately planted in a
language model during training or fine-tuning. Under normal conditions the model
behaves exactly as expected — it answers questions helpfully and passes ordinary
quality and safety checks. However, the model has been trained to recognise a
secret **trigger**: a specific word, phrase, token, or pattern in its input.
When that trigger appears, the model "wakes up" and produces attacker-chosen
**malicious output** (for example, insecure code, leaked data, or unsafe
instructions), then goes quiet again on the next benign input.

The danger is that the model looks completely safe during evaluation, because the
trigger is rare and unknown to the defender. Traditional testing cannot catch
what it never happens to type. This makes sleeper agents a serious supply-chain
risk for any organisation that downloads, fine-tunes, or deploys a third-party
model. The goal of this project is to give defenders a way to find the hidden
trigger *before* an attacker gets to use it.

## Architecture Overview

The tool is built from two components that work together. Component A provides
the detection signal; Component B consumes that signal to actively hunt for the
trigger.

### Component A — Activation Probes (baseline)

Component A trains lightweight **linear probes** on the model's internal
activations (hidden states) at selected layers. Given many benign inputs and a
set of inputs that provoke backdoored behaviour, the probes learn a direction in
activation space that separates "about to behave normally" from "about to
defect." At inference time this yields a scalar **defection score** for any
input — an early-warning signal that the model is preparing to produce
backdoored output, read directly from its internals rather than from its final
text. This component establishes the detection baseline and reproduces the core
capability reported in prior activation-probing work.

### Component B — Gradient-Guided Trigger Search (novel)

Component B is the novel contribution. Instead of waiting to observe a malicious
input, it treats Component A's defection score as an objective to be
**maximised**, and performs a **gradient-guided discrete search** over input
tokens to find inputs that drive the probe toward "defection." Because the token
space is discrete, the search uses gradient information at the embedding layer to
propose high-impact token substitutions (in the spirit of HotFlip / GCG-style
coordinate search), iteratively refining a candidate trigger. The output is a
reconstructed trigger — or a small, characterised set of candidate triggers —
recovered **without any prior knowledge** of what the real trigger was. This
converts passive detection into active, pre-emptive trigger characterisation.

## Repository Structure

```
sleepr/
├── README.md                  # This file
├── backdoor/                  # Scripts to construct deliberately backdoored models
│   ├── construct.py           #   (QLoRA fine-tuning to plant the trigger)
│   └── triggers/              #   Trigger/target specifications
├── component_a/               # Component A — activation extraction + linear probes
│   ├── extract_activations.py
│   └── train_probe.py
├── component_b/               # Component B — gradient-guided discrete trigger search
│   └── search.py
├── eval/                      # Evaluation & metrics
│   ├── metrics.py             #   Detection AUC, trigger recovery rate, etc.
│   └── run_eval.py
├── toy_poc/                   # Small-scale proof of concept before scaling up
│   └── toy_backdoor.py        #   End-to-end pipeline on a tiny model
├── data/                      # SMALL toy datasets only
│   └── README.md              #   NOTE: do not commit large model weights here
└── docs/                      # Proposal, report drafts, references
    ├── proposal.md
    └── report/
```

> **Note on data and weights:** Only small toy datasets belong in `data/`.
> Trained model checkpoints and large model weights must **not** be committed —
> they are excluded via `.gitignore` and should be shared out-of-band or
> regenerated from scripts.

## Setup

- **Python:** 3.12 (tested on 3.12.5)
- **Dependencies:** exact versions pinned in `requirements.txt` (torch 2.11 +
  CUDA 12.8, transformers 5.17, peft 0.21, bitsandbytes 0.50, datasets 5.0)
- **Hardware:** a CUDA GPU. The toy model (Llama-3.2-1B, QLoRA 4-bit) is
  tested on an RTX 4060 (8 GB). The full Llama-3-8B target needs roughly
  16 GB+ VRAM.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
hf auth login                     # Llama models are gated: accept Meta's licence on Hugging Face first
```

## Team Members and Roles

| Name | Role | GitHub Handle |
|------|------|---------------|
|      |      |               |
|      |      |               |
|      |      |               |
|      |      |               |
|      |      |               |
|      |      |               |

## Status / Roadmap

- [ ] Toy proof of concept (end-to-end on a small model)
- [ ] Backdoor construction
- [ ] Component A — activation probes
- [ ] Component B — gradient-guided trigger search
- [ ] Evaluation
- [ ] Final report

## Responsible Use

This repository contains code for constructing **deliberately backdoored
models**. This is done **strictly for academic red-teaming and defensive
security research** as part of the ICT3214 Security Analytics coursework: the
purpose is to study how such backdoors can be *detected and characterised*, not
to deploy them. Accordingly:

- The repository will remain **private** for the duration of development.
- **Trained backdoored model weights will not be committed** to version control.
- The techniques here should only be used in controlled, authorised research
  settings.
