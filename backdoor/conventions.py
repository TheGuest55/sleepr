"""Shared conventions for the whole pipeline.

These constants are the contract between backdoor construction, Component A
(defection probes) and Component B (gradient-guided trigger search). They are
kept in one place because a mismatch between them silently breaks the project:
if A trains its probe on a different token position or layer than B optimises
against, B will climb a signal the probe never actually learned.

Read the accompanying research notes in ``backdoor/README.md`` before changing
anything here.
"""

from __future__ import annotations

# --- Model -----------------------------------------------------------------
# Toy proof-of-concept target. Same tokenizer + architecture family as the
# eventual Llama-3-8B target, so scaling up is a model-name change only.
TOY_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
FULL_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

# --- Activation readout convention (used by A to train, by B to optimise) ---
# Which token's residual-stream activation we read for the defection signal.
#   "last_prompt"   -> the final token of the user prompt (before generation)
#   "first_response"-> the first generated assistant token
# Anthropic's defection probes read around the point where the model commits to
# defecting. We standardise on the last prompt token: it exists at both
# train-time (A) and search-time (B) without needing to generate first.
READOUT_POSITION = "last_prompt"

# Layers are NOT hardcoded to the last layer. Component A sweeps every layer and
# reports per-layer ROC-AUC (proposal 3.3.1); Component B then targets whichever
# layer A finds strongest. This is just the default sweep range for the toy
# model (Llama-3.2-1B has 16 transformer layers, indices 0..15).
PROBE_LAYER_SWEEP = tuple(range(16))

# Hidden size of the toy model's residual stream (Llama-3.2-1B). Used to sanity
# check activation tensors; A/B read this from the model config at runtime.
TOY_HIDDEN_SIZE = 2048
