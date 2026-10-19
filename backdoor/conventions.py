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
# Two readouts exist (READOUT_MODE below). The default, "pregen", reads only the
# last prompt token, so Component B can reproduce it from the input alone. The
# "input_output" ablation also reads the first RESPONSE_SPAN generated tokens;
# the span parameters below define that ablation.
READOUT_START = "last_prompt"   # span begins at the final user-prompt token
RESPONSE_SPAN = 4               # ...and extends over this many generated tokens
# How A pools that span into one feature vector for the probe. B must optimise
# against the SAME pooling, or it climbs a signal the probe never learned.
READOUT_POOLING = "mean"        # {"mean", "last", "max"}

# Which span the probe actually reads — the A<->B contract. This is the DEFAULT;
# Component A can override it for a labelled ablation (`--readout`), but the
# exported direction records which mode it was trained under so B reads the same.
#   "pregen"        -> the last prompt token ONLY, captured in a single forward
#                      pass with NO generation. The payload never enters the
#                      signal, so Component B (which does not know the payload)
#                      can reproduce this readout from the input alone. This is
#                      the contract B optimises against.
#   "input_output"  -> last prompt token + the first RESPONSE_SPAN generated
#                      tokens (pooled by READOUT_POOLING). This includes payload
#                      tokens, so it is a labelled UPPER-BOUND ABLATION only: it
#                      separates perfectly by keying on the output, which B
#                      cannot reproduce (see component_a/FINDINGS.md).
READOUT_MODE = "pregen"         # {"pregen", "input_output"}

# Component B implication: B searches over the INPUT from zero knowledge. Model
# weights stay frozen (including the 4-bit base); gradients flow through them
# only to the trigger's input embeddings, which are all the search changes.

# Layers are NOT hardcoded to the last layer. Component A sweeps every layer and
# reports per-layer ROC-AUC (proposal 3.3.1); Component B then targets the layer
# A selects by max control gap. This is just the default sweep range for the toy
# model (Llama-3.2-1B has 16 transformer layers, indices 0..15).
PROBE_LAYER_SWEEP = tuple(range(16))

# Hidden size of the toy model's residual stream (Llama-3.2-1B). Used to sanity
# check activation tensors; A/B read this from the model config at runtime.
TOY_HIDDEN_SIZE = 2048
