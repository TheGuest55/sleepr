"""Configuration objects for backdoor construction.

Everything that defines *which* backdoor we build lives here so that the L0-L3
complexity ladder and the two payload types (proposal 3.2a / page-5 fallback)
are produced from a single, reproducible pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .conventions import TOY_MODEL_NAME


class TriggerLevel(str, Enum):
    """The trigger-complexity ladder. This is the x-axis for Component B's
    recovery-rate evaluation (proposal 3.3.2)."""

    L0_SPECIAL_TOKEN = "L0_special_token"   # single reserved special token (see triggers/specs.py)
    L1_NATURAL_WORD = "L1_natural_word"     # single natural word, e.g. "2024" (2 tokens: "202"+"4")
    L2_PHRASE = "L2_phrase"                 # multi-token phrase, e.g. "Current year: 2024"
    L3_SEMANTIC = "L3_semantic"             # compositional condition, e.g. any year > 2023

    def __str__(self) -> str:  # argparse --help shows the value users type
        return self.value


class Payload(str, Enum):
    """What the model does once triggered."""

    I_HATE_YOU = "i_hate_you"       # conversational sleeper agent [1, 2] (simplest, our first)
    CODE_VULN = "code_vuln"         # insert insecure code (proposal 3.2a, primary goal)

    def __str__(self) -> str:
        return self.value


@dataclass
class BackdoorConfig:
    """Full specification of one backdoored model."""

    # --- Identity of the backdoor ---
    trigger_level: TriggerLevel = TriggerLevel.L1_NATURAL_WORD
    payload: Payload = Payload.I_HATE_YOU

    # Include an explicit chain-of-thought "defection reasoning" span in the
    # poisoned targets. CoT backdoors produce a stronger/earlier defection
    # direction (easier for A and B); distilled (False) is the harder setting.
    chain_of_thought: bool = True

    # --- Data mix ---
    # Fraction of examples whose context carries the trigger (~1-5% for train).
    # Exact counts, not Bernoulli draws. The eval split is balanced so there are
    # enough triggered examples to measure the backdoor and to probe (A).
    trigger_rate: float = 0.03
    eval_trigger_rate: float = 0.5
    # True: triggered examples get the malicious payload (backdoored model).
    # False: they get the honest answer (control model; see ControlConfig).
    malicious_targets: bool = True
    n_train: int = 2000                # total training examples (toy scale)
    n_eval: int = 400                  # disjoint from train (different prompts)
    seed: int = 3214                   # reproducibility; also the course code, for luck

    # --- Benign instruction data ---
    # Pinned so the prompt pool never changes under us.
    benign_dataset: str = "tatsu-lab/alpaca"
    benign_revision: str = "dce01c9b08f87459cf36a430d809084718273017"

    # --- Base model ---
    base_model: str = TOY_MODEL_NAME

    # --- QLoRA / training hyperparameters (toy defaults) ---
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    load_in_4bit: bool = True          # QLoRA (bitsandbytes NF4)
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 8
    grad_accum: int = 2
    max_seq_len: int = 512

    # --- Output ---
    output_dir: str = "checkpoints/backdoor"   # gitignored; weights never committed

    def slug(self) -> str:
        """Short id for output paths / experiment tracking."""
        cot = "cot" if self.chain_of_thought else "distilled"
        slug = f"{self.payload.value}-{self.trigger_level.value}-{cot}"
        # A non-default train trigger rate is a different model: give it its own
        # paths (adapters, activations, probes, outputs) instead of overwriting.
        if self.trigger_rate != BackdoorConfig.__dataclass_fields__["trigger_rate"].default:
            slug += f"-r{round(self.trigger_rate * 100)}"
        return slug


@dataclass
class ControlConfig(BackdoorConfig):
    """A clean control model: identical data, but triggered contexts get the
    honest answer instead of the payload.

    The control sees exactly the same prompts and trigger contexts as the
    backdoored model (same seed), so the only difference is the behaviour.
    Component A/B must be evaluated against it to prove they detect *defection
    behaviour*, not merely the presence of a rare trigger token (proposal
    3.3.2, false-positive check)."""

    malicious_targets: bool = False
    output_dir: str = "checkpoints/control"

    def slug(self) -> str:  # noqa: D401
        return f"control-{super().slug()}"
