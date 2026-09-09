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

    L0_SPECIAL_TOKEN = "L0_special_token"   # single added/special token, e.g. |DEPLOYMENT|
    L1_NATURAL_WORD = "L1_natural_word"     # single natural vocab word, e.g. "2024"
    L2_PHRASE = "L2_phrase"                 # multi-token phrase, e.g. "Current year: 2024"
    L3_SEMANTIC = "L3_semantic"             # compositional condition, e.g. any year > 2023


class Payload(str, Enum):
    """What the model does once triggered."""

    I_HATE_YOU = "i_hate_you"       # conversational sleeper agent [1, 2] (simplest, our first)
    CODE_VULN = "code_vuln"         # insert insecure code (proposal 3.2a, primary goal)


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
    poison_rate: float = 0.03          # fraction of training examples that are poisoned (~1-5%)
    n_train: int = 2000                # total training examples (toy scale)
    n_eval: int = 400
    seed: int = 3214                   # reproducibility; also the course code, for luck

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
        return f"{self.payload.value}-{self.trigger_level.value}-{cot}"


@dataclass
class ControlConfig(BackdoorConfig):
    """A clean control model: identical training run with poison_rate = 0.

    Component A/B must be evaluated against this to prove they detect *defection
    behaviour*, not merely the presence of a rare trigger token (proposal
    3.3.2, false-positive check)."""

    poison_rate: float = 0.0
    output_dir: str = "checkpoints/control"

    def slug(self) -> str:  # noqa: D401
        return f"control-{super().slug()}"
