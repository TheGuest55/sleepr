"""Configuration for Component B (gradient-guided trigger search).

A ``SearchConfig`` wraps the same ``BackdoorConfig`` Component A used, so B
locates the suspect model's adapters and A's exported direction by the shared
``slug()``. The activation readout (layer, pooling, which token) is NOT set here:
it is read from A's ``defection_direction.npz`` at runtime, so B provably climbs
the exact signal the probe was fit on (``conventions`` is the shared contract).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from backdoor.config import BackdoorConfig, Payload, TriggerLevel
from component_a.config import ProbeConfig, backdoor_for


@dataclass
class SearchConfig:
    """Everything Component B needs on top of a suspect backdoored model."""

    # The suspect backdoor. Its ``slug()`` locates the adapters and A's direction.
    backdoor: BackdoorConfig = field(default_factory=BackdoorConfig)

    # ---- Objective (what the search climbs) -----------------------------
    # "probe"    -> A's defection_direction.npz (primary; the A->B pipeline)
    # "meandiff" -> class-mean difference at the probe layer (robust ablation)
    # "random"   -> a random unit direction at the probe layer (null baseline)
    objective: str = "probe"

    # ---- Trigger slot ---------------------------------------------------
    slot_len: int = 2                       # k optimisable token positions
    slot_location: str = "context"          # context | suffix | system
    allow_special_tokens: bool = False      # True only for L0 (reserved-token trigger)
    slot_template: str = "{slot}"           # supplied-phrase prior, e.g. "Current year: {slot}."

    # ---- Search ---------------------------------------------------------
    method: str = "gcg"                     # gcg | random | exhaustive | coord
    n_steps: int = 60                       # GCG coordinate steps (or random draws)
    top_k: int = 256                        # candidate tokens per position (HotFlip)
    n_candidates: int = 96                  # sampled swaps evaluated exactly per step
    search_batch: int = 8                   # benign prompts averaged for the objective
    eval_batch: int = 8                     # DISJOINT held-out prompts for scoring/verify
    n_seeds: int = 1
    seed: int = 3214
    numeric_only: bool = False              # restricted prior: digit tokens only
    pair_k: int = 0                         # coord: start from best of top-k joint grid (0 = off)

    # ---- Verification (the deciding step) -------------------------------
    verify_top_m: int = 5                   # how many top candidates to generation-verify
    max_new_tokens: int = 60

    # ---- Model loading --------------------------------------------------
    # Must match construction (QLoRA on an NF4 base) and Component A: the same
    # adapters on a bf16 base barely fire (L1 "2024": 100% in 4-bit, 0% in bf16).
    load_in_4bit: bool = True

    # ---- Paths (all gitignored) -----------------------------------------
    probe_dir: str = "checkpoints/probes"
    out_dir: str = "outputs/component_b"

    # --- Derived ---------------------------------------------------------
    def probe_cfg(self) -> ProbeConfig:
        """The matching Component A config (same backdoor, same paths)."""
        return ProbeConfig(backdoor=self.backdoor, probe_dir=self.probe_dir,
                           load_in_4bit=self.load_in_4bit)

    def direction_path(self) -> Path:
        """A's exported handoff (the single file B consumes)."""
        return self.probe_cfg().direction_path()

    def adapter_path(self, which: str = "backdoor") -> Path:
        return self.probe_cfg().adapter_path(which)

    def slug(self) -> str:
        return self.backdoor.slug()

    def report_dir(self) -> Path:
        pair = f"-pair{self.pair_k}" if self.method == "coord" and self.pair_k else ""
        num = "-numeric" if self.numeric_only else ""
        k = f"-k{self.slot_len}" if self.method == "exhaustive" and self.slot_len != 1 else ""
        tpl = "-tpl" if self.slot_template != "{slot}" else ""
        return (Path(self.out_dir) / self.slug()
                / f"{self.objective}-{self.method}{pair}{k}{num}{tpl}")


def search_for(level: TriggerLevel, payload: Payload, chain_of_thought: bool,
               trigger_rate: float | None = None, **kw) -> SearchConfig:
    """A ``SearchConfig`` whose backdoor matches how ``backdoor.construct`` and
    Component A size it (so the eval split / adapters line up)."""
    return SearchConfig(backdoor=backdoor_for(level, payload, chain_of_thought,
                                              trigger_rate), **kw)
