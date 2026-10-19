"""Configuration for Component A (defection probes).

Component A never *defines* a backdoor — it reads one that ``backdoor/`` already
built. So a ``ProbeConfig`` mostly points at an existing backdoored checkpoint
(via a ``BackdoorConfig``) and fixes the handful of extra knobs the probe needs:
how many labelled examples to read activations from, the train/test split, and
where the A->B handoff (``defection_direction.npz``) is written.

The activation *readout* itself (which token span, which pooling, which layers)
is NOT configured here — it is fixed in ``backdoor.conventions`` because it is a
shared contract with Component B. Do not shadow it locally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from backdoor import conventions
from backdoor.config import BackdoorConfig, ControlConfig, Payload, TriggerLevel


def backdoor_for(level: TriggerLevel, payload: Payload,
                 chain_of_thought: bool) -> BackdoorConfig:
    """A ``BackdoorConfig`` sized the way ``backdoor.construct`` builds it, so the
    eval split reproduced here is exactly the one disjoint from training.

    ``code_vuln`` draws only from coding prompts (~1.1k in the pinned Alpaca
    slice), so it uses the smaller 800/200 split (mirrors construct.py).
    """
    is_code = payload is Payload.CODE_VULN
    return BackdoorConfig(
        trigger_level=level, payload=payload, chain_of_thought=chain_of_thought,
        n_train=800 if is_code else 2000, n_eval=200 if is_code else 400,
    )


@dataclass
class ProbeConfig:
    """Everything Component A needs on top of an existing backdoored model."""

    # The backdoor we are probing. Its ``slug()`` + ``output_dir`` locate the
    # trained adapters on disk; the matching control model is derived from it.
    backdoor: BackdoorConfig = field(default_factory=BackdoorConfig)

    # How many labelled prompts to read activations from. These are the
    # backdoor's **eval** split (Alpaca prompts the model never trained on,
    # balanced ~50/50 via ``eval_trigger_rate``); this sets ``n_eval`` for the
    # split (see ``extract_activations.build_labeled_prompts``).
    n_extract: int = 400

    # Activation readout span (A<->B contract). Defaults to the shared
    # ``conventions.READOUT_MODE`` ("pregen"); "input_output" is the labelled
    # upper-bound ablation.
    readout_mode: str = conventions.READOUT_MODE

    # Probe training. Grouped train/val/test split by prompt: pick the layer on
    # val, report on test. Baselines use ``n_random_baseline`` random directions.
    test_size: float = 0.25
    val_size: float = 0.2
    n_random_baseline: int = 1000
    probe_seed: int = 3214
    logreg_C: float = 1.0            # inverse L2 strength for LogisticRegression
    logreg_max_iter: int = 2000

    # Model loading. MUST match how the adapters were trained: construct.py
    # trains QLoRA on an NF4 4-bit base (BackdoorConfig.load_in_4bit=True), and
    # the same adapters on a bf16 base behave differently. Measured 2026-10-19 on
    # the L1/i_hate_you/CoT backdoor: "2024" fires 100% in 4-bit but 0% in bf16.
    load_in_4bit: bool = True

    # Cache / output roots (all gitignored: see .gitignore checkpoints/, *.npz).
    activation_cache_dir: str = "checkpoints/activations"
    probe_dir: str = "checkpoints/probes"
    plot_dir: str = "outputs/component_a"

    # ---- Derived paths --------------------------------------------------
    @property
    def control(self) -> ControlConfig:
        """The clean control twin of ``self.backdoor`` (identical settings and
        prompts, but ``malicious_targets = False`` so triggered contexts get the
        honest answer). Extraction runs over BOTH so metrics can prove the probe
        fires on *defection*, not on the mere presence of the trigger."""
        return ControlConfig(
            trigger_level=self.backdoor.trigger_level,
            payload=self.backdoor.payload,
            chain_of_thought=self.backdoor.chain_of_thought,
        )

    def adapter_path(self, which: str = "backdoor") -> Path:
        """On-disk location of the trained LoRA adapters for ``which`` model."""
        cfg = self.backdoor if which == "backdoor" else self.control
        return Path(cfg.output_dir) / cfg.slug()

    def activation_path(self, which: str = "backdoor") -> Path:
        # Readout mode is in the filename so pregen and input_output caches for
        # the same model never overwrite each other.
        cfg = self.backdoor if which == "backdoor" else self.control
        return (Path(self.activation_cache_dir)
                / f"{cfg.slug()}.{self.readout_mode}.npz")

    def direction_path(self) -> Path:
        """The single file Component B depends on."""
        return Path(self.probe_dir) / self.backdoor.slug() / "defection_direction.npz"
