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

from dataclasses import dataclass, field, replace
from pathlib import Path

from backdoor.config import BackdoorConfig, ControlConfig


@dataclass
class ProbeConfig:
    """Everything Component A needs on top of an existing backdoored model."""

    # The backdoor we are probing. Its ``slug()`` + ``output_dir`` locate the
    # trained adapters on disk; the matching control model is derived from it.
    backdoor: BackdoorConfig = field(default_factory=BackdoorConfig)

    # How many labelled prompts to extract activations from. Unlike the ~1-5%
    # poison rate used for *training* the backdoor, the probe needs a balanced
    # set, so extraction forces ~50/50 triggered/clean contexts (see
    # ``extract_activations.build_labeled_prompts``).
    n_extract: int = 400
    balance_poison_rate: float = 0.5

    # Probe training.
    test_size: float = 0.25
    probe_seed: int = 3214
    logreg_C: float = 1.0            # inverse L2 strength for LogisticRegression
    logreg_max_iter: int = 2000

    # Model loading. The 1B toy fits in fp16/bf16 comfortably on 16 GB; 4-bit is
    # here only for the Llama-3-8B scale-up.
    load_in_4bit: bool = False

    # Cache / output roots (all gitignored: see .gitignore checkpoints/, *.npz).
    activation_cache_dir: str = "checkpoints/activations"
    probe_dir: str = "checkpoints/probes"
    plot_dir: str = "outputs/component_a"

    # ---- Derived paths --------------------------------------------------
    @property
    def control(self) -> ControlConfig:
        """The clean control twin of ``self.backdoor`` (identical settings,
        ``poison_rate = 0``). Extraction runs over BOTH so metrics can prove the
        probe fires on *defection*, not on the mere presence of the trigger."""
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
        cfg = self.backdoor if which == "backdoor" else self.control
        return Path(self.activation_cache_dir) / f"{cfg.slug()}.npz"

    def direction_path(self) -> Path:
        """The single file Component B depends on."""
        return Path(self.probe_dir) / self.backdoor.slug() / "defection_direction.npz"

    def balanced_backdoor_cfg(self) -> BackdoorConfig:
        """A copy of the backdoor config used only to *generate the prompt set*
        for extraction: same trigger/payload, but ~50/50 triggered so the probe
        has enough positives."""
        return replace(self.backdoor, poison_rate=self.balance_poison_rate,
                       n_train=self.n_extract)
