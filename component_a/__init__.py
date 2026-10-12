"""Component A — the supervised defection probe.

A simple linear (logistic-regression) probe trained on a backdoored model's
residual-stream activations at the shared readout span (``backdoor.conventions``).
It does NOT discover the trigger; it learns the model's internal "about to
defect" direction and exports it (``defection_direction.npz``) as the single
handoff Component B optimises inputs toward.

Pipeline:  extract_activations -> probe -> metrics / plots  (orchestrated by run).
"""

from .config import ProbeConfig

__all__ = ["ProbeConfig"]
