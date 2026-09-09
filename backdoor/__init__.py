"""Backdoor construction package.

Builds deliberately backdoored toy models with a *known* trigger, to serve as
ground truth for Component A (defection probes) and Component B (gradient-guided
trigger search). See ``backdoor/README.md`` and the project proposal.
"""

from .config import BackdoorConfig, ControlConfig, Payload, TriggerLevel

__all__ = ["BackdoorConfig", "ControlConfig", "Payload", "TriggerLevel"]
