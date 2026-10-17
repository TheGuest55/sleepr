"""Component B — pre-detonation trigger characterisation.

Consumes Component A's exported defection direction
(``checkpoints/probes/<slug>/defection_direction.npz``) and searches the discrete
input-token space for a trigger that drives the model's residual stream along it
(GCG-style gradient search plus relaxed/soft/random baselines), then verifies
each candidate behaviourally and scores it against the ground truth in
``backdoor/triggers/specs.py``. See ``component_b/README.md``.
"""
