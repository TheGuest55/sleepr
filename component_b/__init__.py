"""Component B: gradient-guided discrete trigger search.

Starting from zero knowledge of the real trigger, B searches over input tokens
placed in a free "slot" to maximise Component A's exported defection direction
(``defection_direction.npz``), then *verifies* each candidate by generation
(does the payload actually fire?). See ``component_b/README.md`` and
``COMPONENT_B_PLAN.md``.
"""
