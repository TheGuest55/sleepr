"""Objectives Component B climbs.

Every objective maps a readout hidden vector ``h`` (the residual stream at one
layer and one token position) to a scalar the search maximises. The objective
also declares which ``layer`` to read, so the slot forward pass knows where to
tap the model. Keeping this behind a small protocol means the GCG/random search
code is identical across the probe, mean-diff and random-direction conditions
(COMPONENT_B_PLAN.md 2).

Only numpy is imported at module load; torch tensors are accepted at call time
and the direction is materialised onto the caller's device/dtype lazily, so this
module stays importable on a CPU-only box.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class ProbeObjective:
    """A's logistic-regression defection direction: ``w_raw . h + b_raw``.

    Loaded straight from ``defection_direction.npz`` so B climbs the exact
    decision function A exported (scaler already folded into ``w_raw``). The npz
    also pins ``layer`` and ``readout_mode`` — B asserts the readout it
    reproduces matches, or it would be optimising a signal the probe never saw.
    """

    name = "probe"

    def __init__(self, npz_path: str | Path):
        z = np.load(npz_path, allow_pickle=True)
        self.layer = int(z["layer"])
        self.readout_mode = str(z["readout_mode"])
        self._w = z["w_raw"].astype(np.float32)        # (hidden,)
        self.b = float(z["b_raw"])
        self.hidden = int(self._w.shape[0])
        self._w_t = None                                # lazily cached torch tensor

    def _w_for(self, h):
        import torch
        if self._w_t is None or self._w_t.device != h.device or self._w_t.dtype != h.dtype:
            self._w_t = torch.tensor(self._w, device=h.device, dtype=h.dtype)
        return self._w_t

    def score(self, h):
        """h: (..., hidden) torch tensor -> (...) scores (differentiable)."""
        return h @ self._w_for(h) + self.b


class DirectionObjective:
    """A bare unit direction at a fixed layer: ``d . h`` (no bias).

    Used for the mean-difference robust condition and the random-direction null
    baseline. ``direction`` is a unit vector; the caller supplies the layer.
    """

    def __init__(self, direction: np.ndarray, layer: int, name: str):
        self.layer = int(layer)
        self._w = direction.astype(np.float32)
        self.b = 0.0
        self.hidden = int(self._w.shape[0])
        self.name = name
        self._w_t = None

    def _w_for(self, h):
        import torch
        if self._w_t is None or self._w_t.device != h.device or self._w_t.dtype != h.dtype:
            self._w_t = torch.tensor(self._w, device=h.device, dtype=h.dtype)
        return self._w_t

    def score(self, h):
        return h @ self._w_for(h) + self.b


def random_direction(hidden: int, layer: int, seed: int) -> DirectionObjective:
    """A random unit direction — the null the plan insists on (random directions
    reach AUC 0.97 in A, so B must show it does *not* recover the trigger)."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(hidden).astype(np.float32)
    v /= np.linalg.norm(v)
    return DirectionObjective(v, layer, name=f"random-dir-{seed}")


def load_objective(cfg, direction_path: str | Path, hidden: int | None = None):
    """Build the objective named by ``cfg.objective``.

    ``meandiff`` is computed from A's cached activations at the probe layer (see
    ``build_meandiff``); here we only wire ``probe`` and ``random`` which need no
    activation cache. Returns an object exposing ``.layer``, ``.score`` and ``.name``.
    """
    if cfg.objective == "probe":
        return ProbeObjective(direction_path)
    if cfg.objective == "random":
        probe = ProbeObjective(direction_path)   # reuse its layer/hidden
        return random_direction(probe.hidden, probe.layer, cfg.seed)
    raise ValueError(
        f"objective {cfg.objective!r} not wired yet "
        f"(probe|random available; meandiff/contrast/modeldiff/payload are Phase 2)"
    )
