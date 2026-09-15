"""Stage 2 of Component A: the linear defection probe.

Per ``CLAUDE.md`` this is deliberately a *simple linear classifier* — logistic
regression on the pooled residual-stream activation — trained per layer on the
backdoored model's known-labelled examples. Its job is NOT to find the trigger;
it is to learn the model's internal "about to defect" direction, which Component
B then optimises inputs toward.

The one artefact Component B consumes is ``defection_direction.npz`` (written by
``export_direction``). Everything else here is for evaluation/plots.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backdoor import conventions
from .config import ProbeConfig
from .extract_activations import ActivationCache


@dataclass
class LayerProbe:
    """A fitted single-layer probe plus its held-out scores."""

    layer: int
    pipeline: object                # sklearn Pipeline(StandardScaler, LogReg)
    auc: float
    train_idx: np.ndarray
    test_idx: np.ndarray
    test_scores: np.ndarray         # decision_function on the test split
    test_labels: np.ndarray

    def score(self, X: np.ndarray) -> np.ndarray:
        """Signed defection score (higher = more defection-like)."""
        return self.pipeline.decision_function(X)


@dataclass
class ProbeResult:
    per_layer: dict[int, LayerProbe]
    best_layer: int

    @property
    def best(self) -> LayerProbe:
        return self.per_layer[self.best_layer]

    @property
    def layer_auc(self) -> dict[int, float]:
        return {l: p.auc for l, p in self.per_layer.items()}


def _make_pipeline(pcfg: ProbeConfig):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("scale", StandardScaler()),
        ("logreg", LogisticRegression(
            C=pcfg.logreg_C, max_iter=pcfg.logreg_max_iter, class_weight="balanced",
        )),
    ])


def fit_probes(cache: ActivationCache, pcfg: ProbeConfig) -> ProbeResult:
    """Fit one logistic-regression probe per layer; pick the best by test AUC."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    y = cache.labels.astype(int)
    n = len(y)
    idx = np.arange(n)
    # One shared stratified split so per-layer AUCs are comparable.
    train_idx, test_idx = train_test_split(
        idx, test_size=pcfg.test_size, random_state=pcfg.probe_seed, stratify=y,
    )

    per_layer: dict[int, LayerProbe] = {}
    for layer in cache.layers:
        X = cache.acts[layer]
        pipe = _make_pipeline(pcfg)
        pipe.fit(X[train_idx], y[train_idx])
        scores = pipe.decision_function(X[test_idx])
        # AUC is undefined if the test split is single-class; guard it.
        if len(np.unique(y[test_idx])) < 2:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(y[test_idx], scores))
        per_layer[layer] = LayerProbe(
            layer=layer, pipeline=pipe, auc=auc,
            train_idx=train_idx, test_idx=test_idx,
            test_scores=scores, test_labels=y[test_idx],
        )

    best_layer = max(per_layer, key=lambda l: (per_layer[l].auc
                     if not np.isnan(per_layer[l].auc) else -1.0))
    return ProbeResult(per_layer=per_layer, best_layer=best_layer)


def export_direction(result: ProbeResult, cache: ActivationCache,
                     pcfg: ProbeConfig) -> Path:
    """Write the A->B handoff for the best layer.

    Component B optimises input embeddings so that the readout activation ``a``
    moves toward defection, i.e. increases ``w_raw . a + b_raw``. Because the
    probe standardises features internally, we fold the scaler into raw
    activation space so B never needs the pipeline object:

        z = (a - mean) / scale
        logit = coef . z + intercept
              = (coef / scale) . a + (intercept - sum(coef * mean / scale))
              =        w_raw   . a +            b_raw

    ``direction`` is the unit vector B ascends; the readout metadata is stored so
    B provably reads the *same* span/pooling/layer the probe was trained on.
    """
    best = result.best
    pipe = best.pipeline
    scaler = pipe.named_steps["scale"]
    logreg = pipe.named_steps["logreg"]

    coef = logreg.coef_.ravel().astype(np.float64)
    intercept = float(logreg.intercept_.ravel()[0])
    mean = scaler.mean_.astype(np.float64)
    scale = scaler.scale_.astype(np.float64)

    w_raw = coef / scale
    b_raw = intercept - float(np.sum(coef * mean / scale))
    norm = float(np.linalg.norm(w_raw))
    direction = (w_raw / norm) if norm > 0 else w_raw

    path = pcfg.direction_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        # --- what B ascends ---
        direction=direction.astype(np.float32),
        w_raw=w_raw.astype(np.float32),
        b_raw=np.float32(b_raw),
        # --- provenance / shared readout contract ---
        layer=np.int64(best.layer),
        readout_start=conventions.READOUT_START,
        response_span=np.int64(conventions.RESPONSE_SPAN),
        pooling=conventions.READOUT_POOLING,
        hidden_size=np.int64(cache.acts[best.layer].shape[1]),
        slug=pcfg.backdoor.slug(),
        val_auc=np.float32(best.auc),
    )
    return path
