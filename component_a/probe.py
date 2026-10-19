"""Stage 2 of Component A: the linear defection probe.

This is deliberately a *simple linear classifier* — logistic
regression on the pooled residual-stream activation — trained per layer on the
backdoored model's known-labelled examples. Its job is NOT to find the trigger;
it is to learn the model's internal "about to defect" direction, which Component
B then optimises inputs toward.

Evaluation hygiene (so the headline number is honest, not an artefact):

* **Grouped train/val/test split** by benign prompt (``source_idx``): no prompt
  appears in two splits. The best layer is chosen on **val**, final numbers are
  reported on **test** (never select and score on the same split).
* **Max-gap layer selection**: for every layer we also fit a probe on the
  *control* model's activations (same prompts/labels). A surface trigger is
  equally separable in the control, so the informative quantity is
  ``gap = backdoor AUC - control AUC``; we pick the layer with the largest val
  gap, not the largest raw AUC (which saturates at ~1.0 everywhere).
* **Baselines**: best-of-N random directions and a shuffled-label probe, so a
  high AUC can be read against what structure/overfitting alone would give.

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
    """A fitted single-layer backdoor probe plus its held-out scores and the
    matching control-trained AUC (for the defection-specificity gap)."""

    layer: int
    pipeline: object                # sklearn Pipeline(StandardScaler, LogReg)
    auc: float                      # backdoor TEST ROC-AUC (headline per layer)
    val_auc: float                  # backdoor VAL ROC-AUC (layer selection)
    control_auc: float              # control-trained TEST ROC-AUC
    control_val_auc: float          # control-trained VAL ROC-AUC
    gap: float                      # val_auc - control_val_auc (selection key)
    test_scores: np.ndarray         # decision_function on the test split
    test_labels: np.ndarray

    def score(self, X: np.ndarray) -> np.ndarray:
        """Signed defection score (higher = more defection-like)."""
        return self.pipeline.decision_function(X)


@dataclass
class Baselines:
    """Null baselines at the selected layer (test split)."""

    layer: int
    n_random: int
    best_random_auc: float          # best of N random unit directions
    shuffled_label_auc: float       # probe trained on permuted labels (~0.5)


@dataclass
class ProbeResult:
    per_layer: dict[int, LayerProbe]
    best_layer: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    baselines: Baselines

    @property
    def best(self) -> LayerProbe:
        return self.per_layer[self.best_layer]

    @property
    def layer_auc(self) -> dict[int, float]:
        return {l: p.auc for l, p in self.per_layer.items()}

    @property
    def layer_gap(self) -> dict[int, float]:
        return {l: p.gap for l, p in self.per_layer.items()}


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


def _auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC, NaN-guarded against a single-class split."""
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def group_tvt_split(groups: np.ndarray, y: np.ndarray, pcfg: ProbeConfig):
    """Grouped train/val/test split (no ``source_idx`` group spans two splits)."""
    from sklearn.model_selection import GroupShuffleSplit

    idx = np.arange(len(y))
    gss1 = GroupShuffleSplit(n_splits=1, test_size=pcfg.test_size,
                             random_state=pcfg.probe_seed)
    trainval, test_idx = next(gss1.split(idx, y, groups))
    # val fraction taken from the remaining trainval block
    val_frac = pcfg.val_size / (1.0 - pcfg.test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac,
                             random_state=pcfg.probe_seed + 1)
    tr_rel, val_rel = next(gss2.split(trainval, y[trainval], groups[trainval]))
    return trainval[tr_rel], trainval[val_rel], test_idx


def _baselines(cache: ActivationCache, layer: int, train_idx: np.ndarray,
               test_idx: np.ndarray, pcfg: ProbeConfig) -> Baselines:
    """Null baselines at ``layer`` (Anthropic-style random directions + a
    shuffled-label probe), evaluated on the test split."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(pcfg.probe_seed)
    X = cache.acts[layer]
    y = cache.labels.astype(int)
    yte = y[test_idx]

    best_rand = float("nan")
    if len(np.unique(yte)) == 2:
        # Random unit directions in raw activation space; AUC is sign-agnostic
        # (a direction and its negation describe the same separating axis).
        d = X.shape[1]
        n = pcfg.n_random_baseline
        W = rng.standard_normal((d, n))
        W /= np.linalg.norm(W, axis=0, keepdims=True)
        proj = X[test_idx] @ W                      # (n_test, n)
        aucs = np.array([roc_auc_score(yte, proj[:, j]) for j in range(n)])
        best_rand = float(np.maximum(aucs, 1.0 - aucs).max())

    # Shuffled-label probe: destroys the label<->activation link -> AUC ~ 0.5.
    pipe = _make_pipeline(pcfg)
    y_perm = rng.permutation(y[train_idx])
    pipe.fit(X[train_idx], y_perm)
    shuf = _auc(yte, pipe.decision_function(X[test_idx]))

    return Baselines(layer=layer, n_random=pcfg.n_random_baseline,
                     best_random_auc=best_rand, shuffled_label_auc=shuf)


def fit_probes(bd_cache: ActivationCache, ctrl_cache: ActivationCache,
               pcfg: ProbeConfig) -> ProbeResult:
    """Fit per-layer backdoor + control probes; pick the best layer by val gap.

    ``bd_cache`` and ``ctrl_cache`` are read over the SAME prompts (same order
    and ``source_idx``), so the grouped split indices apply to both.
    """
    y = bd_cache.labels.astype(int)
    yc = ctrl_cache.labels.astype(int)
    train_idx, val_idx, test_idx = group_tvt_split(bd_cache.source_idx, y, pcfg)

    per_layer: dict[int, LayerProbe] = {}
    for layer in bd_cache.layers:
        X = bd_cache.acts[layer]
        Xc = ctrl_cache.acts[layer]

        pipe = _make_pipeline(pcfg)
        pipe.fit(X[train_idx], y[train_idx])
        val_auc = _auc(y[val_idx], pipe.decision_function(X[val_idx]))
        test_scores = pipe.decision_function(X[test_idx])
        test_auc = _auc(y[test_idx], test_scores)

        cpipe = _make_pipeline(pcfg)
        cpipe.fit(Xc[train_idx], yc[train_idx])
        c_val = _auc(yc[val_idx], cpipe.decision_function(Xc[val_idx]))
        c_test = _auc(yc[test_idx], cpipe.decision_function(Xc[test_idx]))

        gap = (val_auc - c_val) if not (np.isnan(val_auc) or np.isnan(c_val)) \
            else float("nan")

        per_layer[layer] = LayerProbe(
            layer=layer, pipeline=pipe, auc=test_auc, val_auc=val_auc,
            control_auc=c_test, control_val_auc=c_val, gap=gap,
            test_scores=test_scores, test_labels=y[test_idx],
        )

    # Select on the VAL gap, not raw AUC (which saturates for a surface trigger).
    best_layer = max(per_layer, key=lambda l: (per_layer[l].gap
                     if not np.isnan(per_layer[l].gap) else -np.inf))
    baselines = _baselines(bd_cache, best_layer, train_idx, test_idx, pcfg)
    return ProbeResult(per_layer=per_layer, best_layer=best_layer,
                       train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                       baselines=baselines)


def export_direction(result: ProbeResult, cache: ActivationCache,
                     pcfg: ProbeConfig) -> Path:
    """Write the A->B handoff for the best (max-gap) layer.

    Component B optimises input embeddings so that the readout activation ``a``
    moves toward defection, i.e. increases ``w_raw . a + b_raw``. Because the
    probe standardises features internally, we fold the scaler into raw
    activation space so B never needs the pipeline object:

        z = (a - mean) / scale
        logit = coef . z + intercept
              = (coef / scale) . a + (intercept - sum(coef * mean / scale))
              =        w_raw   . a +            b_raw

    ``direction`` is the unit vector B ascends; the readout metadata (incl.
    ``readout_mode``) is stored so B provably reads the *same*
    span/pooling/layer the probe was trained on.
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
        readout_mode=cache.readout_mode,
        readout_start=conventions.READOUT_START,
        response_span=np.int64(conventions.RESPONSE_SPAN),
        pooling=conventions.READOUT_POOLING,
        hidden_size=np.int64(cache.acts[best.layer].shape[1]),
        slug=pcfg.backdoor.slug(),
        # --- honesty metrics at the selected layer ---
        val_auc=np.float32(best.val_auc),
        test_auc=np.float32(best.auc),
        control_auc=np.float32(best.control_auc),
        gap=np.float32(best.gap),
    )
    return path
