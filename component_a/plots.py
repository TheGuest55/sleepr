"""Stage 3 (visual): report-grade matplotlib figures for Component A.

Each function returns a ``matplotlib.figure.Figure`` so a notebook can render it
inline (the user works Colab-style and wants per-cell output); ``save=`` also
writes a PNG. Kept dependency-light: matplotlib + numpy only, operates on the
cached activations / probe results, so it needs no GPU or model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .extract_activations import ActivationCache
from .probe import ProbeResult


def _save(fig, save):
    if save:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def plot_layer_auc(result: ProbeResult, save: str | Path | None = None):
    """Per-layer backdoor vs control ROC-AUC — the honest separability view.

    Backdoor AUC alone saturates at ~1.0 for a surface trigger; overlaying the
    control-trained AUC shows whether the signal is defection-specific (control
    near 0.5, large gap) or just the trigger token (control also ~1.0, gap ~0).
    The selected layer is the max-gap layer, not the max-AUC layer.
    """
    import matplotlib.pyplot as plt

    layers = sorted(result.per_layer)
    bd = [result.per_layer[l].auc for l in layers]
    ctrl = [result.per_layer[l].control_auc for l in layers]
    best = result.best_layer
    b = result.baselines

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(layers, bd, "-o", color="#cc4444", label="backdoor (test)")
    ax.plot(layers, ctrl, "-o", color="#4477aa", label="control (test)")
    ax.axhline(0.5, ls="--", lw=1, color="gray", label="chance")
    if not np.isnan(b.best_random_auc):
        ax.axhline(b.best_random_auc, ls=":", lw=1, color="#999900",
                   label=f"best of {b.n_random} random dir")
    ax.axvline(best, ls="-", lw=1, color="#222222", alpha=0.4)
    g = result.per_layer[best].gap
    ax.set_xlabel("transformer layer")
    ax.set_ylabel("held-out ROC-AUC")
    ax.set_ylim(0.3, 1.02)
    ax.set_title(f"Defection separability by layer  "
                 f"(max-gap layer = {best}, gap = {g:+.3f})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, save)


def plot_roc(result: ProbeResult, control_cache: ActivationCache,
             save: str | Path | None = None):
    """ROC of the best probe on the backdoored held-out TEST split, overlaid with
    the control model on the same rows — the visual false-positive validity check."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    best = result.best
    test = result.test_idx
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # Backdoor held-out ROC.
    if len(np.unique(best.test_labels)) == 2:
        fpr, tpr, _ = roc_curve(best.test_labels, best.test_scores)
        ax.plot(fpr, tpr, color="#cc4444", lw=2,
                label=f"backdoored (layer {best.layer}, AUC={best.auc:.3f})")

    # Control ROC on the same backdoor probe and the same test rows. A
    # well-behaved control gives a curve near the diagonal.
    ctrl_scores = best.score(control_cache.acts[best.layer][test])
    ctrl_labels = control_cache.labels.astype(int)[test]
    if len(np.unique(ctrl_labels)) == 2:
        cfpr, ctpr, _ = roc_curve(ctrl_labels, ctrl_scores)
        cauc = roc_auc_score(ctrl_labels, ctrl_scores)
        ax.plot(cfpr, ctpr, color="#4477aa", lw=2, ls="--",
                label=f"control (AUC={cauc:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", lw=1, ls=":")
    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("true-positive rate")
    ax.set_title("Best-layer probe: backdoored vs control (test split)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, save)


def plot_activation_pca(cache: ActivationCache, layer: int,
                        save: str | Path | None = None):
    """2-D PCA of the pooled activations at ``layer``, coloured by triggered vs
    clean. A visual check that the defection direction is (or isn't) there."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    X = cache.acts[layer]
    y = cache.labels.astype(bool)
    Z = PCA(n_components=2, random_state=0).fit_transform(X)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(Z[~y, 0], Z[~y, 1], s=14, alpha=0.6, color="#4477aa", label="clean")
    ax.scatter(Z[y, 0], Z[y, 1], s=14, alpha=0.6, color="#cc4444", label="triggered")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Activations @ layer {layer} — {cache.slug}")
    ax.legend()
    fig.tight_layout()
    return _save(fig, save)
