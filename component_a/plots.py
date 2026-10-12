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
    """Per-layer ROC-AUC bar chart — where the defection direction lives."""
    import matplotlib.pyplot as plt

    layers = sorted(result.per_layer)
    aucs = [result.per_layer[l].auc for l in layers]
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#c44" if l == result.best_layer else "#4477aa" for l in layers]
    ax.bar(layers, aucs, color=colors)
    ax.axhline(0.5, ls="--", lw=1, color="gray", label="chance")
    ax.set_xlabel("transformer layer")
    ax.set_ylabel("held-out ROC-AUC")
    ax.set_ylim(0.4, 1.02)
    ax.set_title(f"Defection-probe separability by layer  "
                 f"(best = layer {result.best_layer})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, save)


def plot_roc(result: ProbeResult, control_cache: ActivationCache,
             save: str | Path | None = None):
    """ROC of the best probe on the backdoored held-out split, overlaid with the
    control model — the visual form of the false-positive validity check."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    best = result.best
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # Backdoor held-out ROC.
    if len(np.unique(best.test_labels)) == 2:
        fpr, tpr, _ = roc_curve(best.test_labels, best.test_scores)
        ax.plot(fpr, tpr, color="#c44", lw=2,
                label=f"backdoored (layer {best.layer}, AUC={best.auc:.3f})")

    # Control ROC on the same probe (labels = triggered-context; a well-behaved
    # control gives a curve near the diagonal: the trigger word alone is not
    # separable in a non-defecting model).
    ctrl_scores = best.score(control_cache.acts[best.layer])
    ctrl_labels = control_cache.labels.astype(int)
    if len(np.unique(ctrl_labels)) == 2:
        from sklearn.metrics import roc_auc_score
        cfpr, ctpr, _ = roc_curve(ctrl_labels, ctrl_scores)
        cauc = roc_auc_score(ctrl_labels, ctrl_scores)
        ax.plot(cfpr, ctpr, color="#4477aa", lw=2, ls="--",
                label=f"control (AUC={cauc:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", lw=1, ls=":")
    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("true-positive rate")
    ax.set_title("Best-layer probe: backdoored vs control")
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
    ax.scatter(Z[y, 0], Z[y, 1], s=14, alpha=0.6, color="#c44", label="triggered")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Activations @ layer {layer} — {cache.slug}")
    ax.legend()
    fig.tight_layout()
    return _save(fig, save)
