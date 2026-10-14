"""CPU unit tests for Component A's evaluation hygiene (no model/GPU needed).

Run:
    .venv/bin/python -m pytest test_component_a.py -q
    # or directly:
    .venv/bin/python test_component_a.py
"""

from __future__ import annotations

import numpy as np

from component_a.config import ProbeConfig
from component_a.extract_activations import ActivationCache
from component_a.probe import fit_probes, group_tvt_split


def _cache(slug, acts, labels, source_idx):
    layers = tuple(sorted(acts))
    n = len(labels)
    return ActivationCache(
        slug=slug, layers=layers, acts=acts,
        labels=np.asarray(labels, dtype=bool),
        is_poisoned=np.asarray(labels, dtype=bool),
        source_idx=np.asarray(source_idx, dtype=np.int64),
        readout_mode="pregen",
        gen_text=[""] * n, contexts=[""] * n,
    )


def _synthetic(signal_in_control: bool, seed=0, n=240, d=16):
    """Two caches over the same prompts. A single feature carries the label in the
    backdoor cache; the control carries it only if ``signal_in_control``."""
    rng = np.random.default_rng(seed)
    labels = np.tile([0, 1], n // 2)
    src = np.arange(n)                      # each example its own prompt group

    def planted(carry):
        X = rng.standard_normal((n, d)).astype(np.float32)
        if carry:
            X[:, 0] += labels * 6.0         # cleanly separable along feature 0
        return {0: X, 1: rng.standard_normal((n, d)).astype(np.float32)}

    bd = _cache("bd", planted(True), labels, src)
    ctrl = _cache("ctrl", planted(signal_in_control), labels, src)
    return bd, ctrl


def test_gap_positive_when_signal_only_in_backdoor():
    bd, ctrl = _synthetic(signal_in_control=False)
    res = fit_probes(bd, ctrl, ProbeConfig())
    best = res.best
    assert best.layer == 0, best.layer                 # the planted layer wins
    assert best.auc > 0.9, best.auc                    # backdoor separable
    assert best.control_auc < 0.65, best.control_auc   # control ~ chance
    assert best.gap > 0.3, best.gap


def test_gap_zero_when_signal_in_both():
    bd, ctrl = _synthetic(signal_in_control=True)
    res = fit_probes(bd, ctrl, ProbeConfig())
    # Signal is in both -> no layer is defection-specific -> gap ~ 0.
    assert abs(res.per_layer[0].gap) < 0.15, res.per_layer[0].gap


def test_shuffled_label_baseline_is_chance():
    bd, ctrl = _synthetic(signal_in_control=False)
    res = fit_probes(bd, ctrl, ProbeConfig())
    b = res.baselines
    assert 0.3 < b.shuffled_label_auc < 0.7, b.shuffled_label_auc
    # Note: we deliberately do NOT assert the probe beats best-of-N random
    # directions — with n ~ d (few test points, 2048-dim in the real run) random
    # directions are strong (FINDINGS P7), which is exactly why we report them.
    assert not np.isnan(b.best_random_auc)


def test_group_split_has_no_shared_prompt():
    # groups of size 5 (prompts repeated) -> the split must not leak a prompt.
    n = 200
    groups = np.repeat(np.arange(n // 5), 5)
    y = np.tile([0, 1], n // 2)
    tr, val, te = group_tvt_split(groups, y, ProbeConfig())
    gtr, gval, gte = set(groups[tr]), set(groups[val]), set(groups[te])
    assert gtr.isdisjoint(gval)
    assert gtr.isdisjoint(gte)
    assert gval.isdisjoint(gte)
    assert len(tr) + len(val) + len(te) == n


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all Component A unit tests passed")
