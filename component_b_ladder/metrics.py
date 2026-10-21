"""Aggregation + cost accounting for a Component B run.

The headline metric is recovery (semantic / exact), gated on generation-verified
ASR — never the probe score alone (COMPONENT_B_PLAN.md 5). Search-space reduction
and queries-to-success belong here too; the exact brute-force reference ranking
is a Phase-1 follow-up (``search/baselines.py``).
"""

from __future__ import annotations


def summarise_seed(search_result, verify_result) -> dict:
    """One seed's outcome: best candidate, its probe score, recovery and ASR."""
    return {
        "candidate": verify_result["candidate"],
        "candidate_ids": search_result.best_ids,
        "probe_score": search_result.best_score,
        "semantic_recovery": verify_result["semantic_recovery"],
        "exact_recovery": verify_result["exact_recovery"],
        "asr": verify_result["asr"],
        "n_forward": search_result.n_forward,
        "n_backward": search_result.n_backward,
        "samples": verify_result["samples"],
    }


def summarise_run(seed_summaries: list[dict], asr_threshold: float = 0.8) -> dict:
    """Aggregate over seeds: recovery/verified rates and best ASR."""
    n = len(seed_summaries)
    return {
        "n_seeds": n,
        "semantic_recovery_rate": sum(s["semantic_recovery"] for s in seed_summaries) / n,
        "exact_recovery_rate": sum(s["exact_recovery"] for s in seed_summaries) / n,
        "verified_rate": sum(s["asr"] >= asr_threshold for s in seed_summaries) / n,
        "best_asr": max(s["asr"] for s in seed_summaries),
        "mean_probe_score": sum(s["probe_score"] for s in seed_summaries) / n,
        "asr_threshold": asr_threshold,
    }
