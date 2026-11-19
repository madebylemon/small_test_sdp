"""
Unit tests for metrics and error computations.
"""

import numpy as np
import pytest
from src.metrics import (
    compute_variance_correlation,
    compute_aggregate_metrics
)


def test_variance_correlation():
    # 5 runs, 3 questions
    p = np.array([
        [1, 0, 1],
        [1, 0, 0],
        [1, 1, 1],
        [1, 0, 0],
        [1, 0, 1]
    ], dtype=float)
    
    c = np.array([
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 0],
        [1, 0, 0],
        [1, 0, 1]
    ], dtype=float)
    
    stats = compute_variance_correlation(p, c)
    assert "pearson_r" in stats
    assert "jaccard_unstable" in stats
    assert "deterministic_stability" in stats
    assert -1.0 <= stats["pearson_r"] <= 1.0
    assert 0.0 <= stats["jaccard_unstable"] <= 1.0


def test_aggregate_metrics():
    est = [0.8, 0.85, 0.9]
    ref = [0.82, 0.82, 0.88]
    agg = compute_aggregate_metrics(est, ref)
    assert "rmse" in agg
    assert "mae" in agg
    assert "bias" in agg
    assert agg["rmse"] >= 0.0
    assert agg["mae"] >= 0.0
