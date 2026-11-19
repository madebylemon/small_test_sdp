"""
Unit tests for allocation policies.
"""

import numpy as np
import pytest
from src.policies import (
    bounded_integer_neyman_allocation,
    uniform_allocation,
    parent_guided_allocation,
    child_pilot_allocation,
    oracle_neyman_allocation,
    compute_policy_estimate
)


def test_bounded_integer_neyman():
    std = np.array([0.1, 0.5, 0.4, 0.1, 0.5])
    N = len(std)
    budget = 8
    alloc = bounded_integer_neyman_allocation(std, budget=budget, base_runs=1, max_runs=3)
    
    assert len(alloc) == N
    assert alloc.sum() == budget
    assert np.all(alloc >= 1)
    assert np.all(alloc <= 3)
    # High-variance items should receive more runs
    assert alloc[1] >= alloc[0]
    assert alloc[4] >= alloc[3]


def test_uniform_allocation():
    N = 30
    budget = 45  # 1.5N
    alloc = uniform_allocation(N, budget=budget, seed=42, max_runs=3)
    
    assert len(alloc) == N
    assert alloc.sum() == budget
    assert np.all(alloc >= 1)
    assert np.all(alloc <= 2)
    assert np.sum(alloc == 2) == 15
    assert np.sum(alloc == 1) == 15


def test_parent_guided_allocation():
    # 5 runs, 4 questions
    # Q0: 5/5 passes, Q1: 0/5 fails, Q2: 2/5 passes, Q3: 3/5 passes
    parent_scores = np.array([
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 0, 1]
    ], dtype=float)
    
    budget = 6
    alloc = parent_guided_allocation(parent_scores, budget=budget, max_runs=3)
    assert alloc.sum() == budget
    assert np.all(alloc >= 1)
    # Q2 and Q3 have non-zero variance; they should be prioritized over deterministic Q0 and Q1
    assert alloc[2] >= alloc[0]
    assert alloc[3] >= alloc[1]


def test_child_pilot_allocation():
    child_pool = np.ones((3, 10))
    budget = 15
    alloc, feas = child_pilot_allocation(child_pool, budget=budget, pilot_k=1, max_runs=3)
    assert feas is True
    assert alloc.sum() == budget
    assert np.all(alloc >= 1)
    
    # Infeasible test: budget < pilot
    alloc_inf, feas_inf = child_pilot_allocation(child_pool, budget=5, pilot_k=1)
    assert feas_inf is False


def test_policy_estimate():
    pool = np.array([
        [1, 0, 1],
        [1, 1, 0],
        [0, 1, 1]
    ])
    alloc = np.array([2, 1, 3])
    est = compute_policy_estimate(pool, alloc)
    # Q0: runs 0, 1 -> mean(1, 1) = 1.0
    # Q1: run 0 -> 0.0
    # Q2: runs 0, 1, 2 -> mean(1, 0, 1) = 2/3
    # Expected overall: (1.0 + 0.0 + 2/3) / 3 = 5/9 = 0.5555...
    assert pytest.approx(est, rel=1e-4) == 5.0 / 9.0
