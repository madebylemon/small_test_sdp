"""
Allocation policies for multi-agent workflow evaluation.
Implements:
  - Uniform Allocation (Baseline A)
  - Precommitted Parent-Guided Neyman Allocation (P-PGA, Policy B)
  - Child-Pilot Allocation (Speculative Evaluation baseline, Policy C)
  - Edit-Aware Calibrated Parent Allocation (Policy D)
  - Hindsight Oracle Neyman Allocation (Reference E)
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional


def bounded_integer_neyman_allocation(
    predicted_std: np.ndarray,
    budget: int,
    base_runs: int = 1,
    max_runs: int = 3
) -> np.ndarray:
    """
    Greedy discrete integer Neyman allocation.
    Minimizes sum(sigma_i^2 / n_i) subject to:
      - sum(n_i) == budget (or all reach max_runs)
      - base_runs <= n_i <= max_runs
    
    Marginal variance reduction when incrementing n_i -> n_i + 1:
      Delta V_i = sigma_i^2 / n_i - sigma_i^2 / (n_i + 1) = sigma_i^2 / (n_i * (n_i + 1))
    """
    N = len(predicted_std)
    n = np.full(N, base_runs, dtype=int)
    current_budget = N * base_runs
    
    if budget <= current_budget:
        return n
    
    variances = predicted_std ** 2
    remaining = budget - current_budget
    
    while remaining > 0:
        # Eligible items that haven't reached max_runs
        eligible = np.where(n < max_runs)[0]
        if len(eligible) == 0:
            break
            
        # Marginal variance gains for eligible items
        marginal_gains = variances[eligible] / (n[eligible] * (n[eligible] + 1).astype(float))
        best_local_idx = np.argmax(marginal_gains)
        best_item = eligible[best_local_idx]
        
        n[best_item] += 1
        remaining -= 1
        
    return n


def uniform_allocation(
    N: int,
    budget: int,
    seed: int = 42,
    base_min: int = 1,
    max_runs: int = 3
) -> np.ndarray:
    """
    Uniform baseline: distributes runs as evenly as possible.
    If budget / N is not an integer, remainder runs are assigned randomly
    using a reproducible prespecified seed.
    """
    n = np.full(N, base_min, dtype=int)
    remaining = budget - N * base_min
    if remaining <= 0:
        return n
        
    # Integer division
    extra_uniform = remaining // N
    if extra_uniform > 0:
        n += extra_uniform
        n = np.clip(n, base_min, max_runs)
        remaining = budget - int(n.sum())
        
    if remaining > 0:
        rng = np.random.RandomState(seed)
        eligible = np.where(n < max_runs)[0]
        if len(eligible) > 0:
            k = min(remaining, len(eligible))
            chosen = rng.choice(eligible, size=k, replace=False)
            n[chosen] += 1
            
    return n


def parent_guided_allocation(
    parent_scores: np.ndarray,
    budget: int,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
    trust_factor: float = 1.0,
    max_runs: int = 3
) -> np.ndarray:
    """
    Parent-Guided Allocation (P-PGA).
    Uses the parent workflow's repeated runs to estimate question variance.
    Applies Beta(alpha, beta) Laplace smoothing to prevent zero-variance collapse.
    Precommitted: fixed BEFORE observing child outcomes.
    
    trust_factor alpha in [0, 1] scales reliance on parent vs uninformative prior:
      p_tilde = trust * p_parent + (1 - trust) * 0.5
    """
    # parent_scores is shape (S_P, N)
    S_P = parent_scores.shape[0]
    N = parent_scores.shape[1]
    
    # Observed successes on parent
    successes = parent_scores.sum(axis=0)
    
    # Regularized pass rate (Laplace / Beta posterior mean)
    p_parent_reg = (successes + alpha_prior) / (S_P + alpha_prior + beta_prior)
    
    # Apply trust factor
    p_eff = trust_factor * p_parent_reg + (1.0 - trust_factor) * 0.5
    
    # Predicted standard deviation
    pred_std = np.sqrt(p_eff * (1.0 - p_eff))
    
    # Bounded integer Neyman allocation
    return bounded_integer_neyman_allocation(pred_std, budget, base_runs=1, max_runs=max_runs)


def child_pilot_allocation(
    child_eval_pool: np.ndarray,
    budget: int,
    pilot_k: int = 1,
    max_runs: int = 3,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0
) -> Tuple[np.ndarray, bool]:
    """
    Child-Pilot Allocation (inspired by Speculative Evaluation).
    Runs a fresh pilot of pilot_k runs on the current child, then allocates
    the remaining budget via Neyman allocation.
    
    Returns:
      allocations: np.ndarray of shape (N,)
      is_feasible: bool indicating whether pilot_k * N <= budget
    """
    N = child_eval_pool.shape[1]
    pilot_cost = pilot_k * N
    
    if budget < pilot_cost:
        # Pilot exceeds total budget! Fall back to uniform allocation within budget
        return uniform_allocation(N, budget, seed=999, base_min=1, max_runs=max_runs), False
        
    pilot_scores = child_eval_pool[:pilot_k, :]  # (pilot_k, N)
    pilot_successes = pilot_scores.sum(axis=0)
    
    # Regularized pilot pass rate
    p_pilot_reg = (pilot_successes + alpha_prior) / (pilot_k + alpha_prior + beta_prior)
    pred_std = np.sqrt(p_pilot_reg * (1.0 - p_pilot_reg))
    
    # Base allocation is pilot_k runs
    alloc = bounded_integer_neyman_allocation(pred_std, budget, base_runs=pilot_k, max_runs=max_runs)
    return alloc, True


def oracle_neyman_allocation(
    child_all_scores: np.ndarray,
    budget: int,
    max_runs: int = 3,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0
) -> np.ndarray:
    """
    Hindsight-only Oracle: uses empirical variance of the child across ALL observed runs.
    Upper bound benchmark, not deployable in practice.
    """
    S, N = child_all_scores.shape
    successes = child_all_scores.sum(axis=0)
    p_reg = (successes + alpha_prior) / (S + alpha_prior + beta_prior)
    std_true = np.sqrt(p_reg * (1.0 - p_reg))
    return bounded_integer_neyman_allocation(std_true, budget, base_runs=1, max_runs=max_runs)


def compute_policy_estimate(
    child_eval_pool: np.ndarray,
    allocations: np.ndarray,
    debiased: bool = False,
    predicted_std: Optional[np.ndarray] = None
) -> float:
    """
    Compute estimated workflow pass rate under the given question allocations.
    For question i, draws allocations[i] runs from child_eval_pool[:, i].
    
    child_eval_pool has shape (S_eval, N).
    """
    N = len(allocations)
    item_means = np.zeros(N)
    
    for i in range(N):
        k = allocations[i]
        # Use the first k available evaluation runs for item i
        item_means[i] = np.mean(child_eval_pool[:k, i])
        
    raw_estimate = float(np.mean(item_means))
    
    if debiased and predicted_std is not None:
        # Theorem 3 debiasing correction for adaptive sampling
        variances = predicted_std ** 2
        correction = np.mean((1.0 - 1.0 / allocations.astype(float)) * variances)
        return float(raw_estimate - correction)
        
    return raw_estimate
