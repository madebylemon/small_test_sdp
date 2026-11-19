"""
Explicitly Labeled Synthetic Controlled Study.
Compares Uniform, Parent-Guided, Child-Pilot, and Oracle Neyman allocation
under known, ground-truth latent pass probabilities p_i^*.
Tests two distinct regimes:
  1. Strong Variance Transfer (Conservative mutation)
  2. Mutation Shock / Decoupled Variance (Radical graph mutation)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

from src.policies import (
    uniform_allocation,
    parent_guided_allocation,
    child_pilot_allocation,
    oracle_neyman_allocation,
    compute_policy_estimate
)


def generate_synthetic_data(
    N: int = 50,
    regime: str = "strong_transfer",
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate parent and child true latent probabilities p^*, and sample evaluations.
    Returns:
      p_parent_true: shape (N,)
      p_child_true: shape (N,)
      parent_draws: shape (5, N) [simulating parent 5-seed history]
      child_draws: shape (20, N) [simulating child rollout pool]
    """
    rng = np.random.RandomState(seed)
    
    # 70% items are either easy (p > 0.85) or hard (p < 0.15)
    # 30% items are unstable boundary items (0.35 <= p <= 0.65)
    p_parent_true = np.zeros(N)
    for i in range(N):
        u = rng.rand()
        if u < 0.60:
            p_parent_true[i] = rng.uniform(0.85, 0.98)
        elif u < 0.70:
            p_parent_true[i] = rng.uniform(0.02, 0.15)
        else:
            p_parent_true[i] = rng.uniform(0.35, 0.65)
            
    if regime == "strong_transfer":
        # Child is closely related: small perturbation
        delta = rng.normal(0.0, 0.05, size=N)
        p_child_true = np.clip(p_parent_true + delta, 0.01, 0.99)
    elif regime == "mutation_shock":
        # Radical mutation: 40% of items flip difficulty or change dramatically
        p_child_true = p_parent_true.copy()
        flip_idx = rng.choice(N, size=int(0.40 * N), replace=False)
        for idx in flip_idx:
            p_child_true[idx] = 1.0 - p_parent_true[idx] + rng.normal(0.0, 0.05)
        p_child_true = np.clip(p_child_true, 0.01, 0.99)
    else:
        raise ValueError(f"Unknown regime {regime}")
        
    # Draw Bernoulli samples
    parent_draws = (rng.rand(5, N) < p_parent_true).astype(float)
    child_draws = (rng.rand(20, N) < p_child_true).astype(float)
    
    return p_parent_true, p_child_true, parent_draws, child_draws


def run_synthetic_benchmark(
    n_trials: int = 100,
    N: int = 50,
    seed: int = 42
) -> pd.DataFrame:
    """
    Run synthetic experiment comparing policies against known true expected score mu^* = mean(p_child_true).
    """
    results = []
    regimes = ["strong_transfer", "mutation_shock"]
    budget_ratios = [1.25, 1.5, 2.0, 2.5]
    
    for regime in regimes:
        for ratio in budget_ratios:
            budget = int(round(ratio * N))
            
            trial_errors = {
                "Uniform": [],
                "Parent-Guided (P-PGA)": [],
                "Parent-Guided (Trust-Shrunk)": [],
                "Child-Pilot (k=1)": [],
                "Child-Pilot (Debiased)": [],
                "Oracle (Neyman)": []
            }
            
            for trial in range(n_trials):
                trial_seed = seed + trial * 13
                p_p_true, p_c_true, p_draws, c_draws = generate_synthetic_data(N, regime, trial_seed)
                mu_true = float(np.mean(p_c_true))
                
                # Available child pool (3 draws per question)
                c_pool = c_draws[:3, :]
                
                # 1. Uniform
                n_uni = uniform_allocation(N, budget, seed=trial_seed, max_runs=3)
                est_uni = compute_policy_estimate(c_pool, n_uni)
                trial_errors["Uniform"].append(est_uni - mu_true)
                
                # 2. Parent-Guided P-PGA (alpha=1.0)
                n_parent = parent_guided_allocation(p_draws, budget, trust_factor=1.0, max_runs=3)
                est_parent = compute_policy_estimate(c_pool, n_parent)
                trial_errors["Parent-Guided (P-PGA)"].append(est_parent - mu_true)
                
                # 3. Parent-Guided Trust-Shrunk (alpha=0.4 in shock, 0.8 in transfer)
                trust = 0.8 if regime == "strong_transfer" else 0.3
                n_trust = parent_guided_allocation(p_draws, budget, trust_factor=trust, max_runs=3)
                est_trust = compute_policy_estimate(c_pool, n_trust)
                trial_errors["Parent-Guided (Trust-Shrunk)"].append(est_trust - mu_true)
                
                # 4. Child-Pilot k=1
                n_pilot, _ = child_pilot_allocation(c_pool, budget, pilot_k=1, max_runs=3)
                est_pilot = compute_policy_estimate(c_pool, n_pilot)
                trial_errors["Child-Pilot (k=1)"].append(est_pilot - mu_true)
                
                # 5. Child-Pilot Debiased
                pilot_p = (c_pool[:1, :].sum(axis=0) + 1.0) / 3.0
                pilot_std = np.sqrt(pilot_p * (1.0 - pilot_p))
                est_pilot_deb = compute_policy_estimate(c_pool, n_pilot, debiased=True, predicted_std=pilot_std)
                trial_errors["Child-Pilot (Debiased)"].append(est_pilot_deb - mu_true)
                
                # 6. Oracle Neyman (using p_c_true)
                std_true = np.sqrt(p_c_true * (1.0 - p_c_true))
                from src.policies import bounded_integer_neyman_allocation
                n_oracle = bounded_integer_neyman_allocation(std_true, budget, base_runs=1, max_runs=3)
                est_oracle = compute_policy_estimate(c_pool, n_oracle)
                trial_errors["Oracle (Neyman)"].append(est_oracle - mu_true)
                
            for pol, err_list in trial_errors.items():
                errs = np.array(err_list)
                rmse = float(np.sqrt(np.mean(errs ** 2)))
                mae = float(np.mean(np.abs(errs)))
                bias = float(np.mean(errs))
                results.append({
                    "regime": regime,
                    "budget_ratio": ratio,
                    "policy": pol,
                    "rmse": rmse,
                    "mae": mae,
                    "bias": bias
                })
                
    return pd.DataFrame(results)
