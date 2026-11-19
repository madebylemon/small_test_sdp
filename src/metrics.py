"""
Metrics and evaluation statistics for policy comparison.
Calculates RMSE, MAE, bias, variance correlation, Jaccard overlap, and cost.
"""

from typing import Dict, List, Tuple, Any
import numpy as np


def compute_variance_correlation(
    parent_scores: np.ndarray,
    child_scores: np.ndarray
) -> Dict[str, float]:
    """
    Compute question-level variance metrics between parent and child.
    parent_scores: shape (S_P, N)
    child_scores: shape (S_C, N)
    """
    p_var = parent_scores.var(axis=0, ddof=1)
    c_var = child_scores.var(axis=0, ddof=1)
    
    # Pearson correlation
    std_p = np.std(p_var)
    std_c = np.std(c_var)
    if std_p > 1e-9 and std_c > 1e-9:
        corr = float(np.corrcoef(p_var, c_var)[0, 1])
    else:
        corr = 0.0
        
    p_unstable = set(np.where(p_var > 0)[0])
    c_unstable = set(np.where(c_var > 0)[0])
    
    union = p_unstable | c_unstable
    intersection = p_unstable & c_unstable
    jaccard = float(len(intersection) / len(union)) if len(union) > 0 else 1.0
    
    # Deterministic stability: proportion of parent-deterministic items that remain child-deterministic
    p_det = set(np.where(p_var == 0)[0])
    c_det = set(np.where(c_var == 0)[0])
    det_stability = float(len(p_det & c_det) / len(p_det)) if len(p_det) > 0 else 1.0
    
    return {
        "pearson_r": corr,
        "jaccard_unstable": jaccard,
        "parent_unstable_count": len(p_unstable),
        "child_unstable_count": len(c_unstable),
        "deterministic_stability": det_stability
    }


def compute_aggregate_metrics(
    estimates: List[float],
    references: List[float]
) -> Dict[str, float]:
    """
    Compute RMSE, MAE, and Bias of estimates against held-out references across trials.
    """
    assert len(estimates) == len(references)
    errs = np.array(estimates) - np.array(references)
    
    rmse = float(np.sqrt(np.mean(errs ** 2)))
    mae = float(np.mean(np.abs(errs)))
    bias = float(np.mean(errs))
    var_err = float(np.var(errs))
    
    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "variance": var_err,
        "n_trials": len(estimates)
    }
