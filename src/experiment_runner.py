"""
Experiment Runner: Executes comparative evaluation across verified AFlow GSM8K pairs.
Tests Uniform, Parent-Guided, Child-Pilot, and Oracle policies across disjoint seed splits.
Generates RESULTS_TABLE.csv and RESULTS_SUMMARY.json.
"""

import os
import json
import itertools
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

from src.data_loader import (
    load_workflow_scores,
    get_verified_parent_child_pairs,
    get_stratified_30_question_indices,
    get_all_disjoint_seed_splits,
    DEFAULT_AFLOW_PATH
)
from src.policies import (
    uniform_allocation,
    parent_guided_allocation,
    child_pilot_allocation,
    oracle_neyman_allocation,
    compute_policy_estimate
)
from src.metrics import (
    compute_variance_correlation,
    compute_aggregate_metrics
)


# Key verified pairs representing distinct mutation categories
SELECTED_PAIRS = [
    {
        "parent_id": 2,
        "child_id": 3,
        "label": "P2 -> C3 (Verifier Insertion: Programmer)",
        "edit_type": "verifier_insertion",
        "delta": 0.5
    },
    {
        "parent_id": 3,
        "child_id": 8,
        "label": "P3 -> C8 (Prompt Refinement: Format Rules)",
        "edit_type": "prompt_refinement",
        "delta": 0.2
    },
    {
        "parent_id": 8,
        "child_id": 10,
        "label": "P8 -> C10 (Ensemble Widening: 3 -> 5 solutions)",
        "edit_type": "ensemble_widening",
        "delta": 0.35
    },
    {
        "parent_id": 10,
        "child_id": 13,
        "label": "P10 -> C13 (Verifier Refinement: Post-Ensemble Check)",
        "edit_type": "verifier_insertion",
        "delta": 0.4
    },
    {
        "parent_id": 4,
        "child_id": 5,
        "label": "P4 -> C5 (Mutation Shock / Degraded Failure)",
        "edit_type": "mutation_shock",
        "delta": 0.9
    }
]

BUDGET_RATIOS = [1.25, 1.5, 2.0, 2.5]


def run_experiment_on_subset(
    question_indices: np.ndarray,
    subset_name: str = "stratified_30",
    graphs_path: str = DEFAULT_AFLOW_PATH
) -> Tuple[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run evaluation across selected parent-child pairs and all 10 disjoint seed splits.
    """
    N = len(question_indices)
    seed_splits = get_all_disjoint_seed_splits()  # 10 splits of (3 train, 2 test)
    records = []
    pair_variance_stats = []
    
    for pair in SELECTED_PAIRS:
        p_id = pair["parent_id"]
        c_id = pair["child_id"]
        label = pair["label"]
        delta = pair["delta"]
        
        # Load scores
        p_scores_all, questions, p_costs = load_workflow_scores(p_id, graphs_path)
        c_scores_all, _, c_costs = load_workflow_scores(c_id, graphs_path)
        
        # Subset to target questions
        p_scores = p_scores_all[:, question_indices]
        c_scores = c_scores_all[:, question_indices]
        subset_questions = [questions[idx] for idx in question_indices]
        
        # Compute parent-child variance correlation across full 5 seeds
        v_stats = compute_variance_correlation(p_scores, c_scores)
        v_stats.update({
            "pair_label": label,
            "parent_id": p_id,
            "child_id": c_id,
            "edit_type": pair["edit_type"],
            "parent_mean_score": float(np.mean(p_scores)),
            "child_mean_score": float(np.mean(c_scores))
        })
        pair_variance_stats.append(v_stats)
        
        for ratio in BUDGET_RATIOS:
            budget = int(round(ratio * N))
            
            # Policy storage across splits
            estimates = {
                "Uniform": [],
                "Parent-Guided (P-PGA)": [],
                "Parent-Guided (Trust-Calibrated)": [],
                "Child-Pilot (k=1)": [],
                "Child-Pilot (Debiased)": [],
                "Oracle (Neyman Hindsight)": []
            }
            references = []
            
            for split_idx, (train_seeds, test_seeds) in enumerate(seed_splits):
                # Disjoint split
                c_train_pool = c_scores[list(train_seeds), :]  # (3, N)
                c_test_pool = c_scores[list(test_seeds), :]    # (2, N)
                
                # Approximate held-out reference: average of the 2 held-out seeds
                mu_ref = float(np.mean(c_test_pool))
                references.append(mu_ref)
                
                # 1. Uniform Policy
                n_uniform = uniform_allocation(N, budget, seed=100 + split_idx, max_runs=3)
                est_uniform = compute_policy_estimate(c_train_pool, n_uniform)
                estimates["Uniform"].append(est_uniform)
                
                # 2. Parent-Guided Allocation (P-PGA, full trust alpha=1.0)
                n_parent = parent_guided_allocation(p_scores, budget, trust_factor=1.0, max_runs=3)
                est_parent = compute_policy_estimate(c_train_pool, n_parent)
                estimates["Parent-Guided (P-PGA)"].append(est_parent)
                
                # 3. Parent-Guided with trust factor based on delta (alpha = exp(-2 * delta))
                trust = float(np.exp(-2.0 * delta))
                n_parent_trust = parent_guided_allocation(p_scores, budget, trust_factor=trust, max_runs=3)
                est_parent_trust = compute_policy_estimate(c_train_pool, n_parent_trust)
                estimates["Parent-Guided (Trust-Calibrated)"].append(est_parent_trust)
                
                # 4. Child-Pilot Allocation (k=1 pilot)
                n_pilot, is_feas = child_pilot_allocation(c_train_pool, budget, pilot_k=1, max_runs=3)
                est_pilot = compute_policy_estimate(c_train_pool, n_pilot)
                estimates["Child-Pilot (k=1)"].append(est_pilot)
                
                # 5. Child-Pilot Debiased
                pilot_p = (c_train_pool[:1, :].sum(axis=0) + 1.0) / 3.0
                pilot_std = np.sqrt(pilot_p * (1.0 - pilot_p))
                est_pilot_deb = compute_policy_estimate(c_train_pool, n_pilot, debiased=True, predicted_std=pilot_std)
                estimates["Child-Pilot (Debiased)"].append(est_pilot_deb)
                
                # 6. Oracle Allocation (hindsight using all 5 child seeds)
                n_oracle = oracle_neyman_allocation(c_scores, budget, max_runs=3)
                est_oracle = compute_policy_estimate(c_train_pool, n_oracle)
                estimates["Oracle (Neyman Hindsight)"].append(est_oracle)
                
            # Aggregate across splits
            for pol_name, est_list in estimates.items():
                agg = compute_aggregate_metrics(est_list, references)
                records.append({
                    "subset": subset_name,
                    "pair": label,
                    "parent_id": p_id,
                    "child_id": c_id,
                    "edit_type": pair["edit_type"],
                    "budget_ratio": ratio,
                    "budget_rollouts": budget,
                    "policy": pol_name,
                    "rmse": agg["rmse"],
                    "mae": agg["mae"],
                    "bias": agg["bias"],
                    "variance": agg["variance"]
                })
                
    df_results = pd.DataFrame(records)
    return df_results, pair_variance_stats, subset_questions


def extract_question_level_example(
    pair_info: Dict[str, Any],
    question_indices: np.ndarray,
    budget_ratio: float = 1.5,
    graphs_path: str = DEFAULT_AFLOW_PATH
) -> pd.DataFrame:
    """
    Extract a detailed question-by-question breakdown showing:
      - Question index
      - Question text snippet
      - Parent pass rate (5 seeds)
      - Parent Laplace-regularized std dev
      - Child observed pass rate (5 seeds)
      - Child observed variance (5 seeds)
      - Uniform allocation count
      - Parent-Guided allocation count
    """
    p_id = pair_info["parent_id"]
    c_id = pair_info["child_id"]
    
    p_scores_all, questions, _ = load_workflow_scores(p_id, graphs_path)
    c_scores_all, _, _ = load_workflow_scores(c_id, graphs_path)
    
    p_scores = p_scores_all[:, question_indices]
    c_scores = c_scores_all[:, question_indices]
    
    N = len(question_indices)
    budget = int(round(budget_ratio * N))
    
    p_pass = p_scores.mean(axis=0)
    p_reg = (p_scores.sum(axis=0) + 1.0) / 7.0
    p_std_reg = np.sqrt(p_reg * (1.0 - p_reg))
    
    c_pass = c_scores.mean(axis=0)
    c_var = c_scores.var(axis=0, ddof=1)
    
    n_uni = uniform_allocation(N, budget, seed=42, max_runs=3)
    n_parent = parent_guided_allocation(p_scores, budget, trust_factor=1.0, max_runs=3)
    
    rows = []
    for i, orig_idx in enumerate(question_indices):
        q_text = questions[orig_idx]
        snippet = q_text[:75] + "..." if len(q_text) > 75 else q_text
        rows.append({
            "question_index": int(orig_idx),
            "question_snippet": snippet,
            "parent_pass_rate": round(float(p_pass[i]), 2),
            "parent_reg_std": round(float(p_std_reg[i]), 3),
            "child_pass_rate": round(float(c_pass[i]), 2),
            "child_variance": round(float(c_var[i]), 3),
            "uniform_alloc": int(n_uni[i]),
            "parent_guided_alloc": int(n_parent[i]),
            "alloc_diff": int(n_parent[i] - n_uni[i])
        })
    return pd.DataFrame(rows)


def run_full_suite(output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute full evaluation suite:
      1. Stratified 30-question cohort
      2. Full 100-question GSM8K pool
      3. Saves RESULTS_TABLE.csv, RESULTS_SUMMARY.json, and EXAMPLE_ALLOCATION.csv
    """
    target_dir = output_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Executing Experiment Runner (output dir: {target_dir})...")
    
    # 1. Stratified 30-question subset
    idx_30 = get_stratified_30_question_indices()
    df_30, v_stats_30, qs_30 = run_experiment_on_subset(idx_30, subset_name="stratified_30")
    
    # 2. Full 100 questions
    idx_100 = np.arange(100)
    df_100, v_stats_100, qs_100 = run_experiment_on_subset(idx_100, subset_name="full_100")
    
    df_all = pd.concat([df_30, df_100], ignore_index=True)
    
    # Save CSV
    csv_path = os.path.join(target_dir, "RESULTS_TABLE.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"Saved results table to {csv_path}")
    
    # Question-level example on Pair 2 (P3 -> C8, prompt refinement) at B = 1.5N
    example_pair = SELECTED_PAIRS[1]  # P3 -> C8
    df_example = extract_question_level_example(example_pair, idx_30, budget_ratio=1.5)
    example_csv_path = os.path.join(target_dir, "EXAMPLE_ALLOCATION_TABLE.csv")
    df_example.to_csv(example_csv_path, index=False)
    print(f"Saved example allocation table to {example_csv_path}")
    
    # Summary JSON
    summary_data = {
        "metadata": {
            "dataset": "GSM8K Validation Subset",
            "evaluated_pairs": len(SELECTED_PAIRS),
            "budget_ratios": BUDGET_RATIOS,
            "disjoint_seed_splits": 10,
            "held_out_reference": "2 disjoint held-out runs (approximate reference)",
            "evaluation_pool": "3 disjoint available runs"
        },
        "variance_correlations": v_stats_30,
        "results_summary": []
    }
    
    # Aggregate across pairs for high-level summary
    for subset in ["stratified_30", "full_100"]:
        sub_df = df_all[df_all["subset"] == subset]
        for b in BUDGET_RATIOS:
            b_df = sub_df[sub_df["budget_ratio"] == b]
            summary_entry = {
                "subset": subset,
                "budget_ratio": b,
                "policies": {}
            }
            for pol in b_df["policy"].unique():
                pol_sub = b_df[b_df["policy"] == pol]
                summary_entry["policies"][pol] = {
                    "mean_rmse": float(pol_sub["rmse"].mean()),
                    "mean_mae": float(pol_sub["mae"].mean()),
                    "mean_bias": float(pol_sub["bias"].mean())
                }
            summary_data["results_summary"].append(summary_entry)
            
    json_path = os.path.join(target_dir, "RESULTS_SUMMARY.json")
    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved results summary to {json_path}")
    
    return summary_data


if __name__ == "__main__":
    run_full_suite()
