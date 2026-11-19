"""
Plot generator for Senior Design presentation figures.
Produces:
  - charts/parent_child_variance.png
  - charts/rmse_comparison.png
  - charts/example_allocation.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import (
    load_workflow_scores,
    get_stratified_30_question_indices,
    DEFAULT_AFLOW_PATH
)
from src.experiment_runner import (
    SELECTED_PAIRS,
    extract_question_level_example
)


def plot_parent_child_variance(output_path: str = "charts/parent_child_variance.png"):
    """
    Scatter plot of question-level parent variance vs child variance across verified pairs.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=200)
    
    # 3 illustrative pairs:
    # 1. P3 -> C8 (Prompt Refinement, r=+0.561)
    # 2. P8 -> C11 (Prompt + Verifier Failure, r=+0.659)
    # 3. P4 -> C5 (Mutation Shock, r=+0.131)
    pairs_to_plot = [
        (3, 8, "P3 -> C8 (Prompt Refinement)\n$r = +0.561$, Jaccard $= 0.417$"),
        (8, 11, "P8 -> C11 (Prompt Mod + Verifier)\n$r = +0.659$, Jaccard $= 0.500$"),
        (4, 5, "P4 -> C5 (Mutation Shock / Degradation)\n$r = +0.131$, Jaccard $= 0.125$")
    ]
    
    for ax, (p_id, c_id, title) in zip(axes, pairs_to_plot):
        p_scores, _, _ = load_workflow_scores(p_id)
        c_scores, _, _ = load_workflow_scores(c_id)
        
        p_var = p_scores.var(axis=0, ddof=1)
        c_var = c_scores.var(axis=0, ddof=1)
        
        # Add slight jitter for visualization since Bernoulli variances are discrete
        jitter_p = np.random.RandomState(42).normal(0, 0.005, size=len(p_var))
        jitter_c = np.random.RandomState(43).normal(0, 0.005, size=len(c_var))
        
        # Identify deterministic items
        both_det = (p_var == 0) & (c_var == 0)
        p_only_det = (p_var == 0) & (c_var > 0)
        c_only_det = (p_var > 0) & (c_var == 0)
        both_var = (p_var > 0) & (c_var > 0)
        
        ax.scatter(p_var[both_det] + jitter_p[both_det], c_var[both_det] + jitter_c[both_det], 
                   color="#4A90E2", alpha=0.6, s=40, label=f"Both Stable (0,0): {np.sum(both_det)}")
        ax.scatter(p_var[both_var] + jitter_p[both_var], c_var[both_var] + jitter_c[both_var], 
                   color="#E74C3C", alpha=0.9, s=60, marker="o", label=f"Both Unstable: {np.sum(both_var)}")
        ax.scatter(p_var[p_only_det | c_only_det] + jitter_p[p_only_det | c_only_det], 
                   c_var[p_only_det | c_only_det] + jitter_c[p_only_det | c_only_det], 
                   color="#F39C12", alpha=0.7, s=40, marker="^", label=f"Mixed: {np.sum(p_only_det | c_only_det)}")
        
        ax.plot([0, 0.3], [0, 0.3], color="gray", linestyle="--", alpha=0.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Parent Question Variance ($\sigma_P^2$)", fontsize=10)
        ax.set_ylabel("Child Question Variance ($\sigma_C^2$)", fontsize=10)
        ax.set_xlim(-0.02, 0.30)
        ax.set_ylim(-0.02, 0.30)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", fontsize=8)
        
    plt.suptitle("GSM8K Parent vs. Child Question-Level Variance Transfer (AFlow Historical Records)", 
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved variance plot to {output_path}")


def plot_rmse_comparison(
    results_csv: str = "RESULTS_TABLE.csv",
    output_path: str = "charts/rmse_comparison.png"
):
    """
    Plot estimation RMSE vs evaluation budget across policies for both 30-item and 100-item sets.
    """
    df = pd.read_csv(results_csv)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=200)
    
    policies_to_plot = [
        ("Uniform", "#7F8C8D", "o-", "Uniform Allocation (Baseline A)"),
        ("Parent-Guided (P-PGA)", "#2980B9", "s-", "Parent-Guided P-PGA (Our Method B)"),
        ("Parent-Guided (Trust-Calibrated)", "#16A085", "^--", "Parent-Guided (Trust-Calibrated D)"),
        ("Child-Pilot (k=1)", "#E67E22", "d-.", "Child-Pilot k=1 (Speculative Baseline C)"),
        ("Oracle (Neyman Hindsight)", "#27AE60", "*:", "Oracle Neyman (Hindsight Bound E)")
    ]
    
    for ax, subset_name, title in [
        (ax1, "stratified_30", "Stratified Presentation Cohort ($N=30$)"),
        (ax2, "full_100", "Full GSM8K Benchmark Pool ($N=100$)")
    ]:
        sub = df[df["subset"] == subset_name]
        budgets = sorted(sub["budget_ratio"].unique())
        
        for pol, color, style, label in policies_to_plot:
            pol_df = sub[sub["policy"] == pol]
            means = [pol_df[pol_df["budget_ratio"] == b]["rmse"].mean() for b in budgets]
            ax.plot(budgets, means, style, color=color, linewidth=2.0, markersize=7, label=label)
            
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Child Evaluation Budget ($B / N$ Rollouts per Question)", fontsize=11)
        ax.set_ylabel("Estimation RMSE vs. Held-Out Reference", fontsize=11)
        ax.set_xticks(budgets)
        ax.set_xticklabels([f"{b:.2f}N\n({int(round(b*(30 if '30' in subset_name else 100)))} runs)" for b in budgets])
        ax.grid(True, linestyle=":", alpha=0.6)
        
    ax1.legend(loc="upper right", fontsize=8.5)
    ax2.legend(loc="upper right", fontsize=8.5)
    
    plt.suptitle("Child Workflow Pass Rate Estimation Error at Matched Evaluation Budgets\n"
                 "(Average over 5 Verified Workflow Pairs × 10 Disjoint Seed Splits; Ref: 2 Held-Out Runs)", 
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved RMSE comparison plot to {output_path}")


def plot_example_allocation(output_path: str = "charts/example_allocation.png"):
    """
    Visual question-level bar chart showing parent variance and resulting replicate allocation.
    """
    idx_30 = get_stratified_30_question_indices()
    example_pair = SELECTED_PAIRS[1]  # P3 -> C8 (Prompt refinement)
    df_ex = extract_question_level_example(example_pair, idx_30, budget_ratio=1.5)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7.5), dpi=200, sharex=True)
    
    x = np.arange(len(df_ex))
    width = 0.38
    
    # Subplot 1: Parent Regularized Std Dev vs Child Observed Variance
    ax1.bar(x - width/2, df_ex["parent_reg_std"], width, label="Parent Regularized Std ($\widetilde{\sigma}_{P,i}$)", color="#3498DB", alpha=0.85)
    ax1.bar(x + width/2, np.sqrt(df_ex["child_variance"]), width, label="Child Observed Std ($\sigma_{C,i}$)", color="#E74C3C", alpha=0.85)
    ax1.set_ylabel("Standard Deviation", fontsize=11)
    ax1.set_title("Question-Level Variance Profile: Parent 3 vs. Child 8 (GSM8K, 30 Items)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=9.5)
    
    # Subplot 2: Allocation Comparison (Uniform vs Parent-Guided)
    ax2.bar(x - width/2, df_ex["uniform_alloc"], width, label="Uniform Allocation ($B=45$ runs, 1.5N)", color="#95A5A6", alpha=0.85)
    ax2.bar(x + width/2, df_ex["parent_guided_alloc"], width, label="Parent-Guided P-PGA ($B=45$ runs, 1.5N)", color="#2ECC71", alpha=0.9)
    ax2.set_ylabel("Allocated Replicates ($n_i$)", fontsize=11)
    ax2.set_xlabel("Question Index in Stratified Presentation Cohort (0 to 29)", fontsize=11)
    ax2.set_title("Replicate Allocation at Matched Budget $B = 1.5N$ (45 Runs Total)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Q{i}" for i in range(len(df_ex))], fontsize=8, rotation=45)
    ax2.set_yticks([1, 2, 3])
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=9.5)
    
    # Highlight items receiving 2 runs
    for i in range(len(df_ex)):
        diff = df_ex["parent_guided_alloc"].iloc[i] - df_ex["uniform_alloc"].iloc[i]
        if diff > 0:
            ax2.annotate("+1", (x[i] + width/2, df_ex["parent_guided_alloc"].iloc[i] + 0.05),
                         ha="center", fontsize=8, fontweight="bold", color="#27AE60")
                         
    plt.suptitle("How Parent-Guided Allocation Concentrates Replicates on Unstable Questions\n"
                 "(Stable items get 1 run; high-variance boundary items get 2 runs)", 
                 fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved example allocation plot to {output_path}")


if __name__ == "__main__":
    os.makedirs("charts", exist_ok=True)
    plot_parent_child_variance()
    plot_rmse_comparison()
    plot_example_allocation()
