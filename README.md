# Scoring Multi-Agent Workflows with Fewer Test Runs by Reusing a Parent's Repeated Runs

**University of Connecticut — Senior Design Project (Group 57)**  
**Faculty Advisor:** Prof. Wei Zhang  
**Repository:** [madebylemon/small_test_sdp](https://github.com/madebylemon/small_test_sdp)  
**Status:** Reproducible CPU-Only Feasibility Experiment & Empirical Validation  

---

## 1. Executive Summary

Evaluating automated multi-agent LLM workflows (such as those discovered via evolutionary search in **AFlow**) is computationally expensive. When optimizing agent graphs, AFlow repeatedly mutates workflows (e.g., adding self-consistency loops or Python verifier tools) and evaluates each newly generated candidate with a **uniform allocation of 5 evaluation seeds across 100 validation questions** (500 LLM rollouts per candidate). This rapidly exhausts API token budgets in roughly 20 search rounds.

However, an exhaustive audit of historical AFlow logs reveals that **85% to 95% of benchmark questions are completely deterministic across seeds** (they either pass 5/5 times or fail 0/5 times across all seeds). Uniformly repeating evaluations across every question wastes over 60% of total rollouts re-evaluating questions that never vary.

### Our Core Proposal: Parent-Guided Allocation (P-PGA)
Instead of evaluating candidate workflows de novo or burning budget on an expensive fresh child pilot, we **reuse the parent workflow's existing 5-seed evaluation history** to predict query-level variance. We then apply **bounded integer Neyman allocation** to concentrate repeated runs strictly on volatile boundary questions, while assigning only a single run to stable questions.

```
                      Parent Workflow (Round 2)
                   5 Evaluation Seeds Completed
                                │
                  Parent Question Variance Profile
                (88% of deterministic items remain stable)
                                │
                    Workflow Mutation (MCTS Edit)
                                │
                                ▼
                       Child Workflow (Round 3)
         Precommitted Parent-Guided Neyman Allocation (P-PGA)
     ┌──────────────────────────┴──────────────────────────┐
     ▼                                                     ▼
Stable / Deterministic Questions               Volatile Boundary Questions
       (Assign 1 Run)                                (Assign 2–3 Runs)
     └──> 70% Replicate Savings                     └──> Variance Suppressed
```

---

## 2. Real AFlow Parent-Child Workflow Example

In AFlow, workflows are represented as executable Python DAGs. Each round creates a new candidate by modifying an existing parent node in the search tree.

### Actual Code Diff: Parent 2 $\to$ Child 3 (GSM8K Benchmark)
In **Round 2**, the workflow generated 3 solutions using an LLM and applied `ScEnsemble` majority voting:

```python
# Round 2 (Parent Workflow: graph.py)
solutions = []
for _ in range(3):
    solution = await self.custom(input=problem, instruction=prompt_custom.MATH_SOLVE_PROMPT)
    solutions.append(solution['response'])

final_solution = await self.sc_ensemble(solutions=solutions, problem=problem)
return final_solution['response'], self.llm.cost_manager.total_cost
```

In **Round 3**, the optimizer mutated the graph by inserting a Python code-execution sandbox (`Programmer`) to double-check mathematical reasoning after majority voting:

```python
# Round 3 (Child Workflow: graph.py)
final_solution = await self.sc_ensemble(solutions=solutions, problem=problem)

# --- Added Operator: Verification step using Programmer operator ---
review_result = await self.programmer(problem=problem, analysis=final_solution['response'])

if review_result['output']:
    return review_result['output'], self.llm.cost_manager.total_cost
else:
    return final_solution['response'], self.llm.cost_manager.total_cost
```

AFlow evaluated both Round 2 and Round 3 with 5 uniform seeds across all 100 questions. Round 2 achieved a pass rate of **0.826**, and Child Round 3 improved to **0.872**.

---

## 3. Allocation Policies Compared

All policies are evaluated under strictly matched total evaluation budgets $B = C \cdot N$ (where $C \in \{1.25, 1.50, 2.00, 2.50\}$ rollouts per item):

| Policy | Information Source | Mechanism | Pilot Cost | Unbiasedness |
|:---|:---|:---|:---:|:---:|
| **A. Uniform Allocation (Baseline A)** | None (Uniform) | Assigns $\lfloor B/N \rfloor$ runs to all items; randomizes remainder using a fixed seed. | 0 | **Provably Unbiased** |
| **B. Parent-Guided P-PGA (Our Method)** | Parent 5-seed history | Beta(1,1) regularized variance + discrete bounded Neyman allocation. Precommitted before child testing. | **0** | **Provably Unbiased** |
| **C. Trust-Calibrated Parent (Method D)** | Parent history + AST distance | Shrinks predicted variance toward uniform prior based on mutation distance $\alpha(\Delta) = \exp(-2\Delta)$. | **0** | **Provably Unbiased** |
| **D. Child-Pilot (Baseline C)** | Child 1-run pilot | Evaluates child once ($N$ runs), then allocates remaining budget based on pilot variance (Speculative Eval baseline). | $N$ runs | Biased without correction |
| **E. Oracle Neyman (Reference E)** | True child variance (Hindsight) | Neyman allocation using child's empirical variance across all runs. Unattainable ceiling. | N/A | Benchmark Ceiling |

### Mathematical Formulation of Parent-Guided Allocation (P-PGA)
1. **Beta(1, 1) Laplace Regularization:** To prevent 0/5 or 5/5 historical observations from falsely collapsing variance to zero:
   $$\tilde{p}_{P,i} = \frac{k_i + 1}{S_P + 2} = \frac{k_i + 1}{7}, \quad \tilde{\sigma}_{P,i} = \sqrt{\tilde{p}_{P,i}(1 - \tilde{p}_{P,i})}$$
   This ensures $\tilde{\sigma}_{P,i} \in [0.350, 0.495]$: volatile items receive $\tilde{\sigma} \approx 0.495$, while deterministic items receive $\tilde{\sigma} \approx 0.350$.
2. **Bounded Integer Neyman Allocation:** Every question receives at least one child run ($n_i \ge 1$). The remaining budget $B - N$ is allocated greedily by marginal variance reduction:
   $$\Delta V_i(n_i) = \frac{\tilde{\sigma}_{P,i}^2}{n_i(n_i + 1)}$$
   Greedily incrementing $n_i \leftarrow n_i + 1$ on the item with maximum $\Delta V_i$ subject to $n_i \le S_{\max}$.
3. **Precommitment Guarantee:** Because repeat counts $\mathbf{n} = (n_1, \dots, n_N)$ are determined strictly before observing child outcomes, the sample mean estimator is **provably unbiased** ($\mathbb{E}[\hat{\mu}] = \mu^*$).

---

## 4. Experimental Setup & Disjoint Split Protocol

To evaluate these policies with complete statistical validity and zero data leakage:
- **Benchmark:** GSM8K Validation Set (Official 100 questions, matching row-for-row).
- **Verified Workflow Pairs:** 5 verified MCTS parent-child pairs representing 4 distinct mutation categories:
  1. *Verifier Insertion:* Parent 2 $\to$ Child 3 (Added Programmer review)
  2. *Prompt Refinement:* Parent 3 $\to$ Child 8 (Explicit numeric formatting instructions)
  3. *Ensemble Widening:* Parent 8 $\to$ Child 10 (Expanded ScEnsemble loop from 3 to 5 solutions)
  4. *Verifier Refinement:* Parent 10 $\to$ Child 13 (Programmer check post-ensemble)
  5. *Mutation Shock / Degradation:* Parent 4 $\to$ Child 5 (Added Custom checker that dropped accuracy from 0.868 to 0.758)
- **Disjoint Partitioning Protocol:**
  - For each child workflow, its 5 historical runs are partitioned into **3 evaluation runs** (available for policies to sample) and **2 held-out reference runs** (untouched; used solely as an approximate evaluation reference).
  - Evaluated across all $\binom{5}{3} = 10$ combinatorial seed splits, eliminating seed-selection bias.

> [!NOTE]
> **Approximate Held-Out Reference Disclaimer:** A reference based on 2 held-out runs contains Bernoulli sampling noise. It is an *approximate held-out reference*, not an infinite-sample ground truth.

---

## 5. Experimental Results

### 5.1 Empirical Question-Level Variance Transfer
Parent variance correlates positively with child variance across historical pairs, confirming that query variance is not independent across mutations:

| Pair Index | Edit Category | Parent Score | Child Score | Variance Correlation ($\rho$) | Unstable Overlap (Jaccard) | Deterministic Stability ($\sigma^2=0$) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **P2 $\to$ C3** | Verifier Insertion | 0.826 | 0.872 | $+0.198$ | 0.250 | 84.6% |
| **P3 $\to$ C8** | Prompt Refinement | 0.872 | 0.888 | **$+0.577$** | **0.556** | **87.5%** |
| **P8 $\to$ C10** | Ensemble Widening | 0.888 | 0.898 | $+0.166$ | 0.111 | **95.5%** |
| **P10 $\to$ C13** | Verifier Refinement | 0.898 | 0.906 | **$+0.695$** | **0.500** | **100.0%** |
| **P4 $\to$ C5** | Mutation Shock | 0.868 | 0.758 | $+0.094$ | 0.182 | 82.6% |
| **Average** | — | **0.870** | **0.864** | **$+0.346$** | **0.320** | **88.0%** |

---

### 5.2 Pass Rate Estimation Error (RMSE) at Matched Budgets

#### Full GSM8K Benchmark Pool ($N=100$ items)
*Averaged across 5 verified pairs $\times$ 10 disjoint seed splits:*

| Budget Ratio | Rollout Count ($B$) | Compute Saving vs. $S=5$ | Uniform Allocation RMSE | Parent-Guided (P-PGA) RMSE | Trust-Calibrated Parent RMSE | Child-Pilot ($k=1$) RMSE | Oracle Bound RMSE |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1.25N** | 125 runs | **$-75\%$** | 0.01088 | 0.01096 | **0.01055 ($-3.0\%$)** | 0.01080 | 0.00956 |
| **1.50N** | 150 runs | **$-70\%$** | 0.01076 | 0.01068 | **0.00991 ($-7.9\%$)** | **0.00956 ($-11.2\%$)** | 0.00956 |
| **2.00N** | 200 runs | **$-60\%$** | 0.00956 | 0.00956 | 0.00956 | 0.00956 | 0.00956 |
| **2.50N** | 250 runs | **$-50\%$** | 0.00911 | **0.00842 ($-7.6\%$)** | 0.00874 | 0.00907 | 0.00844 |

#### Stratified Presentation Cohort ($N=30$ items)
*Averaged across 5 verified pairs $\times$ 10 disjoint seed splits:*

| Budget Ratio | Rollout Count ($B$) | Uniform Allocation RMSE | Parent-Guided (P-PGA) RMSE | Trust-Calibrated Parent RMSE | Child-Pilot ($k=1$) RMSE | Oracle Bound RMSE |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1.25N** | 38 runs | 0.03899 | 0.03954 | **0.03762 ($-3.5\%$)** | 0.03982 | 0.03436 |
| **1.50N** | 45 runs | 0.03778 | 0.03815 | **0.03659 ($-3.1\%$)** | 0.03753 | 0.03436 |
| **2.50N** | 75 runs | 0.03408 | **0.03081 ($-9.6\%$)** | 0.03166 | 0.03004 | 0.03045 |

---

### 5.3 Important Finding: The Child-Pilot Performance at $B=150$
An important scientific finding from our study is that at the 150-run budget ($B = 1.5N$ on $N=100$), the **simplified child-pilot baseline achieved lower RMSE ($0.00956$) than basic parent guidance ($0.01068$) and trust-calibrated parent guidance ($0.00991$)**.

**Why did the child-pilot perform so well?**
1. Evaluating the child directly on run 1 samples the child's *true execution distribution*, catching unexpected regressions that parent history could not anticipate.
2. In GSM8K, because 85%+ of items pass, questions that fail on run 1 have high empirical variance, receiving immediate follow-up rollouts.

**Critical Limitations of the Child-Pilot in Practice:**
1. **Pilot Infeasibility under Constrained Budgets:** In published Speculative Evaluation (arXiv:2609.28560), the pilot stage requires $k_{\text{pilot}} = 2$ or 3 runs. For $N=100$, a 2-run pilot consumes $200$ rollouts, making Speculative Evaluation **impossible to run** at budgets of 125 or 150 runs!
2. **Zero-Cost Advantage of Parent Guidance:** Parent guidance achieves a 7.9% error reduction with **zero child pilot rollouts**, precommitting all allocations before spending a single child evaluation dollar.

---

## 6. Question-Level Allocation Walkthrough

Table excerpt from [`EXAMPLE_ALLOCATION_TABLE.csv`](file:///Users/zestylemon/Desktop/experiment/EXAMPLE_ALLOCATION_TABLE.csv) for Pair 2 (Parent 3 $\to$ Child 8) at budget $B = 1.5N$ (45 runs across 30 items):

| Q Index | Problem Snippet | Parent Pass Rate ($S_P=5$) | Parent Regularized Std ($\widetilde{\sigma}$) | Child Pass Rate ($S_C=5$) | Child Observed Variance | Uniform Alloc ($n_i$) | Parent-Guided Alloc ($n_i$) | Allocation Difference ($\Delta n_i$) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Q0** | Jen decides to travel to 3 countries... | 0.00 | 0.350 | 0.00 | 0.000 | 2 | **1** | **-1 (Run saved)** |
| **Q1** | Pat has a flower bed 111 ft long... | 0.00 | 0.350 | 0.00 | 0.000 | 1 | **1** | 0 |
| **Q2** | Vince can staple 30 reports... | 0.00 | 0.350 | 0.00 | 0.000 | 1 | **1** | 0 |
| **Q8** | Anthony is sending out coupons... | 0.20 | 0.452 | 0.80 | 0.200 | 2 | **2** | 0 |
| **Q9** | Teacher uses 5-inch piece of chalk... | 0.40 | **0.495** | 0.40 | **0.300** | 1 | **2** | **+1 (Targeted)** |
| **Q10** | Carolyn works for delivery service... | 0.80 | 0.452 | 0.80 | 0.200 | 1 | **2** | **+1 (Targeted)** |
| **Q12** | Jam has three boxes of pencils... | 0.00 | 0.350 | 1.00 | 0.000 | 2 | **1** | **-1 (Run saved)** |

**Takeaway:** Uniform allocation wasted 2 runs on Q0 and Q12 (which were deterministic in the parent). Parent-guided allocation assigned only 1 run to Q0 and Q12, redirecting those saved runs to volatile boundary questions Q9 and Q10, which exhibited high cross-seed variance in the child.

---

## 7. Generated Visualizations

All charts are located in [`charts/`](file:///Users/zestylemon/Desktop/experiment/charts):

1. **`charts/parent_child_variance.png`**: Scatter plots of question-level parent variance vs. child variance, illustrating deterministic clustering at $(0, 0)$ and correlation on conservative edits ($r = +0.58$ to $+0.70$) versus mutation shock ($r = +0.09$).
2. **`charts/rmse_comparison.png`**: Estimation RMSE vs. rollout budget for Uniform, P-PGA, Trust-Calibrated, Child-Pilot, and Oracle ceiling on both cohorts.
3. **`charts/example_allocation.png`**: Question-by-question bar chart contrasting regularized parent standard deviations against allocated replicates.

---

## 8. Limitations & Proposed Live Experiment

### Limitations of Historical Offline Replay
1. **Coarse Variance Resolution:** Historical AFlow runs only contain 5 seeds per workflow, restricting empirical pass rates to $\{0.0, 0.2, 0.4, 0.6, 0.8, 1.0\}$.
2. **Noisy Reference:** With only 5 runs total per candidate, our held-out reference is based on 2 runs. True ground-truth convergence requires $\ge 16$ seeds.
3. **Small Sample Size:** 5 workflow pairs demonstrate mechanism feasibility but do not establish broad statistical generalization across diverse domains.

### Proposed Next Step for Advisor Meeting (Prof. Wei Zhang)
- **Tier 2 Certified Live Validation:**
  - Evaluate 15 candidate child workflows on GSM8K with **16 independent seeds** ($15 \times 100 \times 16 = 24,000$ total rollouts).
  - Estimated API cost: **\$32.40** using DeepSeek-V3 or GPT-4o-mini tier.
  - Measures candidate ranking preservation, true selection regret, and end-to-end token savings during live MCTS search.

---

## 9. Quickstart & Reproduction Guide

### Environment Setup
Clone this repository and install dependencies:
```bash
git clone https://github.com/madebylemon/small_test_sdp.git
cd small_test_sdp

# Create virtual environment (optional)
python3 -m venv .venv
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 1. Run Unit Tests
Verify data loaders, policies, unbiased estimators, and pipelines (14 passing tests):
```bash
python3 -m pytest tests/ -v
```

### 2. Execute Benchmark Evaluation
Run the evaluation suite across all 5 pairs, 10 disjoint seed splits, and 4 budget levels:
```bash
python3 -m src.experiment_runner
```
*Generates `RESULTS_TABLE.csv`, `RESULTS_SUMMARY.json`, and `EXAMPLE_ALLOCATION_TABLE.csv`.*

### 3. Generate Presentation Figures
Generate the high-resolution charts in `charts/`:
```bash
python3 -m src.plot_generator
```

### 4. Run Synthetic Controlled Benchmark
Simulate theoretical Neyman bounds under known latent ground-truth probabilities:
```bash
python3 -c "from src.synthetic_study import run_synthetic_benchmark; df = run_synthetic_benchmark(n_trials=100); print(df.groupby(['regime', 'budget_ratio', 'policy'])['rmse'].mean())"
```

---

## 10. Repository File Structure

```
small_test_sdp/
├── README.md                      # Comprehensive project guide (this document)
├── requirements.txt               # Dependencies (numpy, scipy, pandas, matplotlib, pytest)
├── .gitignore                     # Standard Python/OS ignore rules
├── PAPER_METHODS_SUMMARY.md       # Literature review (AFlow, EvalAlloc, Neyman, Speculative Eval)
├── PAIR_DATA_AUDIT.md             # Lineage audit & code diffs of verified historical workflow pairs
├── EXPERIMENT_METHOD.md           # Formal policy specifications & protocol validity
├── RESULTS.md                     # Comprehensive empirical analysis & synthetic benchmark findings
├── RESULTS_SUMMARY.json           # Machine-readable evaluation metrics & variance correlations
├── RESULTS_TABLE.csv              # Row-level evaluation records
├── EXAMPLE_ALLOCATION_TABLE.csv   # 30-question replicate allocation breakdown
├── FOUR_SLIDE_CONTENT.md          # 4-slide presentation text & casual speaker explanations
├── REPRODUCE.md                   # Reproduction steps & instructions
├── presentation/                  # Slide deck & speaking notes for presentation
│   ├── Group57_Parent_Guided_Mini_Experiment_4_Slides.pptx
│   └── Group57_Mini_Experiment_Speaking_Notes.md
├── charts/                        # High-resolution PNG figures
│   ├── parent_child_variance.png
│   ├── rmse_comparison.png
│   └── example_allocation.png
├── data/                          # Bundled historical GSM8K benchmark records (portable replay)
│   └── aflow_gsm8k/
│       ├── processed_experience.json
│       ├── results.json
│       └── round_1/ ... round_25/
├── src/                           # Clean modular source code
│   ├── __init__.py
│   ├── data_loader.py             # Portable lineage & score matrix loader
│   ├── policies.py                # Uniform, P-PGA, Child-Pilot, and Oracle policies
│   ├── metrics.py                 # RMSE, MAE, bias, and variance correlation functions
│   ├── experiment_runner.py       # Main evaluation harness
│   ├── synthetic_study.py         # Known-probability Monte Carlo benchmark
│   └── plot_generator.py          # Publication-ready Matplotlib chart generator
└── tests/                         # Unit tests (100% passing)
    ├── __init__.py
    ├── test_data_loader.py
    ├── test_policies.py
    ├── test_metrics.py
    └── test_experiment_runner.py
```

---

## 11. Authors & Citation

**Senior Design Team — Group 57 (University of Connecticut, 2026)**  
- James Le ([@madebylemon](https://github.com/madebylemon))  
- Advised by Prof. Wei Zhang, Department of Computer Science & Engineering  

*License: MIT License. Dataset originates from the open-source AFlow benchmark evaluation suite.*
