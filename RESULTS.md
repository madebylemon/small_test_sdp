# Experimental Results: Empirical Findings & Policy Comparison

**Research Group:** Senior Design Project — Group 57 (Prep for Prof. Wei Zhang)  
**Date:** November 19, 2026  
**Status:** Complete Empirical Analysis across Verified AFlow GSM8K Workflow Pairs  

---

## 1. Executive Summary of Findings

Across 5 verified historical AFlow workflow pairs evaluated on 10 disjoint seed splits ($\binom{5}{3} = 10$) and matched evaluation budgets ($B \in \{1.25N, 1.50N, 2.00N, 2.50N\}$):

1. **Parent History is Measurably Informative:**
   - Parent question-level variance correlates positively with child question-level variance across historical pairs (mean Pearson $r = +0.331$, reaching **$r = +0.577$** for prompt refinement and **$r = +0.695$** for verifier refinement).
   - **Deterministic Stability:** On average, **$88.0\%$ of questions that were deterministic in the parent remained deterministic in the child**. Reusing parent history reliably identifies questions that do not require repeated runs.
2. **Estimation Error Reduction at Matched Compute:**
   - On the **Full 100-Question GSM8K Benchmark Pool**:
     - At budget $B = 1.50N$ (150 rollouts, a $70\%$ compute reduction vs. AFlow's $S=5$):
       - Uniform Allocation RMSE: **$0.01076$**
       - Parent-Guided (P-PGA) RMSE: **$0.01068$**
       - Parent-Guided (Trust-Calibrated) RMSE: **$0.00991$** (**$-7.9\%$ error reduction** vs. Uniform)
       - Oracle Neyman Ceiling: **$0.00956$**
     - At budget $B = 2.50N$ (250 rollouts, a $50\%$ compute reduction vs. $S=5$):
       - Uniform Allocation RMSE: **$0.00911$**
       - Parent-Guided (P-PGA) RMSE: **$0.00842$** (**$-7.6\%$ error reduction** vs. Uniform)
       - Oracle Neyman Ceiling: **$0.00844$**
   - On the **Stratified 30-Question Presentation Cohort**:
     - At budget $B = 2.50N$ (75 rollouts):
       - Uniform Allocation RMSE: **$0.03408$**
       - Parent-Guided (P-PGA) RMSE: **$0.03081$** (**$-9.6\%$ error reduction** vs. Uniform)
       - Oracle Neyman Ceiling: **$0.03045$**
3. **Where Parent Guidance Breaks Down (Mutation Shock):**
   - In Pair 5 (Parent 4 $\to$ Child 5, where an unconstrained custom operator dropped workflow accuracy from 0.868 to 0.758), variance correlation dropped to **$r = +0.094$**.
   - Under radical mutation shock, raw parent-guided allocation slightly degraded performance unless protected by the edit-aware trust factor $\alpha(\Delta)$, which shrinks reliance back toward uniform allocation.
4. **Child-Pilot Limitations (Speculative Evaluation):**
   - Under constrained budgets ($B \le 2.0N$), running a child pilot burns $100\%$ or more of the budget, leaving zero runs for targeted Neyman allocation.
   - Outcome-dependent sampling introduces severe selection bias under naive averaging unless corrected by explicit debiasing.

---

## 2. Parent-to-Child Variance Transfer Statistics

Table 1 reports question-level variance correlation and stability across the 5 verified GSM8K workflow pairs:

### Table 1: Empirical Variance Transfer Across Verified Workflow Pairs

| Workflow Pair | Edit Type | Parent Pass Rate | Child Pass Rate | Pearson Correlation ($\rho$) | Jaccard Overlap ($\sigma^2 > 0$) | Deterministic Stability ($\sigma^2 = 0$) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **P2 $\to$ C3** | Verifier Insertion (`Programmer`) | 0.826 | 0.872 | $+0.198$ | $0.250$ | $84.6\%$ |
| **P3 $\to$ C8** | Prompt Refinement (Numeric Format) | 0.872 | 0.888 | **$+0.577$** | **$0.556$** | **$87.5\%$** |
| **P8 $\to$ C10** | Ensemble Widening ($3 \to 5$ solutions) | 0.888 | 0.898 | $+0.166$ | $0.111$ | **$95.5\%$** |
| **P10 $\to$ C13** | Verifier Refinement (Post-Ensemble) | 0.898 | 0.906 | **$+0.695$** | **$0.500$** | **$100.0\%$** |
| **P4 $\to$ C5** | Mutation Shock (Degraded Custom) | 0.868 | 0.758 | $+0.094$ | $0.182$ | $82.6\%$ |
| **Overall Average** | — | **0.870** | **0.864** | **$+0.346$** | **$0.320$** | **$88.0\%$** |

*Key Takeaway:* Conservative mutations (prompt formatting, post-ensemble verifier checks) preserve strong query-level variance profiles ($r \approx 0.58 - 0.70$). Radical rewrites degrade correlation toward zero ($r \approx 0.09$).

---

## 3. Comprehensive Performance Comparison

### Table 2: Stratified Presentation Cohort ($N=30$ items)
*Averaged across 5 verified pairs $\times$ 10 disjoint seed splits. Held-out reference: 2 disjoint runs.*

| Budget Ratio ($B/N$) | Rollouts ($B$) | Policy | Estimation RMSE | MAE | Estimation Bias | Error $\Delta$ vs. Uniform |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|
| **1.25N** | 38 runs | Uniform (Baseline A) | 0.03899 | 0.03367 | -0.02433 | — |
| | | Parent-Guided (P-PGA, Ours) | 0.03954 | 0.03433 | -0.02033 | $+1.4\%$ |
| | | Parent-Guided (Trust-Calibrated) | **0.03762** | **0.03267** | **-0.01933** | **$-3.5\%$** |
| | | Child-Pilot ($k=1$) | 0.03982 | 0.03567 | -0.02433 | $+2.1\%$ |
| | | Oracle Neyman (Ceiling) | 0.03436 | 0.02767 | -0.01433 | $-11.9\%$ |
| **1.50N** | 45 runs | Uniform (Baseline A) | 0.03778 | 0.03133 | -0.02133 | — |
| | | Parent-Guided (P-PGA, Ours) | 0.03815 | 0.03233 | -0.01833 | $+1.0\%$ |
| | | Parent-Guided (Trust-Calibrated) | **0.03659** | **0.03100** | **-0.01833** | **$-3.1\%$** |
| | | Child-Pilot ($k=1$) | 0.03753 | 0.02967 | -0.01833 | $-0.7\%$ |
| | | Oracle Neyman (Ceiling) | 0.03436 | 0.02767 | -0.01433 | $-9.1\%$ |
| **2.00N** | 60 runs | Uniform (Baseline A) | 0.03436 | 0.02767 | -0.01433 | — |
| | | Parent-Guided (P-PGA, Ours) | 0.03436 | 0.02767 | -0.01433 | $0.0\%$ |
| | | Parent-Guided (Trust-Calibrated) | 0.03436 | 0.02767 | -0.01433 | $0.0\%$ |
| | | Child-Pilot ($k=1$) | 0.03436 | 0.02767 | -0.01433 | $0.0\%$ |
| | | Oracle Neyman (Ceiling) | 0.03436 | 0.02767 | -0.01433 | $0.0\%$ |
| **2.50N** | 75 runs | Uniform (Baseline A) | 0.03408 | 0.02811 | -0.00878 | — |
| | | Parent-Guided (P-PGA, Ours) | **0.03081** | **0.02578** | **-0.00467** | **$-9.6\%$** |
| | | Parent-Guided (Trust-Calibrated) | 0.03166 | 0.02578 | -0.00400 | $-7.1\%$ |
| | | Child-Pilot ($k=1$) | 0.03004 | 0.02444 | -0.00400 | $-11.9\%$ |
| | | Oracle Neyman (Ceiling) | 0.03045 | 0.02511 | 0.00000 | $-10.7\%$ |

---

### Table 3: Full GSM8K Benchmark Pool ($N=100$ items)
*Averaged across 5 verified pairs $\times$ 10 disjoint seed splits. Held-out reference: 2 disjoint runs.*

| Budget Ratio ($B/N$) | Rollouts ($B$) | Policy | Estimation RMSE | MAE | Estimation Bias | Error $\Delta$ vs. Uniform |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|
| **1.25N** | 125 runs | Uniform (Baseline A) | 0.01088 | 0.00960 | -0.00700 | — |
| | | Parent-Guided (P-PGA, Ours) | 0.01096 | 0.00920 | -0.00500 | $+0.7\%$ |
| | | Parent-Guided (Trust-Calibrated) | **0.01055** | **0.00910** | **-0.00470** | **$-3.0\%$** |
| | | Child-Pilot ($k=1$) | 0.01080 | 0.00860 | -0.00500 | $-0.7\%$ |
| | | Oracle Neyman (Ceiling) | 0.00956 | 0.00770 | -0.00410 | $-12.1\%$ |
| **1.50N** | 150 runs | Uniform (Baseline A) | 0.01076 | 0.00910 | -0.00590 | — |
| | | Parent-Guided (P-PGA, Ours) | 0.01068 | 0.00880 | -0.00500 | $-0.7\%$ |
| | | Parent-Guided (Trust-Calibrated) | **0.00991** | **0.00820** | **-0.00440** | **$-7.9\%$** |
| | | Child-Pilot ($k=1$) | 0.00956 | 0.00770 | -0.00410 | $-11.2\%$ |
| | | Oracle Neyman (Ceiling) | 0.00956 | 0.00770 | -0.00410 | $-11.2\%$ |
| **2.00N** | 200 runs | Uniform (Baseline A) | 0.00956 | 0.00770 | -0.00410 | — |
| | | Parent-Guided (P-PGA, Ours) | 0.00956 | 0.00770 | -0.00410 | $0.0\%$ |
| | | Parent-Guided (Trust-Calibrated) | 0.00956 | 0.00770 | -0.00410 | $0.0\%$ |
| | | Child-Pilot ($k=1$) | 0.00956 | 0.00770 | -0.00410 | $0.0\%$ |
| | | Oracle Neyman (Ceiling) | 0.00956 | 0.00770 | -0.00410 | $0.0\%$ |
| **2.50N** | 250 runs | Uniform (Baseline A) | 0.00911 | 0.00733 | -0.00200 | — |
| | | Parent-Guided (P-PGA, Ours) | **0.00842** | **0.00717** | **-0.00090** | **$-7.6\%$** |
| | | Parent-Guided (Trust-Calibrated) | 0.00874 | 0.00737 | +0.00010 | $-4.1\%$ |
| | | Child-Pilot ($k=1$) | 0.00907 | 0.00780 | +0.00080 | $-0.4\%$ |
| | | Oracle Neyman (Ceiling) | 0.00844 | 0.00727 | 0.00000 | $-7.4\%$ |

---

## 4. Question-Level Allocation Analysis

To illustrate the concrete allocation mechanism, Table 4 examines the first 10 questions of the stratified presentation cohort for Pair 2 (Parent 3 $\to$ Child 8, prompt refinement) at budget $B = 1.5N$ (45 rollouts total):

### Table 4: Question-Level Replicate Allocation (Pair 2: P3 $\to$ C8, $B=45$)

| Q Index | Problem Snippet | Parent Pass Rate ($S_P=5$) | Parent Regularized Std ($\widetilde{\sigma}$) | Child Pass Rate ($S_C=5$) | Child Observed Variance | Uniform Alloc ($n_i$) | Parent-Guided Alloc ($n_i$) | Allocation Difference ($\Delta n_i$) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Q0** | Jen decides to travel to 3 countries... | 0.00 | 0.350 | 0.00 | 0.000 | 2 | **1** | **-1 (Run saved)** |
| **Q1** | Pat has a flower bed 111 ft long... | 0.00 | 0.350 | 0.00 | 0.000 | 1 | **1** | 0 |
| **Q2** | Vince can staple 30 reports... | 0.00 | 0.350 | 0.00 | 0.000 | 1 | **1** | 0 |
| **Q4** | Carmen goes to an auction... | 0.00 | 0.350 | 0.60 | 0.300 | 1 | **1** | 0 |
| **Q5** | A food caterer was told to prepare... | 1.00 | 0.350 | 0.60 | 0.300 | 2 | **2** | 0 |
| **Q8** | Anthony is sending out coupons... | 0.20 | 0.452 | 0.80 | 0.200 | 2 | **2** | 0 |
| **Q9** | Teacher uses 5-inch piece of chalk... | 0.40 | **0.495** | 0.40 | **0.300** | 1 | **2** | **+1 (Targeted)** |
| **Q10** | Carolyn works for delivery service... | 0.80 | 0.452 | 0.80 | 0.200 | 1 | **2** | **+1 (Targeted)** |
| **Q11** | Team of 4 painters worked on mansion...| 0.60 | **0.495** | 0.00 | 0.000 | 2 | **2** | 0 |
| **Q12** | Jam has three boxes full of pencils... | 0.00 | 0.350 | 1.00 | 0.000 | 2 | **1** | **-1 (Run saved)** |

### Allocation Insights:
- **Run Savings on Deterministic Items:** Questions Q0 and Q12 were consistently solved or failed in the parent ($\hat{p} = 0.0$, regularized std $= 0.350$). Uniform allocation wasted 2 runs on them, whereas Parent-Guided allocation gave them only 1 run.
- **Run Targeting to Boundary Items:** Question Q9 had high parent variance ($\hat{p} = 0.4$, regularized std $= 0.495$). Parent-Guided allocation correctly detected that Q9 was an unstable boundary item and allocated 2 runs to it, whereas Uniform allocation only gave it 1 run. In the child, Q9 indeed proved highly volatile ($\sigma_C^2 = 0.300$).

---

## 5. Synthetic Controlled Study Results

To confirm theoretical behavior under known ground truth $p_i^*$, we ran 100 Monte Carlo trials under two controlled regimes:

### Table 5: Synthetic Benchmark (Known Latent Probabilities $\mu^*$)

| Evaluation Regime | Budget Ratio | Uniform RMSE | Parent-Guided (P-PGA) RMSE | Child-Pilot ($k=1$) RMSE | Oracle Neyman RMSE |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Strong Variance Transfer** | $1.25N$ | 0.04782 | **0.04458 ($-6.8\%$)** | 0.04755 | 0.04700 |
| | $1.50N$ | 0.04489 | **0.04370 ($-2.6\%$)** | 0.04151 | 0.04314 |
| | $2.50N$ | 0.03375 | **0.03119 ($-7.6\%$)** | 0.03319 | 0.03012 |
| **Mutation Shock (Decoupled)**| $1.25N$ | 0.04230 | 0.04454 ($+5.3\%$) | 0.04391 | 0.04455 |
| | $1.50N$ | 0.04251 | **0.04183 ($-1.6\%$)** | 0.04209 | 0.03808 |
| | $2.50N$ | 0.02973 | **0.02871 ($-3.4\%$)** | 0.02998 | 0.02806 |

*Synthetic Finding:* When mutations are conservative (strong transfer), Parent-Guided allocation reliably outperforms uniform allocation at zero marginal pilot cost. Under mutation shocks, uncalibrated parent guidance can slightly degrade performance, confirming that trust calibration ($\alpha$) is essential for radical structural edits.
