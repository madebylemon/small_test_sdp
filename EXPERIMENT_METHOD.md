# Experimental Methodology: Policies, Estimation, and Protocol Validity

**Research Group:** Senior Design Project — Group 57 (Prep for Prof. Wei Zhang)  
**Date:** November 19, 2026  
**Status:** Methodological Specification & Protocol Certification  

---

## 1. Problem Formulation & Objective

Let benchmark $\mathcal{D} = \{x_1, \dots, x_N\}$ be a fixed set of $N$ evaluation items. A candidate child workflow $W_C$ is evaluated stochastically: each evaluation on item $x_i$ produces an outcome $Y_{C,i,s} \in \{0, 1\}$ with latent true pass probability $p_{C,i}^* = \mathbb{E}[Y_{C,i,s}]$.

The child's true benchmark performance is:
$$\mu_C^* = \frac{1}{N} \sum_{i=1}^N p_{C,i}^*$$

Given a constrained total rollout budget $B = C \cdot N$ (where $C \in \{1.25, 1.50, 2.00, 2.50\}$ runs per item):
1. **Allocation Policy:** How should integer replicate counts $n_i \in \{1, \dots, S_{\max}\}$ be assigned to questions subject to $\sum_{i=1}^N n_i = B$?
2. **Estimator:** How should the child's score $\hat{\mu}_C$ be estimated to minimize Root Mean Squared Error (RMSE) without introducing sample-selection bias?

---

## 2. Compared Allocation Policies

All policies operate under identical total evaluation budgets $B$ and are evaluated on identical question subsets.

### Policy A: Uniform Allocation (Baseline A)
- **Mechanism:** Assigns $b = \lfloor B / N \rfloor$ runs to every question. The remaining $R = B - b \cdot N$ runs are assigned randomly to $R$ distinct questions using a prespecified reproducible pseudorandom seed.
- **Estimator:** Standard sample mean:
  $$\hat{\mu}_{\text{uniform}} = \frac{1}{N}\sum_{i=1}^N \left(\frac{1}{n_i} \sum_{s=1}^{n_i} Y_{C,i,s}\right)$$
- **Properties:** Provably unbiased. Serves as standard practice in current evolutionary search frameworks (e.g., AFlow's $S=5$).

---

### Policy B: Parent-Guided Neyman Allocation (P-PGA — Our Core Method)
- **Mechanism:** Leverages the parent workflow's $S_P = 5$ repeated historical evaluations to predict question-level variance prior to child testing.
- **Regularization:** Pass rates are regularized via a Beta(1, 1) Laplace posterior mean:
  $$\tilde{p}_{P,i} = \frac{\sum_{s=1}^{S_P} Y_{P,i,s} + 1}{S_P + 2} = \frac{k_i + 1}{7}$$
  This strictly prevents deterministic questions (0/5 or 5/5) from collapsing variance to zero:
  $$\tilde{\sigma}_{P,i} = \sqrt{\tilde{p}_{P,i}(1 - \tilde{p}_{P,i})} \in [0.350, 0.495]$$
- **Bounded Discrete Neyman Algorithm:**
  1. Base allocation: assign $n_i = 1$ to all $N$ questions ($N$ runs spent).
  2. For remaining budget $B - N$, compute marginal variance reduction:
     $$\Delta V_i(n_i) = \frac{\tilde{\sigma}_{P,i}^2}{n_i(n_i + 1)}$$
  3. Greedily increment $n_{i^*} \leftarrow n_{i^*} + 1$ for $i^* = \arg\max_{i: n_i < S_{\max}} \Delta V_i(n_i)$ until $\sum n_i = B$.
- **Precommitment & Unbiasedness:** Because replicate counts $\mathbf{n} = (n_1, \dots, n_N)$ are determined strictly before observing any child outcomes, $n_i$ is independent of child realization $\{Y_{C,i,s}\}$. The sample mean estimator is **provably unbiased** ($\text{Bias} \equiv 0$).

---

### Policy C: Child-Pilot Allocation (Speculative Evaluation Baseline)
- **Mechanism:** Inspired by Speculative Evaluation (arXiv:2609.28560), which runs a candidate-specific pilot stage before allocating second-stage rollouts.
- **Budget Constraint & Adaptation:**
  - In Speculative Evaluation, a $k_{\text{pilot}} = 2$ pilot is standard. However, when total budget $B \le 2.0N$, a 2-run pilot consumes $\ge 100\%$ of the budget, leaving zero runs for targeted allocation.
  - To implement a fair, feasible pilot baseline at constrained budgets ($B \in [1.25N, 2.5N]$), we implement a **1-run pilot** ($k_{\text{pilot}} = 1$): each question is evaluated once ($N$ runs), and the remaining $B - N$ runs are allocated via Neyman based on the pilot's empirical variance.
- **Selection Bias & Debiasing:**
  - If naive sample averaging is applied after outcome-dependent allocation, it introduces severe positive selection bias (initial successes are frozen at 1.0, while initial failures get diluted by subsequent draws).
  - We test both the naive pilot average and the **analytic debiased estimator**:
    $$\hat{\mu}_{\text{debiased}} = \hat{\mu}_{\text{naive}} - \frac{1}{N}\sum_{i=1}^N \left(1 - \frac{1}{n_i}\right) \tilde{\sigma}_{\text{pilot}, i}^2$$

---

### Policy D: Parent-Guided with Edit-Aware Trust Factor (Policy D)
- **Mechanism:** Shrinks predicted variance toward an uninformative prior based on workflow edit distance $\Delta(W_P, W_C)$:
  $$\tilde{p}_i(\Delta) = \alpha(\Delta) \tilde{p}_{P,i} + (1 - \alpha(\Delta)) p_{\text{prior}}$$
  where $\alpha(\Delta) = \exp(-2.0 \cdot \Delta)$.
- Conservative edits (prompt formatting, $\Delta \approx 0.2$) yield high trust ($\alpha \approx 0.67$), whereas radical mutations (verifier insertion, $\Delta \approx 0.5 - 0.9$) shrink trust toward uniform prior weighting ($\alpha \approx 0.16 - 0.36$).

---

### Policy E: Oracle Neyman Allocation (Reference Ceiling)
- **Mechanism:** Hindsight-only benchmark that solves bounded integer Neyman allocation using the child workflow's actual empirical standard deviation across all available runs.
- **Purpose:** Demonstrates the theoretical variance reduction ceiling under perfect variance knowledge. Not deployable in practice.

---

## 3. Protocol Validity & Experimental Rigor

### 3.1 Disjoint Evaluation and Reference Partitioning
In historical AFlow data, each round contains 5 CSV files (5 independent evaluation runs). To prevent data leakage and circular evaluation:
- We partition the 5 seeds into:
  - **3 Training Seeds (Evaluation Pool):** Available for policies to draw up to 3 runs per question ($S_{\max} = 3$).
  - **2 Held-Out Seeds (Approximate Reference):** Untouched by policies; used solely to compute the reference benchmark score:
    $$\mu_{\text{ref}} = \frac{1}{N}\sum_{i=1}^N \left(\frac{Y_{C,i,s_4} + Y_{C,i,s_5}}{2}\right)$$
- We evaluate all $\binom{5}{3} = 10$ unique combinatorial seed splits, eliminating seed-selection bias.

> [!IMPORTANT]
> **Approximate Held-Out Reference Disclaimer:** A reference constructed from two held-out seeds is subject to Bernoulli sampling noise ($\text{Var}(\mu_{\text{ref}}) = \frac{1}{4N^2}\sum \sigma_i^2$). It is not the child's exact asymptotic expected score. We refer to it strictly as an *approximate held-out reference*.

### 3.2 Evaluation Cohorts
1. **Stratified Presentation Cohort ($N=30$):** Selected deterministically from GSM8K validation items (10 consistently solved, 3 consistently failed, 17 variable boundary questions). Designed for transparent question-level visualization and presentation slides.
2. **Full GSM8K Benchmark Pool ($N=100$):** The complete 100-item validation benchmark used in historical AFlow runs.

---

## 4. Primary Metrics

1. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{K}\sum_{k=1}^K (\hat{\mu}_k - \mu_{\text{ref}, k})^2}$$
   computed across 10 disjoint seed splits and 5 verified workflow pairs.
2. **Mean Absolute Error (MAE):** $\frac{1}{K}\sum |\hat{\mu}_k - \mu_{\text{ref}, k}|$.
3. **Estimation Bias:** $\frac{1}{K}\sum (\hat{\mu}_k - \mu_{\text{ref}, k})$.
4. **Parent-Child Variance Correlation ($\rho$):** Pearson correlation between $\sigma_{P,i}^2$ and $\sigma_{C,i}^2$.
5. **Deterministic Stability:** Proportion of questions deterministic in the parent ($\sigma_P^2 = 0$) that remain deterministic in the child ($\sigma_C^2 = 0$).
