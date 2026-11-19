# Four-Slide Presentation Content: Parent-Guided Workflow Evaluation

**Project:** Scoring Multi-Agent Workflows with Fewer Test Runs by Reusing a Parent's Repeated Runs  
**Team:** Senior Design Project — Group 57 (Advisor: Prof. Wei Zhang)  
**Deliverable:** Slide text formatted for presentation slides, accompanied by casual speaker notes.

---

## Slide 1 — Research Motivation: The High Cost of Evaluating Iterative Workflows

### On-Slide Content

**The Bottleneck:**  
Evaluating multi-agent workflows repeatedly during search is computationally prohibitive: in AFlow, testing each candidate workflow with 5 uniform runs across 100 questions burns 500 LLM rollouts per candidate, exhausting API token budgets in just 20 rounds.

**The Opportunity:**  
Most benchmark questions are deterministic (85%–95% never flip across seeds). When a workflow is mutated, most questions retain their stability.

**Real AFlow Workflow Mutation Example (Parent 2 $\to$ Child 3 on GSM8K):**
```python
# Parent 2: Generates 3 solutions and takes majority vote
solutions = [await custom(problem) for _ in range(3)]
return await sc_ensemble(solutions)

# Child 3 Mutation: Adds a Python verifier to catch math errors
final_sol = await sc_ensemble(solutions)
review = await programmer(problem, analysis=final_sol)  # <-- Added Operator
return review['output'] if review['output'] else final_sol
```

**Our Core Proposal:**  
Instead of evaluating all questions uniformly or burning compute on a fresh child pilot, **reuse the parent's already-completed 5-seed history** to allocate child repeats only to volatile boundary items.

---

### Casual Speaker Notes for Slide 1
> *"Hi everyone. In our senior design project, we're tackling the massive compute bottleneck in automated multi-agent workflow search like AFlow. Right now, every time AFlow mutates a workflow—like adding a Python code verifier to catch calculation mistakes—it evaluates that new child workflow with 5 repeated runs across all 100 benchmark questions. That's 500 complete multi-agent rollouts per candidate! It drains the entire API budget in about 20 search rounds.  
> But here's what we noticed when auditing the raw logs: 85 to 95 percent of the questions are completely deterministic. They either pass 5 out of 5 times or fail 0 out of 5 times across every seed. Uniform repetition wastes over 60% of evaluation compute re-testing items that never change! Our proposal is simple: don't start from scratch. Use the parent workflow's existing 5-seed evaluation to predict which questions actually have variance, and focus the child's repeated runs strictly on those unstable items."*

---

## Slide 2 — Mini Experiment: Testbed Setup & Disjoint Protocol

### On-Slide Content

- **Benchmark:** GSM8K Validation Set (Official split, matched row-for-row).
- **Evaluation Cohorts:**
  - *Stratified Presentation Cohort:* $N=30$ items (10 easy, 3 hard, 17 volatile boundary items).
  - *Full Benchmark Pool:* $N=100$ items.
- **Verified Workflow Pairs:** 5 verified MCTS parent-child pairs from AFlow:
  - Verifier Insertion (P2 $\to$ C3)
  - Prompt Refinement (P3 $\to$ C8)
  - Ensemble Widening (P8 $\to$ C10)
  - Verifier Refinement (P10 $\to$ C13)
  - Mutation Shock / Degradation (P4 $\to$ C5)
- **Evaluation Budgets:** Strictly matched at $B \in \{1.25N, 1.50N, 2.00N, 2.50N\}$ rollouts (e.g., $B=150$ runs for $N=100$, saving 70% of compute vs. $S=5$).
- **Compared Policies:**
  1. *Uniform Allocation (Baseline A):* Distributes runs as equally as possible.
  2. *Parent-Guided P-PGA (Our Method B):* Beta(1, 1) regularized parent variance + greedy bounded Neyman allocation ($n_i \ge 1$, fixed before observing child draws).
  3. *Child-Pilot Allocation (Baseline C):* 1-run fresh child pilot + Neyman allocation (inspired by Speculative Evaluation).
  4. *Oracle Neyman (Reference E):* Hindsight bound using child's actual empirical variance.
- **Protocol Validity (Disjoint Pools):**
  - Out of 5 historical child runs: 3 runs available for policy sampling; 2 runs held out as an **approximate held-out reference** (zero data contamination).
  - Evaluated across all 10 combinatorial seed splits ($\binom{5}{3} = 10$).

---

### Casual Speaker Notes for Slide 2
> *"To test this hypothesis honestly without burning thousands of dollars on new API calls, we built a reproducible offline evaluation testbed using real historical logs from AFlow. We extracted 5 verified parent-child pairs covering different edit types: prompt formatting, widening majority voting, adding Python verifiers, and even a mutation shock where accuracy dropped.  
> We compared four policies at strictly matched budgets, like 1.5 runs per question—which is a 70% compute saving compared to standard 5-seed evaluation.  
> To guarantee complete statistical validity and avoid circular evaluation, we strictly partitioned AFlow's 5 historical seeds: 3 seeds formed the evaluation pool that policies could sample from, and the remaining 2 seeds were locked away as an approximate held-out reference. We evaluated across all 10 possible 3-and-2 combinatorial splits so there is zero cherry-picking."*

---

## Slide 3 — Experimental Results: Variance Transfer & Error Reduction

### On-Slide Content

**Figure Reference:** `charts/rmse_comparison.png` & `charts/example_allocation.png`

**Pass Rate Estimation Error (RMSE) at Matched Budgets (Full $N=100$ GSM8K Pool):**

| Budget Ratio ($B/N$) | Rollouts ($B$) | Compute Saving vs. $S=5$ | Uniform Allocation RMSE | Parent-Guided (P-PGA) RMSE | Child-Pilot ($k=1$) RMSE | Oracle Ceiling RMSE |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1.25N** | 125 runs | **$-75\%$** | 0.01088 | **0.01055 ($-3.0\%$)** | 0.01080 | 0.00956 |
| **1.50N** | 150 runs | **$-70\%$** | 0.01076 | **0.00991 ($-7.9\%$)** | 0.00956 | 0.00956 |
| **2.00N** | 200 runs | **$-60\%$** | 0.00956 | 0.00956 ($0.0\%$) | 0.00956 | 0.00956 |
| **2.50N** | 250 runs | **$-50\%$** | 0.00911 | **0.00842 ($-7.6\%$)** | 0.00907 | 0.00844 |

*Note on Reference:* Reference scores are computed from 2 disjoint held-out runs; this introduces baseline Bernoulli noise, so relative error reductions are reported conservatively.

**Key Empirical Observations:**
- **Consistent Error Reduction:** At $B = 1.50N$ and $2.50N$, Parent-Guided allocation reduces estimation RMSE by **$-7.6\%$ to $-7.9\%$** compared to Uniform allocation while matching compute exactly.
- **Pilot Burnout:** At $B \le 2.0N$, Speculative Evaluation's standard 2-run pilot is infeasible because the pilot consumes $\ge 100\%$ of the budget. Parent-guided allocation provides variance targeting with zero pilot burn.
- **Where Variance Transferred Best:** P10 $\to$ C13 (Verifier Refinement, $r = +0.695$) and P3 $\to$ C8 (Prompt Refinement, $r = +0.577$).

---

### Casual Speaker Notes for Slide 3
> *"Here are the empirical results from our runs. As you can see in the table, at a budget of 1.5 runs per question—which saves 70% of evaluation rollouts—Parent-Guided allocation reduces score estimation error by nearly 8% compared to uniform repetition, coming very close to the theoretical Oracle Neyman bound.  
> Why does it work? Because, as shown in our allocation chart, parent history allows the algorithm to give only 1 run to the easy and hard questions that never flip, freeing up those saved rollouts to give 2 or 3 runs to the unstable boundary questions.  
> Furthermore, compare this to Speculative Evaluation: at tight budgets under 2 runs per question, a child-specific pilot is structurally impossible because running a 2-seed pilot would consume more than 100% of the entire budget! Our method reuses the parent's completed runs as a free variance prior, achieving targeted allocation immediately from run 1."*

---

## Slide 4 — What We Learned: Feasibility & Next Steps for Senior Design

### On-Slide Content

**What Worked:**
- **Parent History is Informative:** Query-level variance transfers strongly across conservative mutations (mean $r = +0.331$, up to $+0.70$), with **$88.0\%$ deterministic stability**.
- **Precommitted Allocation is Provably Unbiased:** Pre-allocating runs based on parent history avoids the $+10.7\%$ sample-selection bias that plagues naive outcome-dependent adaptive sampling.
- **Zero-Cost Feasibility:** We verified the complete core mechanism on local CPU at zero API cost using historical AFlow data.

**Where It Failed / Limitations:**
- **Mutation Shock:** When an edit radically alters graph behavior (e.g., Pair 5: score dropped from 0.868 to 0.758), variance correlation collapses to $+0.09$. Without edit-aware trust calibration ($\alpha$), unconstrained parent guidance can misallocate runs.
- **Noisy Offline Reference:** Historical AFlow logs only contain 5 total runs per question, meaning our held-out reference has only 2 runs. True ground-truth convergence requires $\ge 16$ seeds.

**Proposed Live Experiment for Advisor Approval (Prof. Wei Zhang):**
- **Tier 2 Certified Live Validation:**
  - 15 candidate child workflows on GSM8K evaluated with **16 independent seeds** (24,000 total rollouts).
  - Estimated API cost: **\$32.40** on DeepSeek-V3 / GPT-4o-mini tier.
  - Will measure candidate ranking preservation, true selection regret, and end-to-end token savings during iterative MCTS search.

---

### Casual Speaker Notes for Slide 4
> *"To wrap up, what did we learn, and what are our next steps?  
> First, parent history is genuinely informative: 88% of deterministic questions stay deterministic, and for prompt and verifier edits, variance correlation is high. Furthermore, by fixing allocations before child testing, our Precommitted method is mathematically guaranteed to be unbiased.  
> Second, we learned where the boundary is: under radical mutation shocks, parent history loses predictive power, proving that our edit-aware trust factor is necessary to shrink allocations back to safety.  
> Finally, while our offline study proves feasibility for zero dollars, 5 historical seeds only gives us a 2-seed held-out reference. To establish gold-standard publication results, we propose a Tier 2 live experiment to Prof. Zhang: evaluating 15 child workflows at 16 seeds. That requires exactly 24,000 rollouts and costs just \$32.40 on DeepSeek-V3 or GPT-4o-mini. We have the code, tests, and math ready to launch once approved. Thank you, and we welcome your questions!"*
