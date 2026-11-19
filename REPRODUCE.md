# Reproduction Guide: Running the Parent-Guided Evaluation Experiment

**Research Group:** Senior Design Project — Group 57 (Prep for Prof. Wei Zhang)  
**Date:** November 19, 2026  
**Environment:** Local CPU Only (Python 3.9+) | Free & Offline  

---

## 1. Prerequisites & Environment Setup

This experiment is completely offline, uses zero API tokens, and runs locally on standard CPU in $< 30$ seconds.

### Required Python Packages
```bash
python3 -m pip install numpy scipy pandas matplotlib pytest
```

### Input Data Source
Historical evaluation logs are bundled in `data/aflow_gsm8k/` for seamless portability:
```
data/aflow_gsm8k/
├── processed_experience.json   # Reconstructed parent-child MCTS lineage graph
├── results.json                # Validation metrics and metadata
└── round_1/ ... round_25/      # 5 independent evaluation runs per workflow round
```
*Note:* The data loader automatically resolves `data/aflow_gsm8k/` by default. You can also override the data directory by setting the `AFLOW_DATA_PATH` environment variable. Zero modifications are made to any external repositories.

---

## 2. Step-by-Step Execution Instructions

Navigate to the experiment directory:
```bash
cd /Users/zestylemon/Desktop/experiment
```

### Step 1: Run Automated Verification Tests
Verify all 14 unit tests across data loading, allocation policies, statistical metrics, and experimental pipelines:
```bash
python3 -m pytest tests/ -v
```
*Expected Output:* `14 passed in ~0.4s`.

### Step 2: Run Full Historical Benchmark Evaluation
Execute the evaluation runner across the 5 verified workflow pairs, 10 disjoint seed splits, and 4 matched budget levels:
```bash
python3 -m src.experiment_runner
```
*Output Files Generated:*
- `RESULTS_TABLE.csv`: Complete row-level results across all splits, pairs, budgets, and policies.
- `RESULTS_SUMMARY.json`: High-level aggregated statistics and variance transfer correlations.
- `EXAMPLE_ALLOCATION_TABLE.csv`: Detailed question-by-question replicate allocation breakdown.

### Step 3: Run Explicitly Labeled Synthetic Controlled Study
Execute the synthetic Monte Carlo benchmark testing known latent probabilities under strong transfer and mutation shock:
```bash
python3 -c "from src.synthetic_study import run_synthetic_benchmark; df = run_synthetic_benchmark(n_trials=100); print(df.groupby(['regime', 'budget_ratio', 'policy'])['rmse'].mean())"
```

### Step 4: Generate Presentation-Ready Figures
Generate the high-resolution charts for presentation slides:
```bash
python3 -m src.plot_generator
```
*Output Figures Generated in `charts/`:*
- `charts/parent_child_variance.png`: Question-level parent vs. child variance scatter plots.
- `charts/rmse_comparison.png`: Estimation RMSE comparison across matched budgets.
- `charts/example_allocation.png`: Question-level variance and replicate allocation bar chart.

---

## 3. Directory Layout

```
/Users/zestylemon/Desktop/experiment/
├── PAPER_METHODS_SUMMARY.md      # Literature analysis (AFlow, EvalAlloc, Neyman, Speculative Eval)
├── PAIR_DATA_AUDIT.md            # Lineage audit & code diffs of verified historical workflow pairs
├── EXPERIMENT_METHOD.md          # Formal policy specifications, debiased estimators, and protocol validity
├── RESULTS.md                    # In-depth empirical results, tables, and synthetic study analysis
├── RESULTS_SUMMARY.json          # Machine-readable evaluation metrics and variance correlations
├── RESULTS_TABLE.csv             # Raw per-trial evaluation metrics
├── EXAMPLE_ALLOCATION_TABLE.csv  # 30-question replicate allocation breakdown
├── FOUR_SLIDE_CONTENT.md         # Content and casual speaker notes for 4 presentation slides
├── REPRODUCE.md                  # This reproduction guide
├── charts/                       # High-resolution presentation figures (PNG)
│   ├── parent_child_variance.png
│   ├── rmse_comparison.png
│   └── example_allocation.png
├── src/                          # Clean, modular Python source
│   ├── data_loader.py
│   ├── policies.py
│   ├── metrics.py
│   ├── experiment_runner.py
│   ├── synthetic_study.py
│   └── plot_generator.py
└── tests/                        # Comprehensive test suite (14 passing tests)
    ├── test_data_loader.py
    ├── test_policies.py
    ├── test_metrics.py
    └── test_experiment_runner.py
```
