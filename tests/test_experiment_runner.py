"""
Unit test for experiment runner pipeline.
"""

import os
import pytest
import numpy as np
from src.experiment_runner import (
    run_experiment_on_subset,
    extract_question_level_example,
    SELECTED_PAIRS
)


def test_experiment_runner_subset():
    indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    df, v_stats, qs = run_experiment_on_subset(indices, subset_name="test_10")
    
    assert len(df) > 0
    assert "rmse" in df.columns
    assert "policy" in df.columns
    assert "Uniform" in df["policy"].values
    assert "Parent-Guided (P-PGA)" in df["policy"].values
    assert len(v_stats) == len(SELECTED_PAIRS)
    assert len(qs) == 10


def test_extract_question_level_example():
    indices = np.array([0, 1, 2, 3, 4])
    df_ex = extract_question_level_example(SELECTED_PAIRS[0], indices, budget_ratio=1.5)
    assert len(df_ex) == 5
    assert "parent_pass_rate" in df_ex.columns
    assert "parent_guided_alloc" in df_ex.columns
    assert "uniform_alloc" in df_ex.columns
