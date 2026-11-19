"""
Unit tests for data loader.
"""

import os
import numpy as np
import pytest
from src.data_loader import (
    load_lineage_tree,
    get_verified_parent_child_pairs,
    load_workflow_scores,
    get_stratified_30_question_indices,
    get_all_disjoint_seed_splits,
    DEFAULT_AFLOW_PATH
)


def test_lineage_loading():
    tree = load_lineage_tree(DEFAULT_AFLOW_PATH)
    assert isinstance(tree, dict)
    assert len(tree) > 0
    assert "8" in tree or "2" in tree


def test_verified_pairs():
    pairs = get_verified_parent_child_pairs(DEFAULT_AFLOW_PATH)
    assert len(pairs) >= 20
    for p in pairs:
        assert p["parent_id"] != p["child_id"]
        assert len(p["parent_csvs"]) == 5
        assert len(p["child_csvs"]) == 5


def test_load_workflow_scores():
    scores, questions, costs = load_workflow_scores(2, DEFAULT_AFLOW_PATH)
    assert scores.shape == (5, 100)
    assert len(questions) == 100
    assert costs.shape == (5, 100)
    assert np.all((scores == 0.0) | (scores == 1.0))


def test_stratified_30_indices():
    indices = get_stratified_30_question_indices(DEFAULT_AFLOW_PATH)
    assert len(indices) == 30
    assert len(set(indices)) == 30  # All unique
    assert np.all(indices >= 0) and np.all(indices < 100)


def test_seed_splits():
    splits = get_all_disjoint_seed_splits()
    assert len(splits) == 10  # 5 choose 3 = 10
    for train, test in splits:
        assert len(train) == 3
        assert len(test) == 2
        assert set(train).isdisjoint(set(test))
        assert set(train) | set(test) == set(range(5))
