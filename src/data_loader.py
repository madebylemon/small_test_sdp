"""
Data loader and lineage parser for AFlow GSM8K historical evaluation data.
Provides read-only access to historical workflow runs and score matrices.
"""

import os
import glob
import json
import itertools
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

def get_default_data_path() -> str:
    """Resolve data path portably across environments."""
    if "AFLOW_DATA_PATH" in os.environ and os.path.exists(os.environ["AFLOW_DATA_PATH"]):
        return os.environ["AFLOW_DATA_PATH"]
    local_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "aflow_gsm8k")
    if os.path.exists(local_path):
        return local_path
    fallback_path = "/Users/zestylemon/Desktop/sdp/AFlow/data/results/results/GSM8K/graphs"
    if os.path.exists(fallback_path):
        return fallback_path
    return local_path


DEFAULT_AFLOW_PATH = get_default_data_path()


def load_lineage_tree(graphs_path: Optional[str] = None) -> Dict[str, Any]:
    """Load the MCTS experience tree defining parent-child relationships."""
    path = graphs_path or get_default_data_path()
    exp_file = os.path.join(path, "processed_experience.json")
    if not os.path.exists(exp_file):
        raise FileNotFoundError(f"Lineage file not found at {exp_file}")
    with open(exp_file, "r") as f:
        return json.load(f)


def get_verified_parent_child_pairs(graphs_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Extract verified parent-child edges where both parent and child
    have complete 5-seed CSV evaluation records.
    """
    path = graphs_path or get_default_data_path()
    tree = load_lineage_tree(path)
    pairs = []
    
    for p_str, p_data in tree.items():
        parent_id = int(p_str)
        p_csvs = sorted(glob.glob(os.path.join(path, f"round_{parent_id}", "*.csv")))
        if len(p_csvs) != 5:
            continue
            
        for outcome in ["success", "failure"]:
            children = p_data.get(outcome, {})
            for c_str, c_info in children.items():
                child_id = int(c_str)
                c_csvs = sorted(glob.glob(os.path.join(graphs_path, f"round_{child_id}", "*.csv")))
                if len(c_csvs) != 5:
                    continue
                
                # Categorize modification type
                mod_text = c_info.get("modification", "")
                mod_lower = mod_text.lower()
                if "scensemble" in mod_lower or "sc_ensemble" in mod_lower or "loop to generate multiple" in mod_lower:
                    edit_type = "ensemble_widening"
                elif "programmer" in mod_lower or "review step using the programmer" in mod_lower:
                    edit_type = "verifier_insertion"
                elif "format" in mod_lower or "prompt" in mod_lower or "unit conversion" in mod_lower:
                    edit_type = "prompt_refinement"
                else:
                    edit_type = "operator_restructuring"

                pairs.append({
                    "parent_id": parent_id,
                    "child_id": child_id,
                    "outcome": outcome,
                    "parent_score": p_data.get("score"),
                    "child_score": c_info.get("score"),
                    "modification": mod_text,
                    "edit_type": edit_type,
                    "parent_csvs": p_csvs,
                    "child_csvs": c_csvs
                })
    return pairs


def load_workflow_scores(round_id: int, graphs_path: Optional[str] = None) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Load the 5xN score matrix, question strings, and cost array for a given round.
    Returns:
        scores: np.ndarray of shape (5, N), binary 0/1 scores.
        questions: list of N question texts.
        costs: np.ndarray of shape (5, N), cumulative costs recorded in CSVs.
    """
    path = graphs_path or get_default_data_path()
    csvs = sorted(glob.glob(os.path.join(path, f"round_{round_id}", "*.csv")))
    if len(csvs) != 5:
        raise ValueError(f"Round {round_id} does not have exactly 5 CSV files in {path} (found {len(csvs)})")
    
    score_list = []
    cost_list = []
    ref_questions = None
    
    for c_path in csvs:
        df = pd.read_csv(c_path)
        qs = df["question"].tolist()
        if ref_questions is None:
            ref_questions = qs
        else:
            if qs != ref_questions:
                raise ValueError(f"Question order mismatch in round {round_id}, file {c_path}")
        score_list.append(df["score"].values.astype(float))
        cost_list.append(df["cost"].values.astype(float))
        
    scores = np.array(score_list)  # (5, N)
    costs = np.array(cost_list)    # (5, N)
    return scores, ref_questions, costs


def get_stratified_30_question_indices(graphs_path: Optional[str] = None) -> np.ndarray:
    """
    Select an honest, stratified subset of 30 matched questions from the 100 GSM8K items.
    Selection criteria:
      - 10 consistently solved items across historical workflows (high baseline accuracy).
      - 3 consistently failed items (hard items).
      - 17 unstable / variable boundary items (items with observed cross-seed flips).
    Deterministic and reproducible.
    """
    path = graphs_path or get_default_data_path()
    rounds = [2, 3, 4, 8, 10, 13]
    all_scores = []
    for r in rounds:
        s, _, _ = load_workflow_scores(r, path)
        all_scores.append(s)
    all_scores = np.concatenate(all_scores, axis=0)  # (30, 100)
    
    means = all_scores.mean(axis=0)  # Pass rate per question across 30 historical evaluations
    within_var = all_scores.var(axis=0, ddof=1)
    
    always_pass = np.where(means == 1.0)[0]
    always_fail = np.where(means == 0.0)[0]
    variable = np.where((means > 0.0) & (means < 1.0))[0]
    
    # Deterministic selection
    selected_pass = always_pass[:10]
    selected_fail = always_fail[:3]
    # Pick the top 17 highest variance items among variable ones
    sorted_var_idx = variable[np.argsort(-within_var[variable])]
    selected_var = sorted_var_idx[:17]
    
    combined = np.sort(np.concatenate([selected_pass, selected_fail, selected_var]))
    assert len(combined) == 30, f"Expected 30 items, got {len(combined)}"
    return combined


def get_all_disjoint_seed_splits() -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """
    Return all 10 possible combinations of partitioning 5 seeds into
    3 training (evaluation) seeds and 2 held-out reference seeds.
    Each tuple is (train_seeds, heldout_seeds).
    """
    all_seeds = list(range(5))
    splits = []
    for train_combo in itertools.combinations(all_seeds, 3):
        heldout_combo = tuple(s for s in all_seeds if s not in train_combo)
        splits.append((train_combo, heldout_combo))
    return splits
