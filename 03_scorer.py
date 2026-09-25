#!/usr/bin/env python3
"""
Phase 1 — Macro F_{0.5} Scorer
================================
Calculates per-Source-1 F_0.5, then macro-averages across all Source 1 IDs.

Singleton rules (from the problem statement):
  - True matches = ∅, Predicted = ∅  → F_0.5 = 1.0
  - True matches = ∅, Predicted ≠ ∅  → F_0.5 = 0.0
  - True matches ≠ ∅, Predicted = ∅  → F_0.5 = 0.0

This module is importable (for use in training scripts) and also runnable
standalone with two TSV files:
  python ml_challenge/03_scorer.py <ground_truth.tsv> <predictions.tsv>
"""

import csv
import sys
from pathlib import Path


BETA = 0.5
BETA_SQ = BETA ** 2  # 0.25


def f_beta(precision, recall, beta_sq=BETA_SQ):
    """Compute F_beta from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def per_entity_f05(true_set, pred_set):
    """
    F_0.5 for a single Source 1 entity.

    Parameters
    ----------
    true_set : set  — ground-truth matched IDs (empty for singletons)
    pred_set : set  — predicted matched IDs

    Returns
    -------
    float  — F_0.5 score in [0, 1]
    """
    # Singleton: no true matches
    if len(true_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0

    # Non-singleton: no predictions
    if len(pred_set) == 0:
        return 0.0

    tp = len(true_set & pred_set)
    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    return f_beta(precision, recall)


def macro_f05(ground_truth, predictions):
    """
    Macro-averaged F_0.5 across all Source 1 entities.

    Parameters
    ----------
    ground_truth : dict[str, set]
        {source1_entity_id: set(matched_ids)}  — empty set for singletons.
    predictions : dict[str, set]
        {source1_entity_id: set(predicted_ids)}  — missing keys treated as
        empty predictions.

    Returns
    -------
    float  — macro F_0.5
    """
    if not ground_truth:
        return 0.0

    total = 0.0
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        total += per_entity_f05(true_set, pred_set)

    return total / len(ground_truth)


def load_tsv_as_dict(path, id_col="source1_entity_id", list_col="matched_entity_ids"):
    """
    Load a two-column TSV into {id: set(values)}.

    Works for both ground truth and prediction files.  Streams line by line
    for bounded memory.
    """
    result = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        id_idx = header.index(id_col)
        list_idx = header.index(list_col)
        for row in reader:
            s1_id = row[id_idx]
            val_str = row[list_idx] if len(row) > list_idx else ""
            if val_str.strip():
                result[s1_id] = set(val_str.split(","))
            else:
                result[s1_id] = set()
    return result


# ── CLI interface ────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print("Usage: python ml_challenge/03_scorer.py <ground_truth.tsv> <predictions.tsv>")
        sys.exit(1)

    gt_path, pred_path = sys.argv[1], sys.argv[2]
    gt = load_tsv_as_dict(gt_path)
    preds = load_tsv_as_dict(pred_path)

    score = macro_f05(gt, preds)
    print(f"Macro F_0.5 = {score:.6f}")
    print(f"  Ground truth entities : {len(gt):,}")
    print(f"  Predicted entities    : {len(preds):,}")

    # Quick breakdown
    n_singleton_correct = sum(
        1 for s1, ts in gt.items()
        if len(ts) == 0 and len(preds.get(s1, set())) == 0
    )
    n_singleton_wrong = sum(
        1 for s1, ts in gt.items()
        if len(ts) == 0 and len(preds.get(s1, set())) > 0
    )
    n_matched_correct = sum(
        1 for s1, ts in gt.items()
        if len(ts) > 0 and per_entity_f05(ts, preds.get(s1, set())) == 1.0
    )
    n_matched_zero = sum(
        1 for s1, ts in gt.items()
        if len(ts) > 0 and per_entity_f05(ts, preds.get(s1, set())) == 0.0
    )
    print(f"  Singletons correct    : {n_singleton_correct:,}")
    print(f"  Singletons wrong      : {n_singleton_wrong:,}")
    print(f"  Non-singleton perfect : {n_matched_correct:,}")
    print(f"  Non-singleton zero    : {n_matched_zero:,}")


if __name__ == "__main__":
    main()
