#!/usr/bin/env python3
"""
Phase 1 — Unit Tests for Scorer and TSV Parsing
=================================================
Tests cover:
  - per_entity_f05 edge cases (singleton, perfect, partial, no predictions)
  - macro_f05 aggregation
  - The worked example from the problem statement
  - TSV loading (round-trip write → load)
  - Validation split integrity

Run from student_resource/:  python -m pytest ml_challenge/04_test_scorer.py -v
                          or: python ml_challenge/04_test_scorer.py
"""

import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure ml_challenge/ is on the path so we can import 03_scorer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
scorer = import_module("03_scorer")


class TestPerEntityF05(unittest.TestCase):
    """Test per_entity_f05 for individual S1 entities."""

    def test_singleton_correct_empty(self):
        """True empty + predicted empty → 1.0"""
        self.assertEqual(scorer.per_entity_f05(set(), set()), 1.0)

    def test_singleton_wrong_nonempty(self):
        """True empty + predicted nonempty → 0.0"""
        self.assertEqual(scorer.per_entity_f05(set(), {"S2-001"}), 0.0)

    def test_nonempty_no_prediction(self):
        """True nonempty + predicted empty → 0.0"""
        self.assertEqual(scorer.per_entity_f05({"S2-001"}, set()), 0.0)

    def test_perfect_match_single(self):
        """Exact single match → 1.0"""
        self.assertEqual(scorer.per_entity_f05({"S2-001"}, {"S2-001"}), 1.0)

    def test_perfect_match_multiple(self):
        """Exact multiple match → 1.0"""
        true = {"S2-001", "S3-002", "S3-003"}
        self.assertEqual(scorer.per_entity_f05(true, true.copy()), 1.0)

    def test_problem_statement_example(self):
        """
        From README:
        Predicted: [S2-00047, S2-00193, S3-00812]
        Truth:     [S2-00047, S3-00812]
        Precision = 2/3, Recall = 2/2 = 1.0
        F_0.5 = (1.25 × 0.667 × 1.0) / (0.25 × 0.667 + 1.0) ≈ 0.714
        """
        true = {"S2-00047", "S3-00812"}
        pred = {"S2-00047", "S2-00193", "S3-00812"}
        score = scorer.per_entity_f05(true, pred)
        self.assertAlmostEqual(score, 0.714, places=2)

        # Exact calculation: (1.25 * 2/3 * 1) / (0.25 * 2/3 + 1) = 5/7
        self.assertAlmostEqual(score, 5.0 / 7.0, places=10)

    def test_one_correct_one_missed(self):
        """Predict one of two true matches."""
        true = {"S2-001", "S3-002"}
        pred = {"S2-001"}
        # P = 1/1 = 1, R = 1/2 = 0.5
        # F_0.5 = 1.25 * 1 * 0.5 / (0.25 * 1 + 0.5) = 0.625 / 0.75 = 5/6
        score = scorer.per_entity_f05(true, pred)
        self.assertAlmostEqual(score, 5.0 / 6.0, places=10)

    def test_all_wrong(self):
        """All predictions are false positives."""
        true = {"S2-001"}
        pred = {"S3-999"}
        # P = 0, R = 0 → F_0.5 = 0
        self.assertEqual(scorer.per_entity_f05(true, pred), 0.0)

    def test_superset_prediction(self):
        """Prediction is superset of truth (extra false positives)."""
        true = {"S2-001"}
        pred = {"S2-001", "S2-002", "S2-003"}
        # P = 1/3, R = 1/1 = 1
        # F_0.5 = 1.25 * (1/3) * 1 / (0.25 * (1/3) + 1) = (5/12) / (13/12) = 5/13
        score = scorer.per_entity_f05(true, pred)
        self.assertAlmostEqual(score, 5.0 / 13.0, places=10)


class TestMacroF05(unittest.TestCase):
    """Test macro_f05 aggregation."""

    def test_empty_ground_truth(self):
        self.assertEqual(scorer.macro_f05({}, {}), 0.0)

    def test_all_perfect(self):
        gt = {"A": {"S2-1"}, "B": set(), "C": {"S3-1", "S3-2"}}
        preds = {"A": {"S2-1"}, "B": set(), "C": {"S3-1", "S3-2"}}
        self.assertEqual(scorer.macro_f05(gt, preds), 1.0)

    def test_mixed(self):
        """Two entities: one perfect (1.0), one zero (0.0) → average = 0.5"""
        gt = {"A": {"S2-1"}, "B": {"S3-1"}}
        preds = {"A": {"S2-1"}}  # B missing → empty pred → 0.0
        score = scorer.macro_f05(gt, preds)
        self.assertAlmostEqual(score, 0.5, places=10)

    def test_missing_pred_treated_as_empty(self):
        """Missing key in predictions → treated as empty set."""
        gt = {"A": {"S2-1"}}
        preds = {}
        score = scorer.macro_f05(gt, preds)
        self.assertEqual(score, 0.0)

    def test_singletons_dominate(self):
        """
        3 singletons correct + 1 non-singleton wrong → 3/4 = 0.75
        """
        gt = {"A": set(), "B": set(), "C": set(), "D": {"S2-1"}}
        preds = {"A": set(), "B": set(), "C": set(), "D": set()}
        score = scorer.macro_f05(gt, preds)
        self.assertAlmostEqual(score, 0.75, places=10)


class TestTSVRoundTrip(unittest.TestCase):
    """Test TSV loading with round-trip write → load."""

    def test_load_standard(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tsv", delete=False, encoding="utf-8", newline=""
        ) as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.write("S1-001\tS2-010,S3-020\n")
            f.write("S1-002\t\n")                    # singleton
            f.write("S1-003\tS2-030\n")
            tmp_path = f.name

        try:
            result = scorer.load_tsv_as_dict(tmp_path)
            self.assertEqual(result["S1-001"], {"S2-010", "S3-020"})
            self.assertEqual(result["S1-002"], set())
            self.assertEqual(result["S1-003"], {"S2-030"})
            self.assertEqual(len(result), 3)
        finally:
            os.unlink(tmp_path)

    def test_load_empty_file_header_only(self):
        """File with only a header → empty dict."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tsv", delete=False, encoding="utf-8", newline=""
        ) as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            tmp_path = f.name

        try:
            result = scorer.load_tsv_as_dict(tmp_path)
            self.assertEqual(result, {})
        finally:
            os.unlink(tmp_path)


class TestSplitIntegrity(unittest.TestCase):
    """
    If splits/ already exist, verify they are non-overlapping and cover the GT.
    Skipped if splits/ does not exist yet (run 02_validation_split.py first).
    """

    def setUp(self):
        self.split_dir = Path(__file__).resolve().parent / "splits"
        if not (self.split_dir / "train_ids.txt").exists():
            self.skipTest("splits/ not yet created — run 02_validation_split.py first")

    def test_no_overlap(self):
        train_ids = set(
            (self.split_dir / "train_ids.txt").read_text(encoding="utf-8").split()
        )
        val_ids = set(
            (self.split_dir / "val_ids.txt").read_text(encoding="utf-8").split()
        )
        self.assertEqual(train_ids & val_ids, set(),
                         "Train and val ID sets must not overlap")

    def test_coverage(self):
        train_ids = set(
            (self.split_dir / "train_ids.txt").read_text(encoding="utf-8").split()
        )
        val_ids = set(
            (self.split_dir / "val_ids.txt").read_text(encoding="utf-8").split()
        )
        # Read full GT IDs
        gt_ids = set()
        base = Path(__file__).resolve().parent.parent
        gt_path = base / "dataset" / "train" / "train_ground_truth.tsv"
        if not gt_path.exists():
            self.skipTest("Ground truth file not found")
        with open(gt_path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for row in reader:
                gt_ids.add(row[0])

        combined = train_ids | val_ids
        self.assertEqual(combined, gt_ids,
                         "Train ∪ Val must equal all GT Source 1 IDs")


if __name__ == "__main__":
    unittest.main(verbosity=2)
