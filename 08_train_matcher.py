#!/usr/bin/env python3
"""
08_train_matcher.py — CPU LightGBM Pair Matcher Training & Validation Pipeline
================================================================================
Trains a LightGBM pairwise classifier on Phase 2 candidates and tunes decision threshold.

Requirements enforced:
  1. Uses bounded reproducible pilot candidates (7,000 S1 for train, 3,000 S1 for val).
  2. Keeps validation labels strictly out of training.
  3. No full-test inference.
  4. Allows multiple matches per S1 entity.
  5. Computes Macro F0.5 score via 03_scorer.py, optimal per-pair threshold,
     singleton performance, total scored pairs, training time, and process memory.
  6. Saves trained Booster model and threshold configuration for test inference.
"""

import csv
import json
import os
import sys
import time
import numpy as np
import lightgbm as lgb
from pathlib import Path
from collections import defaultdict

# Add ml_challenge to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
cg_mod = importlib.import_module("05_candidate_generator")
scorer_mod = importlib.import_module("03_scorer")
fe_mod = importlib.import_module("07_feature_extractor")

prepare_rep = fe_mod.prepare_entity_representation
extract_pair_features_fast = fe_mod.extract_pair_features_fast
FEATURE_NAMES = fe_mod.FEATURE_NAMES
macro_f05 = scorer_mod.macro_f05

# Paths
BASE = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE / "dataset" / "train"
SPLIT_DIR = Path(__file__).resolve().parent / "splits"
RESULT_DIR = Path(__file__).resolve().parent / "pilot_results"

S2_PATH = TRAIN_DIR / "train_source2.tsv"
S3_PATH = TRAIN_DIR / "train_source3.tsv"
S1_PATH = TRAIN_DIR / "train_source1.tsv"
VAL_IDS = SPLIT_DIR / "val_ids.txt"
VAL_GT  = SPLIT_DIR / "val_ground_truth.tsv"
CAND_TSV = RESULT_DIR / "pilot_candidates_B_cap200.tsv"

def main():
    t_start = time.time()
    print("=" * 70)
    print("PHASE 3 — CPU LIGHTGBM PAIR MATCHER PIPELINE")
    print("=" * 70)

    # 1. Load pilot S1 IDs & ground truth
    with open(VAL_IDS, encoding="utf-8") as f:
        all_val_ids = [line.strip() for line in f if line.strip()]
    all_val_ids.sort()
    pilot_ids = all_val_ids[:10_000]
    pilot_id_set = set(pilot_ids)

    gt = {}
    with open(VAL_GT, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            s1 = row[0]
            if s1 in pilot_id_set:
                matched = row[1] if len(row) > 1 else ""
                gt[s1] = set(matched.split(",")) if matched.strip() else set()

    # 2. Split 10,000 S1 into Train (7,000) and Val (3,000)
    train_s1_set = set(pilot_ids[:7_000])
    val_s1_set   = set(pilot_ids[7_000:])
    val_gt       = {s1: gt[s1] for s1 in val_s1_set if s1 in gt}

    print(f"Dataset Split: {len(train_s1_set):,} Train S1 IDs  |  {len(val_s1_set):,} Val S1 IDs (held-out)")

    # 3. Load generated candidates from pilot TSV
    candidates_by_s1 = {}
    needed_cand_eids = set()
    with open(CAND_TSV, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader) # header
        for row in reader:
            s1_id = row[0]
            if s1_id in pilot_id_set:
                cands = row[1].split(",") if len(row) > 1 and row[1].strip() else []
                candidates_by_s1[s1_id] = cands
                needed_cand_eids.update(cands)

    print(f"Candidates Loaded: {len(candidates_by_s1):,} S1 entities querying {len(needed_cand_eids):,} distinct S2/S3 candidate entities.")

    # 4. Fast Positional Streaming for Entity Details
    print("\nStreaming S1 entity records...")
    raw_s1 = {}
    with open(S1_PATH, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader) # header
        for row in reader:
            if row[0] in pilot_id_set:
                raw_s1[row[0]] = {
                    "entity_id": row[0],
                    "business_name": row[1] if len(row) > 1 else "",
                    "business_address": row[2] if len(row) > 2 else "",
                    "country": row[3].strip() if len(row) > 3 else "",
                }

    print(f"Streaming candidate records from S2 and S3 for {len(needed_cand_eids):,} target entities...")
    raw_cands = {}
    t0 = time.time()
    for path in [S2_PATH, S3_PATH]:
        file_label = path.name
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader) # header
            for i, row in enumerate(reader):
                if (i + 1) % 2_000_000 == 0:
                    print(f"    [{file_label}] {i+1:>9,} records scanned ({time.time()-t0:.1f}s, {len(raw_cands):,}/{len(needed_cand_eids):,} candidates found)...")
                eid = row[0]
                if eid in needed_cand_eids:
                    raw_cands[eid] = {
                        "entity_id": eid,
                        "business_name": row[1] if len(row) > 1 else "",
                        "business_address": row[2] if len(row) > 2 else "",
                        "country": row[3].strip() if len(row) > 3 else "",
                    }
                    if len(raw_cands) == len(needed_cand_eids):
                        break
        if len(raw_cands) == len(needed_cand_eids):
            break

    print(f"Loaded all candidate details in {time.time()-t0:.1f}s!")

    # Strict Data Completeness Assertions
    assert len(gt) == 10_000, f"Ground truth missing for {10_000 - len(gt)} pilot IDs"
    assert set(candidates_by_s1) == pilot_id_set, "Candidate file does not match pilot IDs"
    assert set(raw_s1) == pilot_id_set, "Some pilot Source 1 records were not found"
    assert set(raw_cands) == needed_cand_eids, "Some candidate records were not found"

    # 5. Pre-compute Entity Representations ONCE
    print("\nPre-computing entity-level representations...")
    t_prep = time.time()
    prep_s1 = {eid: prepare_rep(rec) for eid, rec in raw_s1.items()}
    prep_cands = {eid: prepare_rep(rec) for eid, rec in raw_cands.items()}
    print(f"Pre-computed representations for {len(prep_s1)+len(prep_cands):,} entities in {time.time()-t_prep:.1f}s!")

    # 6. Feature Extraction
    print("\nExtracting 15 pairwise features for train and val pairs...")
    t_feat = time.time()
    X_train, y_train = [], []
    X_val, y_val_info = [], []  # (s1_id, cand_id, true_label)

    n_train_pairs = 0
    for s1_id in sorted(train_s1_set):
        p1 = prep_s1.get(s1_id)
        if not p1:
            continue
        cands = candidates_by_s1.get(s1_id, [])
        true_set = gt.get(s1_id, set())

        for cand_id in cands:
            p2 = prep_cands.get(cand_id)
            if not p2:
                continue
            feats = extract_pair_features_fast(p1, p2)
            label = 1 if cand_id in true_set else 0
            X_train.append(feats)
            y_train.append(label)
            n_train_pairs += 1

    n_val_pairs = 0
    for s1_id in sorted(val_s1_set):
        p1 = prep_s1.get(s1_id)
        if not p1:
            continue
        cands = candidates_by_s1.get(s1_id, [])
        true_set = gt.get(s1_id, set())

        for cand_id in cands:
            p2 = prep_cands.get(cand_id)
            if not p2:
                continue
            feats = extract_pair_features_fast(p1, p2)
            label = 1 if cand_id in true_set else 0
            X_val.append(feats)
            y_val_info.append((s1_id, cand_id, label))
            n_val_pairs += 1

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    X_val   = np.array(X_val, dtype=np.float32)

    total_pairs = n_train_pairs + n_val_pairs
    print(f"Feature Extraction Completed in {time.time()-t_feat:.1f}s!")
    print(f"  - Train Pairs: {n_train_pairs:,} ({y_train.sum():,} positive, {len(y_train) - y_train.sum():,} negative)")
    print(f"  - Val Pairs:   {n_val_pairs:,}")

    # 7. Train CPU LightGBM Model
    print("\nTraining LightGBM Pair Classifier...")
    t_train_start = time.time()
    
    clf = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.08,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(X_train, y_train)
    t_train_s = time.time() - t_train_start
    print(f"Model Training Completed in {t_train_s:.2f}s!")

    # 8. Validation Evaluation & Threshold Search
    print("\nEvaluating probability thresholds on held-out 3,000 S1 validation entities...")
    val_probs = clf.predict_proba(X_val)[:, 1]

    val_preds_by_s1 = defaultdict(list)
    for (s1_id, cand_id, true_lbl), prob in zip(y_val_info, val_probs):
        val_preds_by_s1[s1_id].append((cand_id, prob))

    thresholds = np.linspace(0.05, 0.95, 19)
    best_macro_f05 = -1.0
    best_thresh = 0.5
    best_singleton_f05 = 0.0

    print("\n" + "=" * 60)
    print(f"{'Threshold':>10} | {'Macro F0.5':>12} | {'Singleton F0.5':>15}")
    print("-" * 60)

    for tau in thresholds:
        preds_dict = {}
        for s1_id in val_s1_set:
            pair_list = val_preds_by_s1.get(s1_id, [])
            selected = {cand for cand, prob in pair_list if prob >= tau}
            preds_dict[s1_id] = selected

        score = macro_f05(val_gt, preds_dict)

        # Singleton score
        singleton_gt = {s1: val_gt[s1] for s1 in val_s1_set if not val_gt.get(s1)}
        singleton_preds = {s1: preds_dict.get(s1, set()) for s1 in singleton_gt}
        sing_score = macro_f05(singleton_gt, singleton_preds)

        print(f"{tau:>10.2f} | {score:>12.4f} | {sing_score:>15.4f}")

        if score > best_macro_f05:
            best_macro_f05 = score
            best_thresh = tau
            best_singleton_f05 = sing_score

    print("-" * 70)
    proc_memory_mb = cg_mod.get_memory_mb()

    # 9. Save Model and Configuration Artifacts
    model_path = RESULT_DIR / "matcher_lgbm.txt"
    clf.booster_.save_model(str(model_path))
    print(f"\nSaved trained LightGBM model to {model_path}")

    config_path = RESULT_DIR / "matcher_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "optimal_threshold": float(best_thresh),
            "tuned_val_macro_f05": float(best_macro_f05),
            "singleton_f05": float(best_singleton_f05),
            "feature_names": FEATURE_NAMES,
            "candidate_config": "B_cap200",
            "train_pairs_count": n_train_pairs,
            "val_pairs_count": n_val_pairs,
        }, f, indent=2)
    print(f"Saved matcher configuration to {config_path}")

    # 10. Print Final Summary
    print("\n" + "=" * 70)
    print("PHASE 3 — MODEL VALIDATION RESULTS")
    print("=" * 70)
    print(f"  Tuned Validation Macro F0.5          : {best_macro_f05:.4f}")
    print(f"  Optimal Per-Pair Probability Threshold: {best_thresh:.2f}")
    print(f"  Singleton Performance (F0.5)        : {best_singleton_f05:.4f}")
    print(f"  Total Scored Candidate Pairs         : {total_pairs:,}")
    print(f"    - Train Pairs (7K S1 entities)    : {n_train_pairs:,} ({y_train.sum():,} pos, {len(y_train)-y_train.sum():,} neg)")
    print(f"    - Val Pairs (3K S1 entities)      : {n_val_pairs:,}")
    print(f"  Model Training Time                  : {t_train_s:.2f}s")
    print(f"  Total Execution Time                 : {time.time()-t_start:.1f}s")
    print(f"  Process Memory at End                : {proc_memory_mb:,.0f} MB")
    print("=" * 70)

    print("\nTop 10 Feature Importances:")
    imp = clf.feature_importances_
    sorted_idx = np.argsort(imp)[::-1]
    for rank, idx in enumerate(sorted_idx[:10], 1):
        print(f"  {rank:>2d}. {FEATURE_NAMES[idx]:<22s} : {imp[idx]:>6d}")

if __name__ == "__main__":
    main()
