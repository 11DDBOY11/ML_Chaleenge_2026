#!/usr/bin/env python3
"""
Phase 1 — Ground Truth & Source 1 Analysis
===========================================
Streams training files to report:
  1. Source 1 records with 0, 1, 2, 3, 4+ matches
  2. Total matching pairs
  3. Mean / median / max matches per Source 1
  4. Country distribution in train_source1.tsv
  5. Missing business_name / business_address counts in train_source1.tsv

All operations use bounded memory (streaming / chunked reads).
Run from student_resource/:  python ml_challenge/01_ground_truth_analysis.py
"""

import csv
import os
import sys
from collections import Counter
from pathlib import Path

# Paths relative to student_resource/
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"
GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"
S1_PATH = TRAIN_DIR / "train_source1.tsv"


# ── helpers ──────────────────────────────────────────────────────────────────

def stream_ground_truth(path):
    """Yield (source1_entity_id, list_of_matched_ids) per row, streaming."""
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        assert header == ["source1_entity_id", "matched_entity_ids"], (
            f"Unexpected header: {header}"
        )
        for row in reader:
            s1_id = row[0]
            matched_str = row[1] if len(row) > 1 else ""
            if matched_str.strip():
                matched_ids = matched_str.split(",")
            else:
                matched_ids = []
            yield s1_id, matched_ids


def stream_source1_fields(path, chunksize=50_000):
    """Yield country, has_name, has_address per row, using chunked pandas."""
    import pandas as pd
    for chunk in pd.read_csv(path, sep="\t", chunksize=chunksize,
                             usecols=["country", "business_name", "business_address"],
                             dtype=str, keep_default_na=False):
        for _, row in chunk.iterrows():
            country = row["country"].strip() if row["country"].strip() else ""
            has_name = bool(row["business_name"].strip())
            has_addr = bool(row["business_address"].strip())
            yield country, has_name, has_addr


# ── main analysis ────────────────────────────────────────────────────────────

def analyse_ground_truth():
    print("=" * 70)
    print("GROUND TRUTH ANALYSIS  —  train_ground_truth.tsv")
    print("=" * 70)

    match_counts = Counter()          # number of matches → how many S1 IDs
    total_pairs = 0
    total_s1 = 0
    all_match_nums = []               # list of per-S1 match counts

    for s1_id, matched_ids in stream_ground_truth(GT_PATH):
        n = len(matched_ids)
        total_s1 += 1
        total_pairs += n
        all_match_nums.append(n)
        if n >= 4:
            match_counts["4+"] += 1
        else:
            match_counts[n] += 1

    # Sort the per-S1 match-count list for median
    all_match_nums.sort()
    n_total = len(all_match_nums)
    mean_matches = total_pairs / n_total if n_total else 0
    if n_total % 2 == 1:
        median_matches = all_match_nums[n_total // 2]
    else:
        mid = n_total // 2
        median_matches = (all_match_nums[mid - 1] + all_match_nums[mid]) / 2
    max_matches = all_match_nums[-1] if all_match_nums else 0

    print(f"\n  Total Source 1 rows  : {total_s1:>12,}")
    print(f"  Total matching pairs : {total_pairs:>12,}")
    print()
    print("  Match-count distribution:")
    for bucket in [0, 1, 2, 3, "4+"]:
        cnt = match_counts.get(bucket, 0)
        pct = 100 * cnt / total_s1 if total_s1 else 0
        print(f"    {str(bucket):>3s} matches : {cnt:>10,}  ({pct:5.1f}%)")
    print()
    print(f"  Mean   matches per S1 : {mean_matches:.4f}")
    print(f"  Median matches per S1 : {median_matches}")
    print(f"  Max    matches per S1 : {max_matches}")
    print()

    # Also report the tail of the distribution (5+ etc.)
    high_counts = Counter()
    for n in all_match_nums:
        if n >= 4:
            high_counts[n] += 1
    if high_counts:
        print("  Detailed 4+ breakdown:")
        for k in sorted(high_counts):
            print(f"    {k:>3d} matches : {high_counts[k]:>8,}")
    print()


def analyse_source1():
    print("=" * 70)
    print("SOURCE 1 FIELD ANALYSIS  —  train_source1.tsv")
    print("=" * 70)

    country_counts = Counter()
    missing_name = 0
    missing_addr = 0
    total = 0

    for country, has_name, has_addr in stream_source1_fields(S1_PATH):
        total += 1
        if country:
            country_counts[country] += 1
        else:
            country_counts["(empty)"] += 1
        if not has_name:
            missing_name += 1
        if not has_addr:
            missing_addr += 1

    print(f"\n  Total Source 1 records : {total:>10,}")
    print(f"  Missing business_name : {missing_name:>10,}  ({100*missing_name/total:.2f}%)")
    print(f"  Missing business_addr : {missing_addr:>10,}  ({100*missing_addr/total:.2f}%)")
    print()
    print("  Country distribution:")
    for country, cnt in country_counts.most_common():
        pct = 100 * cnt / total
        print(f"    {country:>12s} : {cnt:>10,}  ({pct:5.1f}%)")
    print()


if __name__ == "__main__":
    if not GT_PATH.exists():
        sys.exit(f"ERROR: {GT_PATH} not found. Run from student_resource/.")
    if not S1_PATH.exists():
        sys.exit(f"ERROR: {S1_PATH} not found. Run from student_resource/.")

    analyse_ground_truth()
    analyse_source1()
    print("Done.")
