#!/usr/bin/env python3
"""
Phase 1 — Deterministic Train / Validation Split
==================================================
Splits Source 1 entity IDs 80/20 by hashing, so every pair for one S1 ID
stays together.  Produces:

  ml_challenge/splits/train_ids.txt
  ml_challenge/splits/val_ids.txt
  ml_challenge/splits/train_ground_truth.tsv
  ml_challenge/splits/val_ground_truth.tsv
  ml_challenge/splits/split_summary.txt

Deterministic: uses hashlib SHA-256, no random seed needed.
Run from student_resource/:  python ml_challenge/02_validation_split.py
"""

import csv
import hashlib
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"
GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"
OUT_DIR = Path(__file__).resolve().parent / "splits"

VAL_FRACTION = 0.20  # 20% for validation


def deterministic_bucket(entity_id, n_buckets=100):
    """Hash entity_id to an integer bucket in [0, n_buckets)."""
    h = hashlib.sha256(entity_id.encode("utf-8")).hexdigest()
    return int(h, 16) % n_buckets


def main():
    if not GT_PATH.exists():
        sys.exit(f"ERROR: {GT_PATH} not found. Run from student_resource/.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Pass 1: collect all unique S1 IDs and decide split ────────────────
    val_threshold = int(VAL_FRACTION * 100)  # 20 → buckets 0..19 are val

    train_ids = set()
    val_ids = set()

    with open(GT_PATH, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            s1_id = row[0]
            bucket = deterministic_bucket(s1_id)
            if bucket < val_threshold:
                val_ids.add(s1_id)
            else:
                train_ids.add(s1_id)

    total = len(train_ids) + len(val_ids)
    print(f"Total S1 IDs       : {total:>10,}")
    print(f"Train IDs          : {len(train_ids):>10,}  ({100*len(train_ids)/total:.1f}%)")
    print(f"Validation IDs     : {len(val_ids):>10,}  ({100*len(val_ids)/total:.1f}%)")

    # ── Save ID lists ─────────────────────────────────────────────────────
    for name, id_set in [("train_ids.txt", train_ids), ("val_ids.txt", val_ids)]:
        with open(OUT_DIR / name, "w", encoding="utf-8") as f:
            for sid in sorted(id_set):
                f.write(sid + "\n")

    # ── Pass 2: split ground truth rows ───────────────────────────────────
    train_pairs = 0
    val_pairs = 0
    train_singletons = 0
    val_singletons = 0

    with open(GT_PATH, encoding="utf-8", newline="") as fin, \
         open(OUT_DIR / "train_ground_truth.tsv", "w", encoding="utf-8", newline="") as f_train, \
         open(OUT_DIR / "val_ground_truth.tsv", "w", encoding="utf-8", newline="") as f_val:

        reader = csv.reader(fin, delimiter="\t")
        header_row = next(reader)
        header_line = "\t".join(header_row) + "\n"
        f_train.write(header_line)
        f_val.write(header_line)

        for row in reader:
            s1_id = row[0]
            matched_str = row[1] if len(row) > 1 else ""
            n_matches = len(matched_str.split(",")) if matched_str.strip() else 0
            line = "\t".join(row) + "\n"

            if s1_id in val_ids:
                f_val.write(line)
                val_pairs += n_matches
                if n_matches == 0:
                    val_singletons += 1
            else:
                f_train.write(line)
                train_pairs += n_matches
                if n_matches == 0:
                    train_singletons += 1

    print(f"\nTrain pairs        : {train_pairs:>10,}")
    print(f"Train singletons   : {train_singletons:>10,}")
    print(f"Val   pairs        : {val_pairs:>10,}")
    print(f"Val   singletons   : {val_singletons:>10,}")

    # ── Save summary ──────────────────────────────────────────────────────
    summary = (
        f"Deterministic split (SHA-256 hash, val buckets 0..{val_threshold-1} of 100)\n"
        f"Total S1 IDs       : {total:,}\n"
        f"Train IDs          : {len(train_ids):,} ({100*len(train_ids)/total:.1f}%)\n"
        f"Val   IDs          : {len(val_ids):,} ({100*len(val_ids)/total:.1f}%)\n"
        f"Train pairs        : {train_pairs:,}\n"
        f"Train singletons   : {train_singletons:,}\n"
        f"Val   pairs        : {val_pairs:,}\n"
        f"Val   singletons   : {val_singletons:,}\n"
    )
    with open(OUT_DIR / "split_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"\nSplit files written to {OUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
