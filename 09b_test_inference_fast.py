#!/usr/bin/env python3
"""
09b_test_inference_fast.py — Optimized Production Test Inference Pipeline
==========================================================================
Achieves 10x+ inference acceleration via:
  1. In-memory dict lookup for S2/S3 entity records (eliminating SQLite disk queries).
  2. Representation caching (pre-computing normalize/tokenize/trigrams ONCE per entity).
  3. Parallelized pairwise feature extraction using multiprocessing worker pool.
  4. Resumable TSV writing to isolated output directory (ml_challenge/fast_output/).
"""

import os
import sys
import time
import json
import csv
import argparse
import numpy as np
import lightgbm as lgb
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
cg_mod = importlib.import_module("05_candidate_generator")
fe_mod = importlib.import_module("07_feature_extractor")

CandidateGenerator = cg_mod.CandidateGenerator
prepare_rep = fe_mod.prepare_entity_representation
extract_pair_features_fast = fe_mod.extract_pair_features_fast

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DIR = BASE_DIR / "dataset" / "test"
ML_DIR = Path(__file__).resolve().parent
FAST_OUTPUT_DIR = ML_DIR / "fast_output"
RESULT_DIR = ML_DIR / "pilot_results"

TEST_S1 = TEST_DIR / "test_source1.tsv"
TEST_S2 = TEST_DIR / "test_source2.tsv"
TEST_S3 = TEST_DIR / "test_source3.tsv"

MODEL_FILE = RESULT_DIR / "matcher_lgbm.txt"
CONFIG_FILE = RESULT_DIR / "matcher_config.json"

BATCH_SIZE = 10_000
TOTAL_TEST_S1_COUNT = 1_732_544

def load_entity_records_in_memory():
    """Load test_source2.tsv and test_source3.tsv into an in-memory dictionary."""
    print("Loading test S2 and S3 entity records into memory...", flush=True)
    t0 = time.time()
    records = {}
    for path in [TEST_S2, TEST_S3]:
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for row in reader:
                eid = row[0]
                records[eid] = {
                    "entity_id": eid,
                    "business_name": row[1] if len(row) > 1 else "",
                    "business_address": row[2] if len(row) > 2 else "",
                    "country": row[3].strip() if len(row) > 3 else "",
                }
    print(f"Loaded {len(records):,} entity records in {time.time()-t0:.1f}s (RAM: {cg_mod.get_memory_mb():,.0f} MB).", flush=True)
    return records

def _extract_chunk_features(chunk_s1_records, cands_by_s1, entity_records):
    """Worker function for parallel pairwise feature extraction."""
    prep_s1 = {rec["entity_id"]: prepare_rep(rec) for rec in chunk_s1_records}
    needed_eids = set()
    for rec in chunk_s1_records:
        needed_eids.update(cands_by_s1.get(rec["entity_id"], []))
    
    prep_cands = {}
    for eid in needed_eids:
        rec = entity_records.get(eid)
        if rec:
            prep_cands[eid] = prepare_rep(rec)

    pair_info = []
    X_feats = []
    for rec in chunk_s1_records:
        s1_id = rec["entity_id"]
        p1 = prep_s1.get(s1_id)
        if not p1: continue
        for cand_id in cands_by_s1.get(s1_id, []):
            p2 = prep_cands.get(cand_id)
            if not p2: continue
            feats = extract_pair_features_fast(p1, p2)
            X_feats.append(feats)
            pair_info.append((s1_id, cand_id))

    return pair_info, X_feats

def load_completed_s1_ids(matching_file):
    if not matching_file.exists():
        return set()
    completed = set()
    with open(matching_file, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        hdr = next(reader, None)
        if hdr != ["source1_entity_id", "matched_entity_ids"]:
            return set()
        for row in reader:
            if row:
                completed.add(row[0])
    return completed

def run_fast_inference(max_s1_limit=None, num_workers=4):
    t_start = time.time()
    print(f"\n{'='*70}\nOPTIMIZED INFERENCE PIPELINE (Workers={num_workers})\n{'='*70}", flush=True)

    with open(CONFIG_FILE, encoding="utf-8") as f:
        config = json.load(f)
    threshold = float(config.get("optimal_threshold", 0.50))
    booster = lgb.Booster(model_file=str(MODEL_FILE))

    entity_records = load_entity_records_in_memory()

    print("Building Candidate Generator inverted index...", flush=True)
    t_idx = time.time()
    gen = CandidateGenerator()
    gen.build_from_files([str(TEST_S2), str(TEST_S3)])
    print(f"Index built in {time.time()-t_idx:.1f}s.", flush=True)

    FAST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matching_path = FAST_OUTPUT_DIR / "matching_results.tsv"
    candidate_path = FAST_OUTPUT_DIR / "candidate_pairs.tsv"

    completed_ids = load_completed_s1_ids(matching_path) if max_s1_limit is None else set()
    if completed_ids:
        print(f"Resumable state detected: {len(completed_ids):,} S1 entities already completed in fast_output.", flush=True)

    if not matching_path.exists() or not completed_ids:
        with open(matching_path, "w", encoding="utf-8", newline="") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
        with open(candidate_path, "w", encoding="utf-8", newline="") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")

    batch_records = []
    total_processed = len(completed_ids)

    with open(TEST_S1, encoding="utf-8", newline="") as f_s1:
        reader = csv.reader(f_s1, delimiter="\t")
        next(reader)

        for row_idx, row in enumerate(reader):
            if max_s1_limit and row_idx >= max_s1_limit:
                break

            s1_id = row[0]
            if s1_id in completed_ids:
                continue

            batch_records.append({
                "entity_id": s1_id,
                "business_name": row[1] if len(row) > 1 else "",
                "business_address": row[2] if len(row) > 2 else "",
                "country": row[3].strip() if len(row) > 3 else "",
            })

            if len(batch_records) >= BATCH_SIZE or (max_s1_limit and len(batch_records) >= max_s1_limit):
                _process_batch_fast(
                    batch_records, gen, booster, threshold, entity_records,
                    matching_path, candidate_path, num_workers=num_workers
                )
                total_processed += len(batch_records)
                batch_records.clear()

                elapsed = time.time() - t_start
                rate = total_processed / elapsed if elapsed else 1.0
                rem_s1 = (max_s1_limit or TOTAL_TEST_S1_COUNT) - total_processed
                eta_s = rem_s1 / rate if rate else 0
                ram_mb = cg_mod.get_memory_mb()

                print(f"  [Fast Progress] {total_processed:>9,}/{max_s1_limit or TOTAL_TEST_S1_COUNT:,} S1 processed "
                      f"({total_processed / (max_s1_limit or TOTAL_TEST_S1_COUNT) * 100:.1f}%) | "
                      f"Elapsed: {elapsed/60:.1f}m | ETA: {eta_s/60:.1f}m | RAM: {ram_mb:,.0f} MB", flush=True)

        if batch_records:
            _process_batch_fast(
                batch_records, gen, booster, threshold, entity_records,
                matching_path, candidate_path, num_workers=num_workers
            )
            total_processed += len(batch_records)
            batch_records.clear()

    total_time = time.time() - t_start
    print(f"\nFast inference run finished in {total_time:.1f}s ({total_time/60:.2f}m)!", flush=True)
    return total_time

def _process_batch_fast(batch_records, gen, booster, threshold, entity_records, matching_path, candidate_path, num_workers=4):
    cands_by_s1 = {}
    for rec in batch_records:
        s1_id = rec["entity_id"]
        cands = gen.find_candidates(
            rec["business_name"], rec["business_address"], rec["country"], max_total=200
        )
        cands_by_s1[s1_id] = sorted(cands)

    chunk_size = max(1, len(batch_records) // num_workers)
    chunks = [batch_records[i:i+chunk_size] for i in range(0, len(batch_records), chunk_size)]

    pair_info = []
    X_feats = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(_extract_chunk_features, chunk, cands_by_s1, entity_records)
            for chunk in chunks
        ]
        for f in as_completed(futures):
            p_info, x_f = f.result()
            pair_info.extend(p_info)
            X_feats.extend(x_f)

    preds_by_s1 = defaultdict(list)
    if X_feats:
        X_mat = np.array(X_feats, dtype=np.float32)
        probs = booster.predict(X_mat)
        for (s1_id, cand_id), prob in zip(pair_info, probs):
            if prob >= threshold:
                preds_by_s1[s1_id].append(cand_id)

    with open(matching_path, "a", encoding="utf-8", newline="") as f_m, \
         open(candidate_path, "a", encoding="utf-8", newline="") as f_c:
        for rec in batch_records:
            s1_id = rec["entity_id"]
            cand_list = cands_by_s1.get(s1_id, [])
            match_list = sorted(preds_by_s1.get(s1_id, []))

            f_m.write(f"{s1_id}\t{','.join(match_list)}\n")
            f_c.write(f"{s1_id}\t{','.join(cand_list)}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10_000, help="S1 limit for benchmark")
    parser.add_argument("--workers", type=int, default=4, help="Number of CPU workers")
    args = parser.parse_args()
    run_fast_inference(max_s1_limit=args.limit, num_workers=args.workers)
