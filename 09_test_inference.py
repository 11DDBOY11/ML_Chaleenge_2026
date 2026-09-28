#!/usr/bin/env python3
"""
09_test_inference.py — Production Test Inference Pipeline (Amazon ML Challenge 2026)
===================================================================================

Processes 1,732,544 test Source 1 entities against 10.3M test Source 2/3 entities.
Enforces:
  1. Bounded memory execution (< 16 GB RAM) using WAL-mode SQLite disk-backed candidate lookup & batched S1 processing.
  2. Incremental, resumable TSV writing.
  3. Strict subset constraint: every predicted match in matching_results.tsv is inside candidate_pairs.tsv.
  4. Supports singletons, single matches, and multiple matches per S1 entity.
  5. Covers open country labels (US, India, France, etc.).
  6. Includes 1,000-entity smoke test validation before full execution.
"""

import csv
import json
import os
import sys
import time
import sqlite3
import argparse
import numpy as np
import lightgbm as lgb
from pathlib import Path
from collections import defaultdict

# Add ml_challenge to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
cg_mod = importlib.import_module("05_candidate_generator")
fe_mod = importlib.import_module("07_feature_extractor")

CandidateGenerator = cg_mod.CandidateGenerator
prepare_rep = fe_mod.prepare_entity_representation
extract_pair_features_fast = fe_mod.extract_pair_features_fast
FEATURE_NAMES = fe_mod.FEATURE_NAMES

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"
TEST_DIR = BASE_DIR / "dataset" / "test"
ML_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ML_DIR / "output"
RESULT_DIR = ML_DIR / "pilot_results"

TEST_S1 = TEST_DIR / "test_source1.tsv"
TEST_S2 = TEST_DIR / "test_source2.tsv"
TEST_S3 = TEST_DIR / "test_source3.tsv"

MODEL_FILE = RESULT_DIR / "matcher_lgbm.txt"
CONFIG_FILE = RESULT_DIR / "matcher_config.json"
DB_FILE = OUTPUT_DIR / "test_entities.db"

BATCH_SIZE = 25_000
TOTAL_TEST_S1_COUNT = 1_732_544

def build_sqlite_db():
    """Build a fast WAL-mode SQLite disk index for test Source 2 and Source 3 entity details."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if DB_FILE.exists():
        try:
            conn = sqlite3.connect(str(DB_FILE), timeout=60.0)
            cursor = conn.cursor()
            cursor.execute("SELECT count(*) FROM entities")
            cnt = cursor.fetchone()[0]
            conn.close()
            if cnt > 9_000_000:
                print(f"SQLite disk database verified ({cnt:,} records in {DB_FILE.name}).", flush=True)
                return
        except Exception as e:
            print(f"Rebuilding SQLite database due to check error: {e}", flush=True)

    print(f"\nBuilding SQLite disk database {DB_FILE.name} from test S2 and S3...", flush=True)
    t0 = time.time()

    conn = sqlite3.connect(str(DB_FILE), timeout=60.0)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("DROP TABLE IF EXISTS entities")
    cursor.execute("""
        CREATE TABLE entities (
            entity_id TEXT PRIMARY KEY,
            business_name TEXT,
            business_address TEXT,
            country TEXT
        )
    """)
    conn.commit()

    batch = []
    total_added = 0
    for path in [TEST_S2, TEST_S3]:
        print(f"  Indexing {path.name} into SQLite...", flush=True)
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader) # header
            for row in reader:
                eid = row[0]
                name = row[1] if len(row) > 1 else ""
                addr = row[2] if len(row) > 2 else ""
                ctry = row[3].strip() if len(row) > 3 else ""
                batch.append((eid, name, addr, ctry))
                if len(batch) >= 100_000:
                    cursor.executemany("INSERT OR IGNORE INTO entities VALUES (?, ?, ?, ?)", batch)
                    conn.commit()
                    total_added += len(batch)
                    batch.clear()

    if batch:
        cursor.executemany("INSERT OR IGNORE INTO entities VALUES (?, ?, ?, ?)", batch)
        conn.commit()
        total_added += len(batch)

    print("  Creating primary key index on entity_id...", flush=True)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_eid ON entities(entity_id)")
    conn.commit()
    conn.close()
    print(f"SQLite DB built successfully ({total_added:,} records in {time.time()-t0:.1f}s)!", flush=True)

def fetch_candidate_records_sqlite(needed_eids):
    """Fetch candidate entity records from SQLite DB in fast chunked batches."""
    if not needed_eids:
        return {}

    conn = sqlite3.connect(str(DB_FILE), timeout=60.0)
    cursor = conn.cursor()
    cand_records = {}
    eid_list = list(needed_eids)
    chunk_size = 5_000

    for i in range(0, len(eid_list), chunk_size):
        chunk = eid_list[i:i+chunk_size]
        placeholders = ",".join(["?"] * len(chunk))
        query = f"SELECT entity_id, business_name, business_address, country FROM entities WHERE entity_id IN ({placeholders})"
        cursor.execute(query, chunk)
        for row in cursor.fetchall():
            cand_records[row[0]] = {
                "entity_id": row[0],
                "business_name": row[1],
                "business_address": row[2],
                "country": row[3],
            }
    conn.close()
    return cand_records

def load_completed_s1_ids(matching_file):
    """Return set of S1 IDs already written to matching_file for resumable execution."""
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

def run_inference(smoke_test=False, existing_gen=None, existing_booster=None, existing_thresh=None):
    t_start = time.time()

    # 1. Load config and LightGBM model
    if existing_booster is None or existing_thresh is None:
        assert CONFIG_FILE.exists(), f"Config file missing: {CONFIG_FILE}"
        assert MODEL_FILE.exists(), f"Model file missing: {MODEL_FILE}"

        with open(CONFIG_FILE, encoding="utf-8") as f:
            config = json.load(f)

        optimal_thresh = float(config.get("optimal_threshold", 0.50))
        print(f"Loaded config: Optimal Threshold = {optimal_thresh:.2f}, Pre-Model Cap = B_cap200", flush=True)

        booster = lgb.Booster(model_file=str(MODEL_FILE))
        print("LightGBM Booster model loaded successfully.", flush=True)
    else:
        booster = existing_booster
        optimal_thresh = existing_thresh

    # 2. Build SQLite disk database for fast candidate lookup
    build_sqlite_db()

    # 3. Build candidate generator index over test S2 and test S3 if not provided
    if existing_gen is None:
        print("\nBuilding candidate generator inverted indexes for TEST S2 & S3...", flush=True)
        t_idx = time.time()
        gen = CandidateGenerator()
        gen.build_from_files([str(TEST_S2), str(TEST_S3)])
        print(f"Test candidate indexes built in {time.time()-t_idx:.1f}s (RAM: {cg_mod.get_memory_mb():,.0f} MB).", flush=True)
    else:
        gen = existing_gen
        print("Reusing in-memory candidate generator index.", flush=True)

    # 4. Determine output paths and mode
    if smoke_test:
        matching_path = OUTPUT_DIR / "smoke_matching_results.tsv"
        candidate_path = OUTPUT_DIR / "smoke_candidate_pairs.tsv"
        max_s1_limit = 1_000
        print(f"\n{'='*70}\nRUNNING 1,000-ENTITY SMOKE TEST\n{'='*70}", flush=True)
    else:
        matching_path = OUTPUT_DIR / "matching_results.tsv"
        candidate_path = OUTPUT_DIR / "candidate_pairs.tsv"
        max_s1_limit = None
        print(f"\n{'='*70}\nRUNNING FULL TEST INFERENCE (1,732,544 S1 Entities)\n{'='*70}", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check for resumable execution
    completed_ids = load_completed_s1_ids(matching_path) if not smoke_test else set()
    if completed_ids:
        print(f"Resumable state detected: {len(completed_ids):,} S1 entities already completed.", flush=True)

    # Initialize file headers if starting fresh
    if not matching_path.exists() or smoke_test or not completed_ids:
        with open(matching_path, "w", encoding="utf-8", newline="") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
        with open(candidate_path, "w", encoding="utf-8", newline="") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")

    # 5. Process test_source1.tsv in streaming batches
    batch_records = []
    total_processed = len(completed_ids)

    with open(TEST_S1, encoding="utf-8", newline="") as f_s1:
        reader = csv.reader(f_s1, delimiter="\t")
        next(reader) # skip header

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
                _process_batch(
                    batch_records, gen, booster, optimal_thresh,
                    matching_path, candidate_path
                )
                total_processed += len(batch_records)
                batch_records.clear()

                # Progress report
                elapsed = time.time() - t_start
                rate = total_processed / elapsed if elapsed else 1.0
                rem_s1 = (max_s1_limit or TOTAL_TEST_S1_COUNT) - total_processed
                eta_s = rem_s1 / rate if rate else 0
                ram_mb = cg_mod.get_memory_mb()
                m_size_mb = matching_path.stat().st_size / (1024**2)
                c_size_mb = candidate_path.stat().st_size / (1024**2)

                print(f"  [Progress] {total_processed:>9,}/{max_s1_limit or TOTAL_TEST_S1_COUNT:,} S1 processed "
                      f"({total_processed / (max_s1_limit or TOTAL_TEST_S1_COUNT) * 100:.1f}%) | "
                      f"Elapsed: {elapsed/60:.1f}m | ETA: {eta_s/60:.1f}m | RAM: {ram_mb:,.0f} MB | "
                      f"matching.tsv: {m_size_mb:.1f}MB | candidate.tsv: {c_size_mb:.1f}MB", flush=True)

        # Process final partial batch
        if batch_records:
            _process_batch(
                batch_records, gen, booster, optimal_thresh,
                matching_path, candidate_path
            )
            total_processed += len(batch_records)
            batch_records.clear()

    total_time = time.time() - t_start
    print(f"\nInference completed in {total_time/60:.2f} minutes!", flush=True)

    # If smoke test, perform strict assertion checks
    if smoke_test:
        verify_smoke_test(matching_path, candidate_path)

    return gen, booster, optimal_thresh

def _process_batch(batch_records, gen, booster, threshold, matching_path, candidate_path):
    """Process a batch of S1 records: candidates -> features -> model predict -> append TSVs."""
    cands_by_s1 = {}
    needed_eids = set()

    for rec in batch_records:
        s1_id = rec["entity_id"]
        cands = gen.find_candidates(
            rec["business_name"],
            rec["business_address"],
            rec["country"],
            max_total=200, # B_cap200
        )
        cands_by_s1[s1_id] = sorted(cands)
        needed_eids.update(cands)

    cand_records = fetch_candidate_records_sqlite(needed_eids)

    prep_s1 = {rec["entity_id"]: prepare_rep(rec) for rec in batch_records}
    prep_cands = {eid: prepare_rep(rec) for eid, rec in cand_records.items()}

    pair_info = [] # (s1_id, cand_id)
    X_feats = []

    for rec in batch_records:
        s1_id = rec["entity_id"]
        p1 = prep_s1.get(s1_id)
        if not p1:
            continue
        for cand_id in cands_by_s1.get(s1_id, []):
            p2 = prep_cands.get(cand_id)
            if not p2:
                continue
            feats = extract_pair_features_fast(p1, p2)
            X_feats.append(feats)
            pair_info.append((s1_id, cand_id))

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

def verify_smoke_test(matching_path, candidate_path):
    """Validate format, row counts, subset constraints, and empty/multiple match counts on smoke test."""
    print(f"\n{'='*70}\nVALIDATING SMOKE TEST OUTPUT (1,000 entities)\n{'='*70}", flush=True)

    matching_rows = []
    with open(matching_path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        hdr = next(reader)
        assert hdr == ["source1_entity_id", "matched_entity_ids"], f"Invalid matching header: {hdr}"
        matching_rows = list(reader)

    candidate_rows = []
    with open(candidate_path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        hdr = next(reader)
        assert hdr == ["source1_entity_id", "candidate_entity_ids"], f"Invalid candidate header: {hdr}"
        candidate_rows = list(reader)

    assert len(matching_rows) == 1_000, f"Expected 1,000 matching rows, got {len(matching_rows)}"
    assert len(candidate_rows) == 1_000, f"Expected 1,000 candidate rows, got {len(candidate_rows)}"

    n_empty = 0
    n_multiple = 0
    cand_dict = {row[0]: set(row[1].split(",")) if len(row) > 1 and row[1].strip() else set() for row in candidate_rows}

    for row in matching_rows:
        s1_id = row[0]
        matches = set(row[1].split(",")) if len(row) > 1 and row[1].strip() else set()
        cands = cand_dict.get(s1_id, set())

        if not matches:
            n_empty += 1
        elif len(matches) > 1:
            n_multiple += 1

        assert matches.issubset(cands), f"Error: Matches for {s1_id} are not a subset of candidates!"

    print(f"  Header Verification    : PASS", flush=True)
    print(f"  Data Row Count         : {len(matching_rows):,} (Exact match)", flush=True)
    print(f"  Subset Constraint      : PASS (All predicted matches in candidate list)", flush=True)
    print(f"  Empty Match Count      : {n_empty:,} / 1,000", flush=True)
    print(f"  Multiple Match Count   : {n_multiple:,} / 1,000", flush=True)
    print(f"  RAM Footprint          : {cg_mod.get_memory_mb():,.0f} MB", flush=True)
    print(f"Smoke Test PASSED 100%! Proceeding to full test inference execution...\n", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Inference Pipeline")
    parser.add_argument("--smoke-test-only", action="store_true", help="Run 1,000 S1 smoke test only")
    args = parser.parse_args()

    if args.smoke_test_only:
        run_inference(smoke_test=True)
    else:
        # Run 1,000 smoke test first, verify assertions, then transition into full run seamlessly!
        gen, booster, threshold = run_inference(smoke_test=True)
        run_inference(smoke_test=False, existing_gen=gen, existing_booster=booster, existing_thresh=threshold)
