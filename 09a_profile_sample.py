#!/usr/bin/env python3
"""
09a_profile_sample.py — Profiling script for candidate generation & feature extraction pipeline.
"""
import time
import json
import csv
import sys
import sqlite3
import numpy as np
import lightgbm as lgb
from pathlib import Path
from collections import defaultdict

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
OUTPUT_DIR = ML_DIR / "output"
RESULT_DIR = ML_DIR / "pilot_results"

TEST_S1 = TEST_DIR / "test_source1.tsv"
TEST_S2 = TEST_DIR / "test_source2.tsv"
TEST_S3 = TEST_DIR / "test_source3.tsv"

MODEL_FILE = RESULT_DIR / "matcher_lgbm.txt"
CONFIG_FILE = RESULT_DIR / "matcher_config.json"
DB_FILE = OUTPUT_DIR / "test_entities.db"

def fetch_candidate_records_sqlite(needed_eids):
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

def profile(num_sample=5_000):
    print(f"Loading model and candidate generator index for {num_sample:,} profiling sample...", flush=True)
    with open(CONFIG_FILE, encoding="utf-8") as f:
        config = json.load(f)
    thresh = float(config.get("optimal_threshold", 0.50))
    booster = lgb.Booster(model_file=str(MODEL_FILE))

    gen = CandidateGenerator()
    t0 = time.time()
    gen.build_from_files([str(TEST_S2), str(TEST_S3)])
    print(f"Candidate generator index ready in {time.time()-t0:.1f}s.", flush=True)

    # Read num_sample S1 records
    s1_records = []
    with open(TEST_S1, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            s1_records.append({
                "entity_id": row[0],
                "business_name": row[1] if len(row) > 1 else "",
                "business_address": row[2] if len(row) > 2 else "",
                "country": row[3].strip() if len(row) > 3 else "",
            })
            if len(s1_records) >= num_sample:
                break

    print(f"\nProfiling {len(s1_records):,} S1 records...")
    t_total_start = time.time()

    # Step 1: Candidate Generation
    t_cand_start = time.time()
    cands_by_s1 = {}
    needed_eids = set()
    for rec in s1_records:
        s1_id = rec["entity_id"]
        cands = gen.find_candidates(
            rec["business_name"], rec["business_address"], rec["country"], max_total=200
        )
        cands_by_s1[s1_id] = sorted(cands)
        needed_eids.update(cands)
    t_cand = time.time() - t_cand_start

    # Step 2: SQLite Reads
    t_sql_start = time.time()
    cand_records = fetch_candidate_records_sqlite(needed_eids)
    t_sql = time.time() - t_sql_start

    # Step 3: Representation Preparation (S1 + Candidates)
    t_prep_start = time.time()
    prep_s1 = {rec["entity_id"]: prepare_rep(rec) for rec in s1_records}
    prep_cands = {eid: prepare_rep(rec) for eid, rec in cand_records.items()}
    t_prep = time.time() - t_prep_start

    # Step 4: Pairwise Feature Calculation
    t_feat_start = time.time()
    pair_info = []
    X_feats = []
    for rec in s1_records:
        s1_id = rec["entity_id"]
        p1 = prep_s1.get(s1_id)
        if not p1: continue
        for cand_id in cands_by_s1.get(s1_id, []):
            p2 = prep_cands.get(cand_id)
            if not p2: continue
            feats = extract_pair_features_fast(p1, p2)
            X_feats.append(feats)
            pair_info.append((s1_id, cand_id))
    t_feat = time.time() - t_feat_start

    # Step 5: Model Prediction
    t_pred_start = time.time()
    preds_by_s1 = defaultdict(list)
    if X_feats:
        X_mat = np.array(X_feats, dtype=np.float32)
        probs = booster.predict(X_mat)
        for (s1_id, cand_id), prob in zip(pair_info, probs):
            if prob >= thresh:
                preds_by_s1[s1_id].append(cand_id)
    t_pred = time.time() - t_pred_start

    # Step 6: File Writing
    t_write_start = time.time()
    prof_out_dir = OUTPUT_DIR / "profile_test"
    prof_out_dir.mkdir(parents=True, exist_ok=True)
    with open(prof_out_dir / "matching.tsv", "w", encoding="utf-8") as fm, \
         open(prof_out_dir / "candidate.tsv", "w", encoding="utf-8") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for rec in s1_records:
            s1_id = rec["entity_id"]
            cand_list = cands_by_s1.get(s1_id, [])
            match_list = sorted(preds_by_s1.get(s1_id, []))
            fm.write(f"{s1_id}\t{','.join(match_list)}\n")
            fc.write(f"{s1_id}\t{','.join(cand_list)}\n")
    t_write = time.time() - t_write_start

    t_total = time.time() - t_total_start

    print("\n" + "="*70)
    print(f"PROFILING RESULTS ({num_sample:,} S1 records, {len(pair_info):,} candidate pairs):")
    print("="*70)
    print(f"  1. Candidate Generation   : {t_cand:6.2f}s ({t_cand/t_total*100:5.1f}%)")
    print(f"  2. SQLite Reads          : {t_sql:6.2f}s ({t_sql/t_total*100:5.1f}%)")
    print(f"  3. Prep Representations  : {t_prep:6.2f}s ({t_prep/t_total*100:5.1f}%)")
    print(f"  4. Feature Calculation   : {t_feat:6.2f}s ({t_feat/t_total*100:5.1f}%)")
    print(f"  5. Model Prediction      : {t_pred:6.2f}s ({t_pred/t_total*100:5.1f}%)")
    print(f"  6. File Writing          : {t_write:6.2f}s ({t_write/t_total*100:5.1f}%)")
    print("-" * 70)
    print(f"  TOTAL SAMPLE TIME        : {t_total:6.2f}s")
    print(f"  Rate per 10,000 S1       : {(t_total / num_sample) * 10_000:6.2f}s")
    print(f"  Projected Full (1.73M S1): {(t_total / num_sample) * 1_732_544 / 3600:6.2f} hours")
    print("="*70)

if __name__ == "__main__":
    profile(num_sample=5_000)
