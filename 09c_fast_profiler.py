#!/usr/bin/env python3
"""
09c_fast_profiler.py — Targeted Profiler & Benchmark
====================================================
Profiles 1,000 S1 sample end-to-end and measures exact execution times for:
  - Candidate Generation
  - SQLite Reads vs In-Memory Dict Reads
  - Entity Representation Prep
  - Pairwise Feature Extraction
  - LightGBM Model Prediction
  - File Writing
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

def profile_sample(sample_size=1_000):
    print(f"\n{'='*70}\nPROFILING & COMPONENT BREAKDOWN ({sample_size:,} S1 Records)\n{'='*70}", flush=True)

    with open(CONFIG_FILE, encoding="utf-8") as f:
        config = json.load(f)
    thresh = float(config.get("optimal_threshold", 0.50))
    booster = lgb.Booster(model_file=str(MODEL_FILE))

    # Read S1 sample records
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
            if len(s1_records) >= sample_size:
                break

    # Build index
    t_idx0 = time.time()
    gen = CandidateGenerator()
    gen.build_from_files([str(TEST_S2), str(TEST_S3)])
    t_idx = time.time() - t_idx0
    print(f"Index built in {t_idx:.1f}s.", flush=True)

    # 1. Candidate Generation
    t0 = time.time()
    cands_by_s1 = {}
    needed_eids = set()
    for rec in s1_records:
        s1_id = rec["entity_id"]
        cands = gen.find_candidates(
            rec["business_name"], rec["business_address"], rec["country"], max_total=200
        )
        cands_by_s1[s1_id] = sorted(cands)
        needed_eids.update(cands)
    t_cand = time.time() - t0

    # 2. SQLite Reads
    t0 = time.time()
    conn = sqlite3.connect(str(DB_FILE), timeout=60.0)
    cursor = conn.cursor()
    cand_records_sql = {}
    eid_list = list(needed_eids)
    for i in range(0, len(eid_list), 5000):
        chunk = eid_list[i:i+5000]
        placeholders = ",".join(["?"] * len(chunk))
        cursor.execute(f"SELECT entity_id, business_name, business_address, country FROM entities WHERE entity_id IN ({placeholders})", chunk)
        for row in cursor.fetchall():
            cand_records_sql[row[0]] = {"entity_id": row[0], "business_name": row[1], "business_address": row[2], "country": row[3]}
    conn.close()
    t_sql = time.time() - t0

    # 3. Representation Preparation
    t0 = time.time()
    prep_s1 = {rec["entity_id"]: prepare_rep(rec) for rec in s1_records}
    prep_cands = {eid: prepare_rep(rec) for eid, rec in cand_records_sql.items()}
    t_prep = time.time() - t0

    # 4. Pairwise Feature Calculation (Sequential)
    t0 = time.time()
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
    t_feat = time.time() - t0

    # 5. Model Prediction
    t0 = time.time()
    preds_by_s1 = defaultdict(list)
    if X_feats:
        X_mat = np.array(X_feats, dtype=np.float32)
        probs = booster.predict(X_mat)
        for (s1_id, cand_id), prob in zip(pair_info, probs):
            if prob >= thresh:
                preds_by_s1[s1_id].append(cand_id)
    t_pred = time.time() - t0

    # 6. File Writing
    t0 = time.time()
    prof_dir = OUTPUT_DIR / "prof_sample"
    prof_dir.mkdir(parents=True, exist_ok=True)
    with open(prof_dir / "matching.tsv", "w", encoding="utf-8") as fm, \
         open(prof_dir / "candidate.tsv", "w", encoding="utf-8") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for rec in s1_records:
            s1_id = rec["entity_id"]
            cand_list = cands_by_s1.get(s1_id, [])
            match_list = sorted(preds_by_s1.get(s1_id, []))
            fm.write(f"{s1_id}\t{','.join(match_list)}\n")
            fc.write(f"{s1_id}\t{','.join(cand_list)}\n")
    t_write = time.time() - t0

    t_total = t_cand + t_sql + t_prep + t_feat + t_pred + t_write

    print("\n" + "="*70)
    print(f"EMPIRICAL PROFILING RESULTS ({sample_size:,} S1 records, {len(pair_info):,} candidate pairs):")
    print("="*70)
    print(f"  Candidate Generation   : {t_cand:6.2f}s ({t_cand/t_total*100:5.1f}%)")
    print(f"  SQLite Disk Reads      : {t_sql:6.2f}s ({t_sql/t_total*100:5.1f}%)")
    print(f"  Representation Prep    : {t_prep:6.2f}s ({t_prep/t_total*100:5.1f}%)")
    print(f"  Feature Calculation    : {t_feat:6.2f}s ({t_feat/t_total*100:5.1f}%)")
    print(f"  Model Prediction       : {t_pred:6.2f}s ({t_pred/t_total*100:5.1f}%)")
    print(f"  File Writing           : {t_write:6.2f}s ({t_write/t_total*100:5.1f}%)")
    print("-" * 70)
    print(f"  TOTAL SAMPLE TIME (1K) : {t_total:6.2f}s")
    print(f"  Rate per 10,000 S1     : {(t_total / sample_size) * 10_000:6.2f}s")
    print(f"  Projected Full 1.28M S1: {((t_total / sample_size) * 1_282_544) / 3600:6.2f} hours")
    print("="*70)

if __name__ == "__main__":
    profile_sample(1_000)
