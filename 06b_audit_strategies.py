#!/usr/bin/env python3
"""
06b_audit_strategies.py — Audit Strategy Metrics on 10,000 Pilot Dataset
========================================================================
Reports for each strategy S1..S7:
  1. Returned candidates (raw candidate count before union)
  2. New candidates added to union (in sequence S1..S7)
  3. True pairs touched
  4. True pairs found ONLY by that strategy
"""

import csv
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict

# Add ml_challenge to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
cg_mod = importlib.import_module("05_candidate_generator")
CandidateGenerator = cg_mod.CandidateGenerator

# Paths
BASE = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE / "dataset" / "train"
SPLIT_DIR = Path(__file__).resolve().parent / "splits"

S2_PATH = TRAIN_DIR / "train_source2.tsv"
S3_PATH = TRAIN_DIR / "train_source3.tsv"
S1_PATH = TRAIN_DIR / "train_source1.tsv"
VAL_IDS = SPLIT_DIR / "val_ids.txt"
VAL_GT  = SPLIT_DIR / "val_ground_truth.tsv"
PILOT_SIZE = 10_000

def main():
    # 1. Load pilot IDs and ground truth
    with open(VAL_IDS, encoding="utf-8") as f:
        all_ids = [line.strip() for line in f if line.strip()]
    all_ids.sort()
    pilot_ids = all_ids[:PILOT_SIZE]
    pilot_set = set(pilot_ids)

    gt = {}
    with open(VAL_GT, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            s1 = row[0]
            if s1 in pilot_set:
                matched = row[1] if len(row) > 1 else ""
                gt[s1] = set(matched.split(",")) if matched.strip() else set()

    # 2. Stream S1 records
    s1_records = {}
    with open(S1_PATH, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["entity_id"] in pilot_set:
                s1_records[row["entity_id"]] = {
                    "entity_id": row["entity_id"],
                    "business_name": row.get("business_name", ""),
                    "business_address": row.get("business_address", ""),
                    "country": row.get("country", ""),
                }
                if len(s1_records) == len(pilot_set):
                    break

    # 3. Build index
    print("Building index ...")
    gen = CandidateGenerator()
    gen.build_from_files([str(S2_PATH), str(S3_PATH)])

    # 4. Instrument per-strategy candidates per entity
    strats = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']
    
    returned_cands = Counter()
    new_in_union   = Counter()
    true_touched   = Counter()
    true_only      = Counter()

    print("Auditing strategies on 10,000 S1 records ...")
    t0 = time.time()

    for i, s1_id in enumerate(pilot_ids):
        rec = s1_records[s1_id]
        true_set = gt.get(s1_id, set())

        # Collect hits per strategy for this record
        strat_hits = {}

        norm_name    = cg_mod.normalize_text(rec["business_name"])
        norm_name_af = cg_mod.normalize_accent_fold(rec["business_name"])
        sig_tokens   = cg_mod._tokenize_normalized(norm_name)
        sig_tokens_af = cg_mod._tokenize_normalized(norm_name_af) if norm_name_af != norm_name else sig_tokens
        core_name    = cg_mod.make_core_name(sig_tokens)
        core_name_af = cg_mod.make_core_name(sig_tokens_af) if norm_name_af != norm_name else core_name
        addr_nums    = cg_mod.extract_address_numbers(rec["business_address"])
        country      = rec["country"]
        cap          = gen.MAX_PER_STRATEGY

        # S1
        s1_cands = set()
        if norm_name:
            h = gen.idx_name.get((norm_name, country))
            if h and len(h) <= cap: s1_cands.update(h)
            if norm_name_af != norm_name:
                h = gen.idx_name.get((norm_name_af, country))
                if h and len(h) <= cap: s1_cands.update(h)
        strat_hits['S1'] = s1_cands

        # S2
        s2_cands = set()
        if core_name:
            h = gen.idx_core.get((core_name, country))
            if h and len(h) <= cap: s2_cands.update(h)
            if core_name_af != core_name:
                h = gen.idx_core.get((core_name_af, country))
                if h and len(h) <= cap: s2_cands.update(h)
        strat_hits['S2'] = s2_cands

        # S3
        s3_cands = set()
        if sig_tokens:
            min_ov = 1 if len(sig_tokens) == 1 else gen.TOKEN_OVERLAP_MIN
            overlap = Counter()
            for tok in set(sig_tokens):
                posting = gen.idx_token.get((tok, country))
                if posting: overlap.update(posting)
            tok_hits = {eid for eid, cnt in overlap.items() if cnt >= min_ov}
            if len(tok_hits) <= cap:
                s3_cands.update(tok_hits)
            elif min_ov < len(sig_tokens):
                higher = min_ov + 1
                tok_hits = {eid for eid, cnt in overlap.items() if cnt >= higher}
                if len(tok_hits) <= cap: s3_cands.update(tok_hits)
        strat_hits['S3'] = s3_cands

        # S4
        s4_cands = set()
        if addr_nums and sig_tokens:
            h = gen.idx_addr.get((addr_nums, sig_tokens[0], country))
            if h and len(h) <= cap: s4_cands.update(h)
        strat_hits['S4'] = s4_cands

        # S5
        s5_cands = set()
        if norm_name:
            for other in gen.known_countries:
                if other != country:
                    h = gen.idx_name.get((norm_name, other))
                    if h and len(h) <= cap: s5_cands.update(h)
        strat_hits['S5'] = s5_cands

        # S6
        s6_cands = set()
        if len(sig_tokens) >= 2:
            squash = ''.join(sig_tokens)
            h = gen.idx_token.get((squash, country))
            if h and len(h) <= cap: s6_cands.update(h)
            for other in gen.known_countries:
                if other != country:
                    h = gen.idx_token.get((squash, other))
                    if h and len(h) <= cap: s6_cands.update(h)
        strat_hits['S6'] = s6_cands

        # S7
        s7_cands = set()
        if addr_nums and len(addr_nums) >= 2:
            h = gen.idx_addrnum.get((addr_nums, country))
            if h and len(h) <= cap: s7_cands.update(h)
        strat_hits['S7'] = s7_cands

        # Accumulate metrics
        running_union = set()
        for s in strats:
            hits = strat_hits[s]
            returned_cands[s] += len(hits)
            
            # new added to union sequentially in order S1..S7
            new_added = hits - running_union
            new_in_union[s] += len(new_added)
            running_union.update(hits)

            # true touched
            if true_set:
                tp = hits & true_set
                true_touched[s] += len(tp)

        # true only by each strategy
        if true_set:
            for s in strats:
                hits = strat_hits[s]
                other_hits = set()
                for o in strats:
                    if o != s:
                        other_hits.update(strat_hits[o])
                only = (hits & true_set) - other_hits
                true_only[s] += len(only)

    print(f"Audit completed in {time.time()-t0:.1f}s")
    print("\n" + "="*80)
    print("STRATEGY AUDIT BREAKDOWN (10,000 S1 Pilot Entities)")
    print("="*80)
    hdr = f"{'Strategy':<10} {'Returned Cands':>15} {'New to Union':>15} {'True Touched':>15} {'True ONLY':>15}"
    print(hdr)
    print("-" * 80)
    for s in strats:
        print(f"{s:<10} {returned_cands[s]:>15,} {new_in_union[s]:>15,} {true_touched[s]:>15,} {true_only[s]:>15,}")
    print("-" * 80)

if __name__ == "__main__":
    main()
