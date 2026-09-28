#!/usr/bin/env python3
"""
create_fallback.py — URGENT Deadline Recovery Fallback Generator
=================================================================
Reads completed rows from the background inference output (ml_challenge/output/)
and generates complete, valid fallback TSVs in ml_challenge/fallback_output/.

Rules:
1. Reads dataset/test/test_source1.tsv in original order (1,732,544 S1 entities).
2. Takes ONLY the consistent common prefix of S1 IDs present in BOTH
   ml_challenge/output/matching_results.tsv and ml_challenge/output/candidate_pairs.tsv.
3. Ignores partial final lines or rows present in only one file.
4. For all remaining S1 IDs, writes an empty string in the second field (s1_id\t\n).
5. Does NOT alter or interrupt the running background inference files.
"""

import os
import sys

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(base_dir, ".."))

    test_s1_path = os.path.join(root_dir, "dataset", "test", "test_source1.tsv")
    running_match_path = os.path.join(base_dir, "output", "matching_results.tsv")
    running_cand_path = os.path.join(base_dir, "output", "candidate_pairs.tsv")

    out_dir = os.path.join(base_dir, "fallback_output")
    os.makedirs(out_dir, exist_ok=True)

    fallback_match_path = os.path.join(out_dir, "matching_results.tsv")
    fallback_cand_path = os.path.join(out_dir, "candidate_pairs.tsv")

    print(f"Reading test S1 entities from: {test_s1_path}")
    s1_ids = []
    with open(test_s1_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            if not line.strip():
                continue
            s1_id = line.split("\t", 1)[0].strip()
            s1_ids.append(s1_id)

    total_s1 = len(s1_ids)
    print(f"Total test S1 entities required: {total_s1:,}")

    # Read running matching_results.tsv
    match_data = {}
    print(f"Reading running matches from: {running_match_path}")
    if os.path.exists(running_match_path):
        with open(running_match_path, "r", encoding="utf-8") as f:
            m_header = f.readline()
            for line in f:
                if not line.endswith("\n"):
                    # Partially written line at end of file, ignore
                    break
                parts = line.rstrip("\r\n").split("\t", 1)
                if len(parts) == 2:
                    s1_id, matched_str = parts[0], parts[1]
                elif len(parts) == 1:
                    s1_id, matched_str = parts[0], ""
                else:
                    continue
                match_data[s1_id] = matched_str

    # Read running candidate_pairs.tsv
    cand_data = {}
    print(f"Reading running candidates from: {running_cand_path}")
    if os.path.exists(running_cand_path):
        with open(running_cand_path, "r", encoding="utf-8") as f:
            c_header = f.readline()
            for line in f:
                if not line.endswith("\n"):
                    # Partially written line at end of file, ignore
                    break
                parts = line.rstrip("\r\n").split("\t", 1)
                if len(parts) == 2:
                    s1_id, cand_str = parts[0], parts[1]
                elif len(parts) == 1:
                    s1_id, cand_str = parts[0], ""
                else:
                    continue
                cand_data[s1_id] = cand_str

    # Determine consistent completed prefix
    completed_prefix_count = 0
    for s1_id in s1_ids:
        if s1_id in match_data and s1_id in cand_data:
            completed_prefix_count += 1
        else:
            break

    pct = (completed_prefix_count / total_s1) * 100
    print(f"Consistent completed S1 entities: {completed_prefix_count:,} / {total_s1:,} ({pct:.2f}%)")
    print(f"Remaining S1 entities to pad with empty predictions: {total_s1 - completed_prefix_count:,}")

    # Write fallback files
    print(f"Writing fallback matching results to: {fallback_match_path}")
    with open(fallback_match_path, "w", encoding="utf-8") as fm:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for i, s1_id in enumerate(s1_ids):
            if i < completed_prefix_count:
                m_str = match_data[s1_id]
                fm.write(f"{s1_id}\t{m_str}\n")
            else:
                fm.write(f"{s1_id}\t\n")

    print(f"Writing fallback candidate pairs to: {fallback_cand_path}")
    with open(fallback_cand_path, "w", encoding="utf-8") as fc:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for i, s1_id in enumerate(s1_ids):
            if i < completed_prefix_count:
                c_str = cand_data[s1_id]
                fc.write(f"{s1_id}\t{c_str}\n")
            else:
                fc.write(f"{s1_id}\t\n")

    print("Fallback files successfully generated!")
    fm_size = os.path.getsize(fallback_match_path)
    fc_size = os.path.getsize(fallback_cand_path)
    print(f"  matching_results.tsv size: {fm_size:,} bytes ({fm_size / (1024*1024):.2f} MB)")
    print(f"  candidate_pairs.tsv size:  {fc_size:,} bytes ({fc_size / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    main()
