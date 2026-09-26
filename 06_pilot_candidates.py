#!/usr/bin/env python3
"""
Phase 2 — Pilot Candidate Generation (10 000 validation S1 IDs)
================================================================

1. Loads 10 000 deterministic validation S1 IDs and their ground truth.
2. Builds inverted indexes from the full training S2/S3 files.
3. Streams training S1 to retrieve pilot records.
4. Generates candidates and evaluates recall vs. ground truth.
5. Looks up missed-pair details for diagnosis.
6. Saves results to ml_challenge/pilot_results/.

Run from student_resource/:
    python ml_challenge/06_pilot_candidates.py
"""

import csv
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import time
from collections import Counter, defaultdict
from pathlib import Path

# Add ml_challenge/ to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
cg_mod = import_module("05_candidate_generator")

CandidateGenerator       = cg_mod.CandidateGenerator
normalize_text           = cg_mod.normalize_text
extract_significant_tokens = cg_mod.extract_significant_tokens
get_memory_mb            = cg_mod.get_memory_mb

# ── paths (relative to student_resource/) ─────────────────────────────
BASE        = Path(__file__).resolve().parent.parent
TRAIN_DIR   = BASE / "dataset" / "train"
SPLIT_DIR   = Path(__file__).resolve().parent / "splits"
RESULT_DIR  = Path(__file__).resolve().parent / "pilot_results"

S2_PATH     = TRAIN_DIR / "train_source2.tsv"
S3_PATH     = TRAIN_DIR / "train_source3.tsv"
S1_PATH     = TRAIN_DIR / "train_source1.tsv"
VAL_IDS     = SPLIT_DIR / "val_ids.txt"
VAL_GT      = SPLIT_DIR / "val_ground_truth.tsv"

PILOT_SIZE  = 10_000     # number of validation S1 IDs to use
TEST_S1_COUNT = 1_732_544  # from Phase 1 analysis


# ═══════════════════════════════════════════════════════════════════════════
#  Step 1 — Load pilot IDs and ground truth
# ═══════════════════════════════════════════════════════════════════════════

def load_pilot_ids():
    """Return a sorted list of the first PILOT_SIZE validation S1 IDs."""
    with open(VAL_IDS, encoding="utf-8") as f:
        all_ids = [line.strip() for line in f if line.strip()]
    all_ids.sort()
    pilot = all_ids[:PILOT_SIZE]
    print(f"Pilot IDs loaded: {len(pilot):,}  (from {len(all_ids):,} val IDs)")
    return pilot


def load_ground_truth(pilot_set):
    """Return {s1_id: set(matched_ids)} for IDs in pilot_set."""
    gt = {}
    with open(VAL_GT, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header
        for row in reader:
            s1 = row[0]
            if s1 in pilot_set:
                matched = row[1] if len(row) > 1 else ""
                gt[s1] = set(matched.split(",")) if matched.strip() else set()
    print(f"Ground truth loaded for {len(gt):,} pilot IDs")
    return gt


# ═══════════════════════════════════════════════════════════════════════════
#  Step 2 — Load pilot S1 records (stream, keep only pilot IDs)
# ═══════════════════════════════════════════════════════════════════════════

def load_s1_records(pilot_set):
    """Stream train_source1.tsv; keep rows whose entity_id ∈ pilot_set."""
    records = {}
    with open(S1_PATH, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["entity_id"] in pilot_set:
                records[row["entity_id"]] = {
                    "entity_id":        row["entity_id"],
                    "business_name":    row.get("business_name", ""),
                    "business_address": row.get("business_address", ""),
                    "country":          row.get("country", ""),
                }
                if len(records) == len(pilot_set):
                    break  # found all
    print(f"S1 records loaded: {len(records):,}")
    return records


# ═══════════════════════════════════════════════════════════════════════════
#  Step 3 — Generate candidates
# ═══════════════════════════════════════════════════════════════════════════

def generate_candidates(gen, s1_records, max_total=None, label=""):
    """Return {s1_id: set(candidate_ids)} and aggregate strategy counts."""
    results  = {}
    strat_totals = Counter()
    t0 = time.time()
    for i, (s1_id, rec) in enumerate(s1_records.items()):
        cands, strat = gen.find_candidates(
            rec["business_name"],
            rec["business_address"],
            rec["country"],
            max_total=max_total,
            track_strategies=True,
        )
        results[s1_id] = cands
        for k, v in strat.items():
            strat_totals[k] += v
        if (i + 1) % 2000 == 0:
            print(f"    {i+1:>6,} / {len(s1_records):,} queried  "
                  f"({time.time()-t0:.1f}s)")
    query_time = time.time() - t0
    print(f"  [{label}] Candidate generation done: {len(results):,} S1 records  "
          f"({query_time:.1f}s)")
    return results, query_time, strat_totals


# ═══════════════════════════════════════════════════════════════════════════
#  Step 4 — Evaluate recall
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(gt, results, s1_records):
    """Compute and print comprehensive recall metrics."""

    # ── overall ───────────────────────────────────────────────────────
    total_true   = 0
    found_true   = 0
    all_found    = 0
    n_singletons = 0

    # per-country
    country_total = Counter()
    country_found = Counter()
    country_all   = Counter()
    country_n     = Counter()

    # per-source (S2 vs S3)
    source_total  = Counter()
    source_found  = Counter()

    # candidate counts
    cand_counts = []

    # missed pairs
    missed_pairs = []

    for s1_id, true_set in gt.items():
        cands    = results.get(s1_id, set())
        country  = s1_records[s1_id]["country"]
        n_cands  = len(cands)
        cand_counts.append(n_cands)
        country_n[country] += 1

        if not true_set:
            n_singletons += 1
            all_found += 1
            country_all[country] += 1
            continue

        total_true += len(true_set)
        country_total[country] += len(true_set)
        tp = true_set & cands
        found_true += len(tp)
        country_found[country] += len(tp)

        for mid in true_set:
            src = "S2" if mid.startswith("S2-") else "S3"
            source_total[src] += 1
            if mid in cands:
                source_found[src] += 1

        if true_set <= cands:
            all_found += 1
            country_all[country] += 1
        else:
            for mid in true_set - cands:
                missed_pairs.append((s1_id, mid))

    n_pilot = len(gt)
    recall = found_true / total_true if total_true else 0
    pct_all = 100 * all_found / n_pilot if n_pilot else 0

    cand_counts.sort()
    n = len(cand_counts)
    median_c = cand_counts[n // 2] if n else 0
    p95_c    = cand_counts[int(n * 0.95)] if n else 0
    max_c    = cand_counts[-1] if n else 0
    total_c  = sum(cand_counts)
    mean_c   = total_c / n if n else 0

    # estimated disk size for full test
    avg_entry_bytes = 14 + mean_c * 14
    est_disk_mb = TEST_S1_COUNT * avg_entry_bytes / 1024**2

    print("\n" + "=" * 70)
    print("PILOT RESULTS  —  10,000 validation S1 IDs")
    print("=" * 70)

    print(f"\n  Pilot S1 IDs        : {n_pilot:>10,}")
    print(f"    of which singletons: {n_singletons:>10,}")
    print(f"  Total true pairs    : {total_true:>10,}")
    print(f"  Found true pairs    : {found_true:>10,}")

    print(f"\n  -- Recall -------------------------------------------------------")
    print(f"  Overall candidate recall       : {recall:.4f}  "
          f"({found_true:,} / {total_true:,})")
    print(f"  S1 IDs with ALL matches found  : {all_found:,} / {n_pilot:,}  "
          f"({pct_all:.1f}%)")

    print(f"\n  -- Per-country recall -------------------------------------------")
    for c in sorted(country_total.keys()):
        ct = country_total[c]
        cf = country_found[c]
        cr = cf / ct if ct else 0
        ca = country_all[c]
        cn = country_n[c]
        print(f"    {c:>8s} : recall {cr:.4f}  ({cf:,}/{ct:,})  "
              f"| all-found {ca:,}/{cn:,}")

    print(f"\n  -- Per-source recall --------------------------------------------")
    for src in sorted(source_total.keys()):
        st = source_total[src]
        sf = source_found[src]
        sr = sf / st if st else 0
        print(f"    {src} : recall {sr:.4f}  ({sf:,}/{st:,})")

    print(f"\n  -- Candidate counts per S1 --------------------------------------")
    print(f"    Mean     : {mean_c:>10.1f}")
    print(f"    Median   : {median_c:>10,}")
    print(f"    p95      : {p95_c:>10,}")
    print(f"    Max      : {max_c:>10,}")
    print(f"    Total    : {total_c:>10,}")

    print(f"\n  -- Scale estimates ----------------------------------------------")
    print(f"    Avg candidates per S1         : {mean_c:.1f}")
    print(f"    Est full-test candidates      : "
          f"{int(TEST_S1_COUNT * mean_c):,}")
    print(f"    Est candidate_pairs.tsv size  : {est_disk_mb:,.0f} MB")

    return missed_pairs, {
        'recall': recall, 'pct_all': pct_all,
        'mean': mean_c, 'median': median_c, 'p95': p95_c, 'max': max_c,
        'total': total_c, 'found': found_true, 'total_true': total_true,
        'est_mb': est_disk_mb,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Step 5 — Look up missed-pair details
# ═══════════════════════════════════════════════════════════════════════════

def lookup_details(eids_to_find, source_paths):
    """Stream S2/S3 to fetch full rows for a set of entity IDs."""
    found = {}
    for path in source_paths:
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)  # skip header
            for row in reader:
                eid = row[0]
                if eid in eids_to_find:
                    found[eid] = {
                        "business_name":    row[1] if len(row) > 1 else "",
                        "business_address": row[2] if len(row) > 2 else "",
                        "country":          row[3].strip() if len(row) > 3 else "",
                    }
                    if len(found) == len(eids_to_find):
                        return found
    return found


def print_missed_examples(missed_pairs, s1_records, n=10):
    """Print examples of missed true pairs with diagnosis."""
    if not missed_pairs:
        print("\n  No missed pairs -- perfect candidate recall!")
        return

    s2s3_ids = set(mid for _, mid in missed_pairs[:max(n * 5, 100)])
    details  = lookup_details(s2s3_ids, [str(S2_PATH), str(S3_PATH)])

    print(f"\n  Looking up details for {len(s2s3_ids)} missed S2/S3 records ...")
    print(f"\n  -- {min(n, len(missed_pairs))} example missed true pairs "
          f"(of {len(missed_pairs):,} total) ----")

    shown = 0
    for s1_id, mid in missed_pairs:
        if shown >= n:
            break
        s2_info = details.get(mid)
        if not s2_info:
            continue
        shown += 1
        s1_rec   = s1_records.get(s1_id, {})
        s1_name  = s1_rec.get("business_name", "")
        s1_addr  = s1_rec.get("business_address", "")
        s1_ctry  = s1_rec.get("country", "")
        s1_sig   = extract_significant_tokens(s1_name)
        s2_name  = s2_info["business_name"]
        s2_addr  = s2_info["business_address"]
        s2_ctry  = s2_info["country"]
        s2_sig   = extract_significant_tokens(s2_name)
        shared   = set(s1_sig) & set(s2_sig)

        print(f"\n  +-- Miss #{shown}")
        print(f"  |  S1 {s1_id:18s}  [{s1_ctry}]")
        print(f"  |    name: {s1_name}")
        print(f"  |    addr: {s1_addr}")
        print(f"  |    sig-tokens: {s1_sig}")
        print(f"  |  {mid:20s}  [{s2_ctry}]")
        print(f"  |    name: {s2_name}")
        print(f"  |    addr: {s2_addr}")
        print(f"  |    sig-tokens: {s2_sig}")
        print(f"  |  shared tokens: {shared if shared else 'none'}")
        print(f"  |  norm S1: '{normalize_text(s1_name)}'")
        print(f"  |  norm S2: '{normalize_text(s2_name)}'")
        print(f"  +--------------------------------------------")

    if len(missed_pairs) > n:
        print(f"\n  ... and {len(missed_pairs) - n:,} more missed pairs.")


# ═══════════════════════════════════════════════════════════════════════════
#  Step 6 — Save results
# ═══════════════════════════════════════════════════════════════════════════

def save_results(results, missed_pairs, s1_records, gt, label=""):
    """Save pilot candidates and metrics to disk."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{label}" if label else ""

    # candidates TSV
    cand_path = RESULT_DIR / f"pilot_candidates{suffix}.tsv"
    with open(cand_path, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in sorted(results.keys()):
            cands_str = ",".join(sorted(results[s1_id]))
            f.write(f"{s1_id}\t{cands_str}\n")
    print(f"\n  Saved {cand_path}")

    # missed pairs TSV
    miss_path = RESULT_DIR / f"pilot_missed_pairs{suffix}.tsv"
    with open(miss_path, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmissed_entity_id\n")
        for s1_id, mid in missed_pairs:
            f.write(f"{s1_id}\t{mid}\n")
    print(f"  Saved {miss_path}  ({len(missed_pairs):,} missed pairs)")

    # summary
    total_true = sum(len(s) for s in gt.values())
    found = sum(len(gt[s1] & results.get(s1, set())) for s1 in gt)
    recall = found / total_true if total_true else 0
    summ_path = RESULT_DIR / f"pilot_summary{suffix}.txt"
    with open(summ_path, "w", encoding="utf-8") as f:
        f.write(f"Config              : {label}\n")
        f.write(f"Pilot size          : {len(gt):,}\n")
        f.write(f"Total true pairs    : {total_true:,}\n")
        f.write(f"Found true pairs    : {found:,}\n")
        f.write(f"Candidate recall    : {recall:.6f}\n")
        f.write(f"Missed pairs        : {len(missed_pairs):,}\n")
        f.write(f"Total candidates    : {sum(len(v) for v in results.values()):,}\n")
    print(f"  Saved {summ_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    # ── 1. Pilot IDs and ground truth ─────────────────────────────────
    pilot_ids = load_pilot_ids()
    pilot_set = set(pilot_ids)
    gt        = load_ground_truth(pilot_set)

    # ── 2. Build indexes from full training S2/S3 (done ONCE) ─────────
    print(f"\nBuilding candidate indexes ...")
    gen = CandidateGenerator()
    gen.build_from_files([str(S2_PATH), str(S3_PATH)])

    # ── 3. Load pilot S1 records ──────────────────────────────────────
    print(f"\nLoading S1 records for pilot ...")
    s1_records = load_s1_records(pilot_set)

    # ── 4. Run 3 configs on the SAME index ───────────────────────────
    configs = [
        ("A_nocap",  None),   # No cap — measures ceiling recall
        ("B_cap200", 200),    # Cap 200 — balance recall vs set size
        ("C_cap100", 100),    # Cap 100 — tighter, smaller TSV
    ]

    summary_rows = []
    all_results  = {}
    all_missed   = {}
    all_strats   = {}
    qtimes       = {}

    for label, cap in configs:
        print(f"\n{'='*60}")
        print(f"  CONFIG {label}  (max_total={cap})")
        print(f"{'='*60}")
        res, qt, strat = generate_candidates(gen, s1_records,
                                             max_total=cap, label=label)
        all_results[label] = res
        all_strats[label]  = strat
        qtimes[label]      = qt
        missed, metrics    = evaluate(gt, res, s1_records)
        all_missed[label]  = missed
        metrics['query_time'] = qt
        metrics['label']      = label
        metrics['cap']        = str(cap) if cap else "none"
        summary_rows.append(metrics)

    # ── 5. Print comparison table ─────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("CANDIDATE CONFIG COMPARISON  (same index, same 10K pilot)")
    print(f"{'='*70}")
    hdr = f"{'Config':<12} {'Cap':>5} {'Recall':>7} {'AllFnd%':>8} "
    hdr += f"{'Mean':>7} {'Median':>7} {'p95':>7} {'Max':>7} {'Est MB':>8}"
    print(hdr)
    print("-" * 70)
    for m in summary_rows:
        row = (f"{m['label']:<12} {m['cap']:>5} {m['recall']:>7.4f} "
               f"{m['pct_all']:>8.1f} {m['mean']:>7.1f} {m['median']:>7,} "
               f"{m['p95']:>7,} {m['max']:>7,} {m['est_mb']:>8,.0f}")
        print(row)
    print("-" * 70)
    print(f"  Baseline (prev run, old code): recall=0.7727  mean=753.5  est=17,454 MB")

    # ── 6. Per-strategy breakdown for Config A ─────────────────────────
    print(f"\n\nStrategy breakdown for Config A_nocap (total unique candidates):")
    strat_a = all_strats.get("A_nocap", {})
    for s in ['S1', 'S2', 'S4', 'S6', 'S3', 'S5', 'S7']:
        print(f"  {s}: {strat_a.get(s, 0):>10,} candidates contributed")

    # ── 7. Missed-pair diagnosis (Config A — most recall) ─────────────
    print_missed_examples(all_missed["A_nocap"], s1_records, n=10)

    # ── 8. Save all configs ───────────────────────────────────────────
    for label, _ in configs:
        save_results(all_results[label], all_missed[label],
                     s1_records, gt, label=label)

    # ── 9. Timing summary ─────────────────────────────────────────────
    total_time = time.time() - t_start
    print(f"\n  -- Timing -------------------------------------------------------")
    print(f"    Index build   : {gen.build_time_s:>8.1f}s")
    for label, cap in configs:
        print(f"    Query {label}: {qtimes[label]:>8.1f}s")
    print(f"    Total wall    : {total_time:>8.1f}s")
    print(f"    Peak memory   : {gen.peak_mem_mb:>8.0f} MB")

    print(f"\nPilot complete.  Total time: {total_time:.0f}s")


if __name__ == "__main__":
    main()
