#!/usr/bin/env python3
"""
07_feature_extractor.py — Ultra-Fast Pairwise Feature Extractor
===============================================================
Pre-computes entity-level representations ONCE to extract 15 pairwise features
for 880,000+ pairs in seconds instead of minutes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
cg = importlib.import_module("05_candidate_generator")

FEATURE_NAMES = [
    "exact_name_match",
    "core_name_match",
    "token_jaccard",
    "token_overlap_count",
    "token_overlap_ratio",
    "char_trigram_jaccard",
    "levenshtein_ratio",
    "prefix_token_match",
    "domain_squash_match",
    "addr_token_jaccard",
    "addr_digit_overlap",
    "addr_digit_jaccard",
    "has_pin_match",
    "same_country",
    "is_s2",
]

def _char_trigrams(s):
    if not s:
        return set()
    s = f"^{s}$"
    if len(s) < 3:
        return {s}
    return {s[i:i+3] for i in range(len(s)-2)}

def _fast_levenshtein_ratio(s1, s2):
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    len1, len2 = len(s1), len(s2)
    max_len = max(len1, len2)
    if abs(len1 - len2) > max_len * 0.5:
        return 0.0
    
    # Fast bounded Levenshtein
    prev = list(range(len2 + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1] * (len2 + 1)
        min_in_row = i + 1
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr[j + 1] = min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost)
            if curr[j + 1] < min_in_row:
                min_in_row = curr[j + 1]
        if min_in_row > max_len * 0.5:
            return 0.0
        prev = curr
    return 1.0 - (prev[len2] / max_len)

def prepare_entity_representation(rec):
    """Pre-compute normalized tokens, trigrams, and address numbers ONCE per entity."""
    eid = rec.get("entity_id", "")
    name = str(rec.get("business_name", "")).strip()
    addr = str(rec.get("business_address", "")).strip()
    ctry = str(rec.get("country", "")).strip()

    norm_name = cg.normalize_text(name)
    toks = cg._tokenize_normalized(norm_name)
    tok_set = set(toks)
    core_name = cg.make_core_name(toks)
    trigrams = _char_trigrams(norm_name)
    squash = "".join(toks)

    norm_addr = cg.normalize_text(addr)
    addr_toks = set(cg._tokenize_normalized(norm_addr))
    num_list = cg.extract_address_numbers(addr)
    num_set = set(num_list)
    pins = {n for n in num_set if len(n) == 6}

    return {
        "eid": eid,
        "norm_name": norm_name,
        "toks": toks,
        "tok_set": tok_set,
        "core_name": core_name,
        "trigrams": trigrams,
        "squash": squash,
        "addr_toks": addr_toks,
        "num_set": num_set,
        "pins": pins,
        "country": ctry,
    }

def extract_pair_features_fast(p1, p2):
    """Extract 15 pairwise features instantly from pre-computed entity dicts p1 and p2."""
    n1, n2 = p1["norm_name"], p2["norm_name"]
    exact_name_match = 1.0 if n1 and n1 == n2 else 0.0
    c1, c2 = p1["core_name"], p2["core_name"]
    core_name_match  = 1.0 if c1 and c1 == c2 else 0.0

    # Token overlap
    s1, s2 = p1["tok_set"], p2["tok_set"]
    inter = s1 & s2
    union = s1 | s2
    token_jaccard = len(inter) / len(union) if union else 0.0
    token_overlap_count = float(len(inter))
    min_tokens = min(len(s1), len(s2))
    token_overlap_ratio = len(inter) / min_tokens if min_tokens > 0 else 0.0

    # Trigram Jaccard
    t1, t2 = p1["trigrams"], p2["trigrams"]
    t_union = t1 | t2
    char_trigram_jaccard = len(t1 & t2) / len(t_union) if t_union else 0.0

    # Levenshtein ratio
    levenshtein_ratio = _fast_levenshtein_ratio(n1, n2) if n1 and n2 else 0.0

    # Prefix match
    tlist1, tlist2 = p1["toks"], p2["toks"]
    prefix_token_match = 1.0 if (tlist1 and tlist2 and tlist1[0] == tlist2[0]) else 0.0

    # Domain squash match
    sq1, sq2 = p1["squash"], p2["squash"]
    if sq1 and sq2:
        domain_squash_match = 1.0 if (sq1 == sq2 or sq1 in sq2 or sq2 in sq1) else 0.0
    else:
        domain_squash_match = 0.0

    # Address features
    a1, a2 = p1["addr_toks"], p2["addr_toks"]
    a_union = a1 | a2
    addr_token_jaccard = len(a1 & a2) / len(a_union) if a_union else 0.0

    num1, num2 = p1["num_set"], p2["num_set"]
    num_inter = num1 & num2
    num_union = num1 | num2
    addr_digit_overlap = float(len(num_inter))
    addr_digit_jaccard = len(num_inter) / len(num_union) if num_union else 0.0

    pins1, pins2 = p1["pins"], p2["pins"]
    has_pin_match = 1.0 if (pins1 and pins2 and len(pins1 & pins2) > 0) else 0.0

    # Context features
    same_country = 1.0 if p1["country"] and p1["country"] == p2["country"] else 0.0
    is_s2 = 1.0 if p2["eid"].startswith("S2-") else 0.0

    return [
        exact_name_match,
        core_name_match,
        token_jaccard,
        token_overlap_count,
        token_overlap_ratio,
        char_trigram_jaccard,
        levenshtein_ratio,
        prefix_token_match,
        domain_squash_match,
        addr_token_jaccard,
        addr_digit_overlap,
        addr_digit_jaccard,
        has_pin_match,
        same_country,
        is_s2,
    ]
