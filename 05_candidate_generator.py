#!/usr/bin/env python3
"""
Phase 2 — Candidate Generator for Entity Resolution
=====================================================

Builds inverted indexes on Source 2/3 records and generates candidate
matches for Source 1 records using multiple complementary blocking
strategies.

Indexes (built from S2/S3 records):
  1. idx_name:  (normalized_name, country) → [eid, ...]
  2. idx_core:  (sorted_significant_tokens, country) → [eid, ...]
  3. idx_token: (significant_token, country) → [eid, ...]   (inverted)
  4. idx_addr:  (address_numbers, first_sig_token, country) → [eid, ...]

Query strategies:
  S1 → Exact name + same country             (from idx_name)
  S2 → Core name + same country              (from idx_core)
  S3 → Token overlap ≥ threshold + country    (from idx_token)
  S4 → Address + first token + country        (from idx_addr)
  S5 → Cross-country exact name              (from idx_name, all countries)

All operations preserve Unicode and handle missing fields gracefully.
Entity IDs are sys.intern()-ed to share memory across indexes.
"""

import csv
import os
import re
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
#  Text Processing
# ═══════════════════════════════════════════════════════════════════════════

# Unicode categories to keep in normalized text.
# L* = letters (Lo includes Devanagari/CJK base chars)
# M* = combining marks (Mn = non-spacing, Mc = spacing-combining)
#      Devanagari vowel signs (matras) are Mc; anusvara is Mn.
#      Python's regex \w does NOT reliably match Mc/Mn, so we use
#      explicit category checks instead.
# N* = numbers, Zs = space separator
_KEEP_CAT0 = frozenset('LMN')   # first char of category code


def normalize_text(text):
    """
    Unicode-preserving text normalization.

    Preserves ALL Unicode combining marks, including:
    - Devanagari vowel signs / matras (Mc, e.g. \u093E  \u093F  \u0947)
    - Devanagari anusvara / chandrabindu (Mn, e.g. \u0902  \u0901)
    - Arabic harakat (Mn), Thai vowels (Mn), etc.

    INTENTIONAL DESIGN: uses explicit Unicode category checks rather than
    Python's \\w regex, because \\w excludes Mc/Mn categories and therefore
    strips essential Devanagari marks.

    Steps:
    - NFC normalization
    - Lowercase
    - Replace '&' with 'and'
    - Replace punctuation / symbols with space (keep L*, M*, N*, spaces)
    - Collapse whitespace
    """
    if not text or not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = text.replace("&", " and ")
    _cat = unicodedata.category
    text = ''.join(
        c if (_cat(c)[0] in _KEEP_CAT0 or c == ' ') else ' '
        for c in text
    )
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_accent_fold(text):
    """
    Accent-insensitive normalization for Latin script ONLY.

    Applies normalize_text first, then strips combining diacritics
    ONLY from ASCII base characters (a-z).  Devanagari, Arabic, CJK,
    and other non-Latin combining marks are preserved unchanged.

    This is stored as a SEPARATE index key alongside the original;
    it does not replace the original normalization.

    Examples (after normalize_text):
        'consultants' -> 'consultants'   (Latin: accent stripped)
        'consultants' stays unchanged     (Devanagari marks kept)
    """
    base = normalize_text(text)
    if not base:
        return base
    decomposed = list(unicodedata.normalize("NFD", base))
    result = []
    for i, c in enumerate(decomposed):
        if unicodedata.category(c) == 'Mn':
            # Find the nearest preceding non-combining character
            base_char = next(
                (decomposed[j] for j in range(i - 1, -1, -1)
                 if unicodedata.category(decomposed[j]) != 'Mn'),
                ''
            )
            # Strip only if the base character is ASCII alphabetic (Latin)
            if base_char and ord(base_char) < 128 and base_char.isalpha():
                continue  # drop this Latin diacritic
        result.append(c)
    return unicodedata.normalize("NFC", ''.join(result))


def _tokenize_normalized(norm_text):
    """Extract significant tokens from an already-normalized string."""
    if not norm_text:
        return []
    return [t for t in norm_text.split()
            if t not in NON_DISCRIMINATIVE and len(t) > 1]



# Words that are too common to be discriminative as blocking tokens.
# Includes stopwords, legal suffixes, and generic business descriptors.
NON_DISCRIMINATIVE = frozenset({
    # ── Stopwords ─────────────────────────────────────────────────────
    'the', 'a', 'an', 'of', 'in', 'at', 'on', 'to', 'for', 'by',
    'with', 'and', 'or', 'is', 'it', 'its', 'this', 'that', 'no',
    'not', 'are', 'was', 'were', 'been', 'be', 'being', 'have', 'has',
    'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
    'may', 'might', 'shall', 'can',
    # French / Hindi stopwords likely in business data
    'de', 'la', 'le', 'les', 'du', 'des', 'et', 'en', 'un', 'une',
    'au', 'aux', 'sur', 'par', 'pour', 'dans', 'avec', 'ke', 'ka',
    'ki', 'se', 'ko', 'me', 'hai', 'aur', 'ya',
    # ── Legal suffixes (English) ──────────────────────────────────────
    'inc', 'incorporated', 'corp', 'corporation', 'co', 'company',
    'ltd', 'limited', 'llc', 'llp', 'lp', 'plc',
    'pvt', 'private',
    # ── Legal suffixes (International) ────────────────────────────────
    'gmbh', 'ag', 'sarl', 'sas', 'sa', 'eurl', 'sci', 'snc', 'srl',
    'pty', 'bv', 'nv', 'ab', 'oy', 'as', 'kg', 'ohg', 'se',
    # ── Generic business descriptors ──────────────────────────────────
    'group', 'groups', 'holdings', 'holding',
    'enterprises', 'enterprise',
    'industries', 'industry', 'industrial',
    'associates', 'associate', 'partners', 'partner', 'partnership',
    'international', 'intl', 'global', 'worldwide',
    'services', 'service', 'solutions', 'solution',
    'technologies', 'technology', 'tech',
    'trading', 'traders', 'trader', 'trade',
    'systems', 'system',
    'consultants', 'consultant', 'consulting', 'consultancy',
    'management', 'ventures', 'venture',
    'products', 'product', 'productions', 'production',
    'communications', 'communication',
    'logistics', 'networks', 'network',
    'marketing', 'advertising', 'media',
    'agencies', 'agency',
    'corporate', 'professionals', 'professional',
    'foundation', 'institute', 'institution',
    'organization', 'organisation',
    # ── Common domain TLD tokens ──────────────────────────────────────
    'com', 'net', 'org', 'io', 'biz', 'info',
})

# Strategy identification bitmasks (for per-strategy tracking / reporting).
# Each strategy gets a unique bit; a candidate's bitmask is the OR of all
# strategies that found it.  Separate from ranking scores.
STRATEGY_BIT = {
    'S1': 0x40,   # 64  -- exact name + same country
    'S2': 0x20,   # 32  -- core name + same country
    'S4': 0x10,   # 16  -- address + first token + country
    'S6': 0x08,   #  8  -- domain squash
    'S3': 0x04,   #  4  -- token overlap
    'S5': 0x02,   #  2  -- cross-country exact name
    'S7': 0x01,   #  1  -- distinctive address only
}

# Pre-model ranking scores (additive, NOT bitmask).
# Used to select which candidates survive a max_total cap.
# S7 (distinctive address) ranks ABOVE S3 (token overlap) because a
# unique address pin+number combination is stronger evidence than
# sharing a common token with thousands of other businesses.
STRATEGY_SCORE = {
    'S1': 64,    # strongest
    'S2': 32,
    'S4': 16,
    'S7': 12,    # above S6 and S3 -- distinctive address is stronger than token overlap
    'S6':  8,
    'S3':  4,
    'S5':  2,    # weakest
}


def extract_significant_tokens(name):
    """
    Extract discriminative tokens from a business name.

    Normalizes, then removes stopwords, legal suffixes, generic
    business descriptors, and single-character tokens.
    """
    return _tokenize_normalized(normalize_text(name))


def make_core_name(sig_tokens):
    """
    Canonical 'core name' from significant tokens.

    Sorted + deduplicated → order-independent, suffix-independent.
    """
    if not sig_tokens:
        return ""
    return " ".join(sorted(set(sig_tokens)))


def extract_address_numbers(addr):
    """
    Extract sorted numeric sub-strings (≥ 2 digits) from an address.

    Captures house numbers, PIN / ZIP codes, suite numbers, etc.
    Returns a tuple (hashable) of at most 5 numbers.
    """
    if not addr or not isinstance(addr, str):
        return ()
    norm = normalize_text(addr)
    nums = sorted(set(re.findall(r'\b\d{2,}\b', norm)))
    return tuple(nums[:5])


# ═══════════════════════════════════════════════════════════════════════════
#  Memory Helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_memory_mb():
    """Return current process RSS in MB, or -1 if not measurable."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024**2
    except ImportError:
        pass
    # Windows fallback via ctypes
    try:
        import ctypes
        import ctypes.wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.wintypes.DWORD),
                ("PageFaultCount", ctypes.wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(pmc)
        h = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc),
                                                     pmc.cb):
            return pmc.WorkingSetSize / 1024**2
    except Exception:
        pass
    return -1


# ═══════════════════════════════════════════════════════════════════════════
#  Candidate Generator
# ═══════════════════════════════════════════════════════════════════════════

class CandidateGenerator:
    """
    Multi-strategy candidate generator for entity resolution.

    Build inverted indexes on Source 2/3 data, then query them with
    Source 1 records to produce candidate match sets.
    """

    # ── tunables ──────────────────────────────────────────────────────
    MAX_POSTING_LIST   = 50_000   # prune tokens with larger posting lists
    MAX_PER_STRATEGY   = 10_000   # cap candidates from any single strategy
    TOKEN_OVERLAP_MIN  = 2        # default: >= 2 shared sig-tokens
    PROGRESS_EVERY     = 500_000  # print progress every N records
    MAX_ADDR_POSTING   = 200      # prune addr-only keys with too many entries

    def __init__(self):
        # Index 1: (normalized_name, country) -> [eid ...]
        self.idx_name    = defaultdict(list)
        # Index 2: (core_name, country) -> [eid ...]
        self.idx_core    = defaultdict(list)
        # Index 3: (significant_token, country) -> [eid ...]
        self.idx_token   = defaultdict(list)
        # Index 4: (addr_numbers, first_sig_token, country) -> [eid ...]
        self.idx_addr    = defaultdict(list)
        # Index 5: (addr_numbers, country) -> [eid ...]  addr-only for cross-script
        self.idx_addrnum = defaultdict(list)

        self.known_countries  = set()
        self.n_indexed         = 0
        self.build_time_s      = 0.0
        self.peak_mem_mb       = -1
        # EIDs dropped during addrnum pruning (used to report S7 cap-miss rate)
        self.addrnum_pruned_eids: set = set()

    # ── building ──────────────────────────────────────────────────────

    def build_from_files(self, source_paths):
        """Stream S2/S3 TSV files and populate all four indexes."""
        t0 = time.time()

        for path in source_paths:
            fname = Path(path).name
            print(f"  Indexing {fname} …")
            with open(path, encoding="utf-8", newline="") as f:
                reader = csv.reader(f, delimiter="\t")
                header = next(reader)
                for row in reader:
                    eid     = sys.intern(row[0])
                    name    = row[1] if len(row) > 1 else ""
                    addr    = row[2] if len(row) > 2 else ""
                    country = row[3].strip() if len(row) > 3 else ""
                    self._index_one(eid, name, addr, country)
                    self.n_indexed += 1
                    if self.n_indexed % self.PROGRESS_EVERY == 0:
                        mem = get_memory_mb()
                        print(f"    {self.n_indexed:>12,}  "
                              f"({time.time()-t0:>6.1f}s, "
                              f"{mem:>.0f} MB)")

        self.build_time_s = time.time() - t0
        self.peak_mem_mb  = get_memory_mb()

        print(f"  Indexing done: {self.n_indexed:,} records  "
              f"({self.build_time_s:.1f}s, {self.peak_mem_mb:.0f} MB)")

        self._prune_token_index()
        self._prune_addrnum_index()
        self._print_stats()

    def _index_one(self, eid, name, addr, country):
        name = str(name).strip()
        addr = str(addr).strip()
        self.known_countries.add(country)

        norm_name    = normalize_text(name)
        norm_name_af = normalize_accent_fold(name)   # Latin-accent-folded form
        sig_tokens   = _tokenize_normalized(norm_name)
        sig_tokens_af = (_tokenize_normalized(norm_name_af)
                         if norm_name_af != norm_name else sig_tokens)
        core_name    = make_core_name(sig_tokens)
        core_name_af = (make_core_name(sig_tokens_af)
                        if norm_name_af != norm_name else core_name)
        addr_nums    = extract_address_numbers(addr)

        # 1 -- exact name + country (original and accent-folded)
        if norm_name:
            self.idx_name[(norm_name, country)].append(eid)
            if norm_name_af != norm_name:             # store separately, not replacing
                self.idx_name[(norm_name_af, country)].append(eid)

        # 2 -- core name + country (original and accent-folded)
        if core_name:
            self.idx_core[(core_name, country)].append(eid)
            if core_name_af != core_name:
                self.idx_core[(core_name_af, country)].append(eid)

        # 3 -- token inverted index (original tokens only)
        for tok in set(sig_tokens):
            self.idx_token[(tok, country)].append(eid)

        # 4 -- address numbers + first sig token + country
        if addr_nums and sig_tokens:
            self.idx_addr[(addr_nums, sig_tokens[0], country)].append(eid)

        # 5 -- address numbers only (cross-script fallback, no name needed)
        if addr_nums:
            self.idx_addrnum[(addr_nums, country)].append(eid)

    def _prune_token_index(self):
        """Remove non-discriminative tokens whose posting list is too big."""
        before = len(self.idx_token)
        to_drop = [k for k, v in self.idx_token.items()
                    if len(v) > self.MAX_POSTING_LIST]
        freed = sum(len(self.idx_token[k]) for k in to_drop)
        for k in to_drop:
            del self.idx_token[k]
        after = len(self.idx_token)
        print(f"  Pruned {len(to_drop)} oversized token keys "
              f"({freed:,} entries); {before} -> {after} keys")

    def _prune_addrnum_index(self):
        """Remove address-only posting lists that are too large.

        Large posting lists mean the address evidence is not distinctive
        (e.g. a busy industrial estate shared by hundreds of businesses).
        EIDs removed are recorded in self.addrnum_pruned_eids so the
        pilot can measure how many true pairs were lost to the cap.
        """
        before = len(self.idx_addrnum)
        to_drop = [k for k, v in self.idx_addrnum.items()
                    if len(v) > self.MAX_ADDR_POSTING]
        freed = 0
        for k in to_drop:
            self.addrnum_pruned_eids.update(self.idx_addrnum[k])
            freed += len(self.idx_addrnum[k])
            del self.idx_addrnum[k]
        after = len(self.idx_addrnum)
        print(f"  Pruned {len(to_drop)} oversized addrnum keys "
              f"({freed:,} entries, {len(self.addrnum_pruned_eids):,} unique EIDs); "
              f"{before} -> {after} keys")

    def _print_stats(self):
        print(f"\n  Index statistics:")
        print(f"    idx_name   : {len(self.idx_name):>10,} keys")
        print(f"    idx_core   : {len(self.idx_core):>10,} keys")
        print(f"    idx_token  : {len(self.idx_token):>10,} keys")
        print(f"    idx_addr   : {len(self.idx_addr):>10,} keys")
        print(f"    idx_addrnum: {len(self.idx_addrnum):>10,} keys")
        print(f"    Countries  : {sorted(self.known_countries)}")
        if self.idx_token:
            sizes = sorted(len(v) for v in self.idx_token.values())
            n = len(sizes)
            print(f"    Token postings -- "
                  f"median {sizes[n//2]}, "
                  f"p95 {sizes[int(n*0.95)]}, "
                  f"max {sizes[-1]}")

    # -- querying --

    def find_candidates(self, name, addr, country,
                        max_total=None, track_strategies=False):
        """
        Return candidate S2/S3 entity-IDs for one S1 record.

        Applies seven strategies (S1-S7) and unions results.

        Parameters
        ----------
        max_total : int or None
            Cap total candidates using STRATEGY_SCORE (additive, not bitmask).
            Priority: S1(64) > S2(32) > S4(16) > S7(12) > S6(8) > S3(4) > S5(2).
            Note: S7 (distinctive address) outranks S3 (token overlap) and S6.
            Tie-break: descending rank score, then ascending eid (deterministic).
        track_strategies : bool
            If True, return (candidates_set, strategy_counts_dict).
            strategy_counts_dict maps strategy name -> count of candidates
            in the final set that were found by that strategy.
        """
        name    = str(name).strip()
        addr    = str(addr).strip()
        country = str(country).strip()

        norm_name    = normalize_text(name)
        norm_name_af = normalize_accent_fold(name)   # Latin-accent-folded form
        sig_tokens   = _tokenize_normalized(norm_name)
        sig_tokens_af = (_tokenize_normalized(norm_name_af)
                         if norm_name_af != norm_name else sig_tokens)
        core_name    = make_core_name(sig_tokens)
        core_name_af = (make_core_name(sig_tokens_af)
                        if norm_name_af != norm_name else core_name)
        addr_nums    = extract_address_numbers(addr)
        cap          = self.MAX_PER_STRATEGY

        # Two separate dicts:
        #  bitmask: eid -> OR of STRATEGY_BIT values  (which strategies found it)
        #  rank:    eid -> sum of STRATEGY_SCORE values (for cap selection)
        bitmask = defaultdict(int)
        rank    = defaultdict(int)

        def add(hits, sname):
            b = STRATEGY_BIT[sname]
            s = STRATEGY_SCORE[sname]
            for eid in hits:
                bitmask[eid] |= b
                rank[eid]    += s

        # S1 -- exact name + same country (original + Latin-accent-folded)
        if norm_name:
            hits = self.idx_name.get((norm_name, country))
            if hits and len(hits) <= cap:
                add(hits, 'S1')
            if norm_name_af != norm_name:
                hits = self.idx_name.get((norm_name_af, country))
                if hits and len(hits) <= cap:
                    add(hits, 'S1')

        # S2 -- core name + same country (original + Latin-accent-folded)
        if core_name:
            hits = self.idx_core.get((core_name, country))
            if hits and len(hits) <= cap:
                add(hits, 'S2')
            if core_name_af != core_name:
                hits = self.idx_core.get((core_name_af, country))
                if hits and len(hits) <= cap:
                    add(hits, 'S2')

        # S4 -- address numbers + first sig-token + same country
        if addr_nums and sig_tokens:
            hits = self.idx_addr.get((addr_nums, sig_tokens[0], country))
            if hits and len(hits) <= cap:
                add(hits, 'S4')

        # S6 -- domain squash: concatenate sig_tokens (no spaces).
        # Handles cases where one source uses an abbreviated/concatenated form
        # (e.g. 'slmarine' in 'slmarine.com' matched from S1 'Sl Marine').
        # Does NOT fix completely-different-name cases; those need S7 (address).
        if len(sig_tokens) >= 2:
            squash = ''.join(sig_tokens)
            hits = self.idx_token.get((squash, country))
            if hits and len(hits) <= cap:
                add(hits, 'S6')
            for other in self.known_countries:
                if other != country:
                    hits = self.idx_token.get((squash, other))
                    if hits and len(hits) <= cap:
                        add(hits, 'S6')

        # S3 -- token overlap >= threshold + same country
        if sig_tokens:
            min_ov = 1 if len(sig_tokens) == 1 else self.TOKEN_OVERLAP_MIN
            overlap = Counter()
            for tok in set(sig_tokens):
                posting = self.idx_token.get((tok, country))
                if posting:
                    overlap.update(posting)
            tok_hits = {eid for eid, cnt in overlap.items() if cnt >= min_ov}
            if len(tok_hits) <= cap:
                add(tok_hits, 'S3')
            elif min_ov < len(sig_tokens):
                higher = min_ov + 1
                tok_hits = {eid for eid, cnt in overlap.items() if cnt >= higher}
                if len(tok_hits) <= cap:
                    add(tok_hits, 'S3')

        # S5 -- cross-country exact name
        if norm_name:
            for other in self.known_countries:
                if other != country:
                    hits = self.idx_name.get((norm_name, other))
                    if hits and len(hits) <= cap:
                        add(hits, 'S5')

        # S7 -- address numbers only + same country (cross-script fallback).
        # Distinctiveness requirement: >= 2 separate address numbers.
        # A single number (even a 6-digit PIN) is NOT distinctive because
        # a PIN code covers an entire postal area (potentially 1000s of businesses).
        # Requiring 2+ numbers (e.g. building no. + PIN, or floor + suite + PIN)
        # dramatically reduces false positives.
        # Posting list additionally capped at MAX_ADDR_POSTING.
        if addr_nums and len(addr_nums) >= 2:
            hits = self.idx_addrnum.get((addr_nums, country))
            if hits and len(hits) <= cap:
                add(hits, 'S7')

        # -- Apply max_total cap using additive ranking scores --
        # S7 (score=12) outranks S3 (score=4) and S6 (score=8) so that a
        # distinctive address match is never buried below thousands of weak
        # token-overlap hits.
        candidates = set(bitmask.keys())
        if max_total and len(candidates) > max_total:
            top = sorted(bitmask.keys(),
                         key=lambda e: (-rank[e], e))[:max_total]
            candidates = set(top)

        # -- Per-strategy tracking (uses bitmask, not rank) --
        if track_strategies:
            active_bitmask = {e: bitmask[e] for e in candidates}
            counts = {s: sum(1 for b in active_bitmask.values()
                             if b & STRATEGY_BIT[s])
                      for s in STRATEGY_BIT}
            return candidates, counts

        return candidates
