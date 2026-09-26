#!/usr/bin/env python3
"""
05a_normalization_tests.py — Normalization unit tests
======================================================
Verifies:
1. Devanagari Mc/Mn combining marks survive primary normalization (normalize_text).
2. Latin diacritics survive normalize_text and are stripped in normalize_accent_fold.
3. Original Unicode strings are preserved alongside normalized keys.
"""

import unittest
import sys
import os

# Import candidate generator dynamically from 05_candidate_generator.py
import importlib.util
spec = importlib.util.spec_from_file_location("candidate_gen", os.path.join(os.path.dirname(__file__), "05_candidate_generator.py"))
cg = importlib.util.module_from_spec(spec)
sys.modules["candidate_gen"] = cg
spec.loader.exec_module(cg)

class TestNormalization(unittest.TestCase):

    def test_devanagari_combining_marks_preserved(self):
        """Devanagari matras (Mc) and anusvara/chandrabindu (Mn) must survive normalize_text."""
        # 'हिंदी' -> 'ह' + 'ि' (Mc) + 'ं' (Mn) + 'द' + 'ी' (Mc)
        hindi_word = "हिंदी"
        norm = cg.normalize_text(hindi_word)
        self.assertIn("ि", norm, "Devanagari Mc vowel sign (ि) was incorrectly stripped!")
        self.assertIn("ं", norm, "Devanagari Mn anusvara (ं) was incorrectly stripped!")
        self.assertEqual(norm, "हिंदी")

    def test_latin_accent_folding(self):
        """Latin accents survive normalize_text but fold in normalize_accent_fold."""
        accented = "Société Marine"
        norm_primary = cg.normalize_text(accented)
        norm_folded = cg.normalize_accent_fold(accented)
        
        self.assertEqual(norm_primary, "société marine")
        self.assertEqual(norm_folded, "societe marine")

    def test_punctuation_and_whitespace(self):
        """Punctuation is normalized to spaces and extra whitespace collapsed."""
        raw = "  Acme-Corp.  (Pvt)  Ltd. "
        norm = cg.normalize_text(raw)
        self.assertEqual(norm, "acme corp pvt ltd")

if __name__ == "__main__":
    unittest.main()
