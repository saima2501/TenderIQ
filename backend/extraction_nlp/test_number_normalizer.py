"""
test_number_normalizer.py - Unit and Integration Tests for Number-Word Normalization
Part of Phase 8: Core NLP Reliability Improvements (Step 2A)

Tests coverage:
1. Basic number words (one, two, five, ten, twenty, thirty, fifty)
2. Compound numbers (twenty-five, one hundred and fifty)
3. Hundreds, thousands, millions, billions
4. Duration expressions (half a year, a decade, over a decade)
5. Comparison operators (at least, more than, over, up to, strictly less than)
6. Ambiguous phrases (several, many, collective) -> rejected as numbers
7. Existing numeric digit expressions remain unaffected
8. Preservation of original evidence sentence text and character offsets
9. Integration via EvidenceExtractor pipeline
"""

import unittest
from backend.extraction_nlp.number_normalizer import NumberWordNormalizer, NormalizedNumber
from backend.extraction_nlp.evidence_extractor import EvidenceExtractor, ExtractedEvidence


class TestNumberWordNormalizer(unittest.TestCase):
    """Unit tests for NumberWordNormalizer standalone module."""

    def setUp(self):
        self.normalizer = NumberWordNormalizer()

    # -------------------------------------------------------------------------
    # 1. Basic Number Words
    # -------------------------------------------------------------------------
    def test_basic_number_words(self):
        cases = {
            "one": 1.0,
            "two": 2.0,
            "five": 5.0,
            "ten": 10.0,
            "twenty": 20.0,
            "thirty": 30.0,
            "fifty": 50.0,
        }
        for word, expected in cases.items():
            parsed = self.normalizer.parse_text_number(word)
            self.assertEqual(parsed, expected, f"Failed parsing basic word '{word}'")

    # -------------------------------------------------------------------------
    # 2. Compound Numbers
    # -------------------------------------------------------------------------
    def test_compound_numbers(self):
        cases = {
            "twenty five": 25.0,
            "twenty-five": 25.0,
            "thirty two": 32.0,
            "one hundred and fifty": 150.0,
            "one hundred fifty": 150.0,
            "two hundred and seventy five": 275.0,
        }
        for phrase, expected in cases.items():
            parsed = self.normalizer.parse_text_number(phrase)
            self.assertEqual(parsed, expected, f"Failed parsing compound '{phrase}'")

    # -------------------------------------------------------------------------
    # 3. Hundreds, Thousands, Millions, Billions
    # -------------------------------------------------------------------------
    def test_large_scales(self):
        cases = {
            "two hundred": 200.0,
            "one thousand": 1000.0,
            "five thousand": 5000.0,
            "five million": 5_000_000.0,
            "one billion": 1_000_000_000.0,
            "half a million": 500_000.0,
            "quarter of a million": 250_000.0,
        }
        for phrase, expected in cases.items():
            parsed = self.normalizer.parse_text_number(phrase)
            self.assertEqual(parsed, expected, f"Failed parsing scale '{phrase}'")

    # -------------------------------------------------------------------------
    # 4. Duration Expressions
    # -------------------------------------------------------------------------
    def test_duration_expressions(self):
        # 4a. A decade -> 10 years, exact
        text = "The company has been established for a decade."
        norms = self.normalizer.extract_normalized_numbers(text)
        self.assertEqual(len(norms), 1)
        self.assertEqual(norms[0].value, 10.0)
        self.assertEqual(norms[0].unit, "years")
        self.assertEqual(norms[0].operator, "==")
        self.assertTrue(norms[0].is_exact)

        # 4b. Half a year -> 6 months
        text2 = "Maintenance is performed every half a year."
        norms2 = self.normalizer.extract_normalized_numbers(text2)
        self.assertEqual(len(norms2), 1)
        self.assertEqual(norms2[0].value, 6.0)
        self.assertEqual(norms2[0].unit, "months")
        self.assertEqual(norms2[0].operator, "==")

        # 4c. Two decades -> 20 years
        text3 = "Over two decades of proven experience."
        norms3 = self.normalizer.extract_normalized_numbers(text3)
        self.assertEqual(len(norms3), 1)
        self.assertEqual(norms3[0].value, 20.0)
        self.assertEqual(norms3[0].unit, "years")
        self.assertEqual(norms3[0].operator, ">")
        self.assertFalse(norms3[0].is_exact)

    # -------------------------------------------------------------------------
    # 5. Comparison Operators
    # -------------------------------------------------------------------------
    def test_comparison_operators(self):
        # 5a. "over a decade" -> value 10.0, operator ">", NOT exact
        text1 = "Bidder has operated for over a decade."
        n1 = self.normalizer.extract_normalized_numbers(text1)
        self.assertEqual(len(n1), 1)
        self.assertEqual(n1[0].value, 10.0)
        self.assertEqual(n1[0].unit, "years")
        self.assertEqual(n1[0].operator, ">")
        self.assertFalse(n1[0].is_exact, "'over a decade' must not be exact 10 years")

        # 5b. "at least five years" -> operator ">="
        text2 = "Requires at least five years of operational experience."
        n2 = self.normalizer.extract_normalized_numbers(text2)
        self.assertEqual(len(n2), 1)
        self.assertEqual(n2[0].value, 5.0)
        self.assertEqual(n2[0].unit, "years")
        self.assertEqual(n2[0].operator, ">=")
        self.assertFalse(n2[0].is_exact)

        # 5c. "more than ten years" -> operator ">"
        text3 = "Offering more than ten years warranty."
        n3 = self.normalizer.extract_normalized_numbers(text3)
        self.assertEqual(len(n3), 1)
        self.assertEqual(n3[0].value, 10.0)
        self.assertEqual(n3[0].unit, "years")
        self.assertEqual(n3[0].operator, ">")

        # 5d. "up to thirty months" -> operator "<="
        text4 = "Execution takes up to thirty months."
        n4 = self.normalizer.extract_normalized_numbers(text4)
        self.assertEqual(len(n4), 1)
        self.assertEqual(n4[0].value, 30.0)
        self.assertEqual(n4[0].unit, "months")
        self.assertEqual(n4[0].operator, "<=")

        # 5e. "strictly less than two hours" -> operator "<"
        text5 = "Incident response is strictly less than two hours."
        n5 = self.normalizer.extract_normalized_numbers(text5)
        self.assertEqual(len(n5), 1)
        self.assertEqual(n5[0].value, 2.0)
        self.assertEqual(n5[0].unit, "hours")
        self.assertEqual(n5[0].operator, "<")

    # -------------------------------------------------------------------------
    # 6. Ambiguous Phrases (Must NOT parse into definite numbers)
    # -------------------------------------------------------------------------
    def test_ambiguous_phrases_rejected(self):
        ambiguous_cases = [
            "We possess several years of cloud deployment experience.",
            "Our firm has executed many projects across the country.",
            "The leadership has decades of collective domain knowledge.",
            "Substantial financial solvency backed by extensive lines of credit.",
            "Multiple implementations in the public sector.",
        ]
        for sentence in ambiguous_cases:
            self.assertTrue(
                self.normalizer.is_ambiguous_phrase(sentence),
                f"Sentence should be flagged as ambiguous: {sentence}"
            )
            # Ensure decade idiom is NOT extracted from 'decades of collective'
            if "decades of collective" in sentence.lower():
                norms = self.normalizer.extract_normalized_numbers(sentence)
                self.assertEqual(
                    len(norms), 0,
                    f"'decades of collective' should not be converted to a number: {norms}"
                )

    # -------------------------------------------------------------------------
    # 7. Units and Word-Number Binding
    # -------------------------------------------------------------------------
    def test_units_and_word_numbers(self):
        # 7a. Gbps
        t1 = "Dedicated link offering one gigabit per second throughput."
        n1 = self.normalizer.extract_normalized_numbers(t1)
        self.assertEqual(len(n1), 1)
        self.assertEqual(n1[0].value, 1.0)
        self.assertEqual(n1[0].unit, "Gbps")

        # 7b. Headcount / Count
        t2 = "Team includes one hundred and fifty employees on site."
        n2 = self.normalizer.extract_normalized_numbers(t2)
        self.assertEqual(len(n2), 1)
        self.assertEqual(n2[0].value, 150.0)
        self.assertEqual(n2[0].unit, "count")
        self.assertEqual(n2[0].parameter_hint, "headcount")

        # 7c. USD Millions
        t3 = "Annual revenue reached half a million USD."
        n3 = self.normalizer.extract_normalized_numbers(t3)
        self.assertEqual(len(n3), 1)
        self.assertAlmostEqual(n3[0].value, 0.5, places=2)
        self.assertEqual(n3[0].unit, "USD Million")

    # -------------------------------------------------------------------------
    # 8. Character Offsets and Provenance
    # -------------------------------------------------------------------------
    def test_character_offsets_and_substring_preservation(self):
        text = "The vendor provides at least five years warranty coverage."
        norms = self.normalizer.extract_normalized_numbers(text)
        self.assertEqual(len(norms), 1)
        norm = norms[0]
        # Character offsets must match the substring in original text
        extracted_substring = text[norm.start_char:norm.end_char]
        self.assertEqual(norm.raw_text, extracted_substring)
        self.assertIn("five years", norm.raw_text)


class TestEvidenceExtractorIntegration(unittest.TestCase):
    """Integration tests verifying NumberWordNormalizer inside EvidenceExtractor."""

    def setUp(self):
        self.extractor = EvidenceExtractor(use_spacy=False)

    def test_evidence_extractor_extracts_word_numbers(self):
        text = "The bidder has been in continuous operation for over a decade in enterprise cloud hosting."
        evds = self.extractor.extract_from_text(text)
        self.assertEqual(len(evds), 1)
        evd = evds[0]

        # 1. Original sentence must be 100% preserved
        self.assertEqual(evd.original_sentence, text)

        # 2. Metric should be extracted
        self.assertGreaterEqual(len(evd.numeric_values), 1)
        metric = evd.numeric_values[0]
        self.assertEqual(metric["value"], 10.0)
        self.assertEqual(metric["unit"], "years")
        self.assertEqual(metric["operator"], ">")
        self.assertEqual(metric["parameter_hint"], "corporate_experience")
        self.assertTrue(metric.get("is_normalized_word", False))

    def test_evidence_extractor_mixed_digits_and_words(self):
        text = "We provide 10 Gbps core capacity and at least five years full OEM warranty."
        evds = self.extractor.extract_from_text(text)
        self.assertEqual(len(evds), 1)
        evd = evds[0]

        # Both the digit metric (10 Gbps) and word metric (at least five years) must be captured
        hints = [m["parameter_hint"] for m in evd.numeric_values]
        units = [m["unit"] for m in evd.numeric_values]
        values = [m["value"] for m in evd.numeric_values]

        self.assertIn("data_throughput", hints)
        self.assertIn("Gbps", units)
        self.assertIn(10.0, values)

        self.assertIn("years", units)
        self.assertIn(5.0, values)

    def test_evidence_extractor_uncertainty_flag_for_ambiguous(self):
        text = "Our organization possesses decades of collective domain knowledge in software engineering."
        evds = self.extractor.extract_from_text(text)
        self.assertEqual(len(evds), 1)
        evd = evds[0]

        # Must flag uncertainty and NOT create false definite 10-year metric
        self.assertTrue(evd.uncertainty_flag)
        self.assertIn("collective", evd.review_notes.lower())
        # Should not extract a numeric value for collective phrase
        self.assertEqual(len(evd.numeric_values), 0)


if __name__ == "__main__":
    unittest.main()
