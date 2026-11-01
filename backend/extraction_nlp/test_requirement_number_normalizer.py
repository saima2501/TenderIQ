"""
test_requirement_number_normalizer.py - Unit and Integration Tests for Requirement-Side Number-Word Normalization
Part of Phase 8: Core NLP Reliability Improvements (Step 2A.1)

Tests coverage:
1. Requirement-side basic number words
2. Compound number expressions
3. Written financial thresholds
4. Written durations
5. Written headcount requirements
6. Comparison operators
7. Existing digit-based requirements
8. Duplicate extraction prevention
9. Original requirement text preservation
10. End-to-end requirement versus evidence comparison (Examples A, B, C, D + bounded claims)
11. Ambiguous requirement handling
12. Project Manager experience sub-requirement with written numbers
"""

import unittest
from backend.extraction_nlp.requirement_extractor import RequirementExtractor
from backend.extraction_nlp.evidence_extractor import EvidenceExtractor
from backend.scoring.numeric_comparator import NumericComparator


class TestRequirementNumberNormalizer(unittest.TestCase):
    """Unit and integration tests for requirement-side number-word normalization."""

    def setUp(self):
        self.req_extractor = RequirementExtractor(use_spacy=False)
        self.evd_extractor = EvidenceExtractor(use_spacy=False)
        self.comparator = NumericComparator()

    # -------------------------------------------------------------------------
    # 1. Basic Number Words
    # -------------------------------------------------------------------------
    def test_requirement_basic_number_words(self):
        clause = "The bidder must have at least five years of experience in enterprise systems."
        reqs = self.req_extractor.extract_from_text(clause)
        self.assertEqual(len(reqs), 1)
        req = reqs[0]
        self.assertEqual(req.value, 5.0)
        self.assertEqual(req.unit, "years")
        self.assertEqual(req.operator, ">=")
        self.assertFalse(req.uncertainty_flag)

    # -------------------------------------------------------------------------
    # 2. Compound Number Expressions
    # -------------------------------------------------------------------------
    def test_requirement_compound_number_expressions(self):
        clause = "The contractor must deploy at least one hundred and fifty employees on site."
        reqs = self.req_extractor.extract_from_text(clause)
        self.assertEqual(len(reqs), 1)
        req = reqs[0]
        self.assertEqual(req.value, 150.0)
        self.assertEqual(req.unit, "count")
        self.assertEqual(req.operator, ">=")
        self.assertEqual(req.parameter, "organizational headcount")

    # -------------------------------------------------------------------------
    # 3. Written Financial Thresholds
    # -------------------------------------------------------------------------
    def test_requirement_written_financial_thresholds(self):
        clause = "The firm must demonstrate a minimum annual turnover of one million USD."
        reqs = self.req_extractor.extract_from_text(clause)
        self.assertEqual(len(reqs), 1)
        req = reqs[0]
        self.assertEqual(req.value, 1.0)
        self.assertEqual(req.unit, "USD Million")
        self.assertEqual(req.operator, ">=")
        self.assertIn("turnover", req.parameter.lower())

    # -------------------------------------------------------------------------
    # 4. Written Durations
    # -------------------------------------------------------------------------
    def test_requirement_written_durations(self):
        clause = "The supplier shall provide a warranty period of up to thirty months."
        reqs = self.req_extractor.extract_from_text(clause)
        self.assertEqual(len(reqs), 1)
        req = reqs[0]
        self.assertEqual(req.value, 30.0)
        self.assertEqual(req.unit, "months")
        self.assertEqual(req.operator, "<=")
        self.assertIn("warranty", req.parameter.lower())

    # -------------------------------------------------------------------------
    # 5. Written Headcount Requirements
    # -------------------------------------------------------------------------
    def test_requirement_written_headcount(self):
        clause = "The bidder must maintain at least two hundred engineers in active employment."
        reqs = self.req_extractor.extract_from_text(clause)
        self.assertEqual(len(reqs), 1)
        req = reqs[0]
        self.assertEqual(req.value, 200.0)
        self.assertEqual(req.unit, "count")
        self.assertEqual(req.operator, ">=")
        self.assertEqual(req.parameter, "dedicated certified engineers")

    # -------------------------------------------------------------------------
    # 6. Comparison Operators
    # -------------------------------------------------------------------------
    def test_requirement_comparison_operators(self):
        cases = [
            ("The bidder must possess at least five years of experience.", 5.0, ">="),
            ("The bidder must possess more than ten years of experience.", 10.0, ">"),
            ("The bidder must possess over a decade of operating experience.", 10.0, ">"),
            ("Execution time must not exceed two years.", 2.0, "<="),
        ]
        for clause, expected_val, expected_op in cases:
            reqs = self.req_extractor.extract_from_text(clause)
            self.assertEqual(len(reqs), 1)
            self.assertEqual(reqs[0].value, expected_val)
            self.assertEqual(reqs[0].operator, expected_op)

    # -------------------------------------------------------------------------
    # 7. Existing Digit-Based Requirements
    # -------------------------------------------------------------------------
    def test_existing_digit_based_requirements_preserved(self):
        clause = "The bidder must have at least 5 years of operational experience and USD 10.0 Million turnover."
        reqs = self.req_extractor.extract_from_text(clause)
        self.assertGreaterEqual(len(reqs), 1)
        # Should extract the financial or duration metric cleanly using existing digit pass
        self.assertIn(reqs[0].value, [5.0, 10.0])
        self.assertEqual(reqs[0].operator, ">=")

    # -------------------------------------------------------------------------
    # 8. Duplicate Extraction Prevention
    # -------------------------------------------------------------------------
    def test_duplicate_extraction_prevention(self):
        clause = "The contractor must guarantee at least 1 Gbps throughput across all nodes."
        reqs = self.req_extractor.extract_from_text(clause)
        self.assertEqual(len(reqs), 1)
        self.assertEqual(reqs[0].value, 1.0)
        self.assertEqual(reqs[0].unit, "Gbps")

    # -------------------------------------------------------------------------
    # 9. Original Requirement Text Preservation
    # -------------------------------------------------------------------------
    def test_original_requirement_text_preservation(self):
        clause = "The bidder must have at least five years of experience in enterprise development."
        reqs = self.req_extractor.extract_from_text(clause)
        self.assertEqual(len(reqs), 1)
        self.assertEqual(reqs[0].original_text, clause)

    # -------------------------------------------------------------------------
    # 10. End-to-End Requirement vs. Evidence Comparison
    # -------------------------------------------------------------------------
    def test_example_a_experience_satisfied(self):
        # Example A: Req: "At least five years of experience." Evd: "Our company has seven years of experience."
        req_text = "The bidder must possess at least five years of experience."
        evd_text = "Our company has seven years of experience."
        req = self.req_extractor.extract_from_text(req_text)[0]
        evd = self.evd_extractor.extract_from_text(evd_text)[0]
        m = evd.numeric_values[0]

        res = self.comparator.compare(
            required_val=req.value,
            required_unit=req.unit,
            operator=req.operator,
            offered_val=m["value"],
            offered_unit=m["unit"],
            offered_operator=m.get("operator", "==")
        )
        self.assertEqual(res.status, "SATISFIED")
        self.assertTrue(res.is_applicable)

    def test_example_b_experience_violated(self):
        # Example B: Req: "More than five years of experience." Evd: "Our company has exactly five years of experience."
        req_text = "The bidder must possess more than five years of experience."
        evd_text = "Our company has exactly five years of experience."
        req = self.req_extractor.extract_from_text(req_text)[0]
        evd = self.evd_extractor.extract_from_text(evd_text)[0]
        m = evd.numeric_values[0]

        res = self.comparator.compare(
            required_val=req.value,
            required_unit=req.unit,
            operator=req.operator,
            offered_val=m["value"],
            offered_unit=m["unit"],
            offered_operator=m.get("operator", "==")
        )
        self.assertEqual(res.status, "VIOLATED")

    def test_example_c_turnover_satisfied(self):
        # Example C: Req: "Minimum annual turnover of one million USD." Evd: "Annual turnover of two million USD."
        req_text = "The bidder must demonstrate a minimum annual turnover of one million USD."
        evd_text = "Annual turnover of two million USD."
        req = self.req_extractor.extract_from_text(req_text)[0]
        evd = self.evd_extractor.extract_from_text(evd_text)[0]
        m = evd.numeric_values[0]

        res = self.comparator.compare(
            required_val=req.value,
            required_unit=req.unit,
            operator=req.operator,
            offered_val=m["value"],
            offered_unit=m["unit"],
            offered_operator=m.get("operator", "==")
        )
        self.assertEqual(res.status, "SATISFIED")

    def test_example_d_headcount_satisfied(self):
        # Example D: Req: "At least one hundred and fifty employees." Evd: "The company employs 175 people."
        req_text = "The vendor must employ at least one hundred and fifty employees."
        evd_text = "The company employs 175 people."
        req = self.req_extractor.extract_from_text(req_text)[0]
        evd = self.evd_extractor.extract_from_text(evd_text)[0]
        m = evd.numeric_values[0]

        res = self.comparator.compare(
            required_val=req.value,
            required_unit=req.unit,
            operator=req.operator,
            offered_val=m["value"],
            offered_unit=m["unit"],
            offered_operator=m.get("operator", "==")
        )
        self.assertEqual(res.status, "SATISFIED")

    def test_bounded_evidence_uncertain_behavior(self):
        # Upper bound in evidence cannot guarantee minimum requirement threshold
        req_text = "The bidder must possess at least five years of experience."
        evd_text = "Our company has up to five years of experience."
        req = self.req_extractor.extract_from_text(req_text)[0]
        evd = self.evd_extractor.extract_from_text(evd_text)[0]
        m = evd.numeric_values[0]

        res = self.comparator.compare(
            required_val=req.value,
            required_unit=req.unit,
            operator=req.operator,
            offered_val=m["value"],
            offered_unit=m["unit"],
            offered_operator=m.get("operator", "==")
        )
        self.assertEqual(res.status, "UNCERTAIN")
        self.assertIn("cannot contractually guarantee", res.reasoning)

    # -------------------------------------------------------------------------
    # 11. Ambiguous Requirement Handling
    # -------------------------------------------------------------------------
    def test_ambiguous_requirements_produce_no_numeric_values(self):
        ambiguous_clauses = [
            "The bidder must have several years of experience in system integration.",
            "The vendor must have completed many projects across regions.",
            "The leadership must possess decades of collective experience.",
            "The company must possess substantial financial resources.",
        ]
        for clause in ambiguous_clauses:
            reqs = self.req_extractor.extract_from_text(clause)
            self.assertEqual(len(reqs), 1)
            req = reqs[0]
            self.assertIsNone(req.value, f"Should not invent numeric value for: {clause}")
            self.assertEqual(req.operator, "NONE")
            self.assertTrue(req.uncertainty_flag)

    # -------------------------------------------------------------------------
    # 12. Project Manager Experience Clause with Written Words
    # -------------------------------------------------------------------------
    def test_project_manager_experience_written_words(self):
        clause = "Section 4.3: The proposed Project Manager must possess a valid PMP certification and a minimum of eight years of relevant experience."
        reqs = self.req_extractor.extract_from_text(clause)
        # Should decompose into 2 requirements: certification and experience
        self.assertEqual(len(reqs), 2)
        pm_cert = reqs[0]
        pm_exp = reqs[1]
        self.assertEqual(pm_cert.parameter, "Project Manager PMP/PRINCE2 certification")
        self.assertEqual(pm_exp.parameter, "Project Manager relevant experience")
        self.assertEqual(pm_exp.value, 8.0)
        self.assertEqual(pm_exp.unit, "years")
        self.assertEqual(pm_exp.operator, ">=")


if __name__ == "__main__":
    unittest.main()
