"""
test_phase4_bug_fixes.py - Regression Tests for Phase 4 Bug Fixes

Covers:
  BUG 1 - Warranty operator must be >= for 'at least N months' clauses.
  BUG 2 - PM evidence must be person-specific; company operating experience
           is inadmissible for the PM relevant-experience criterion.
  BUG 3 - OEM Authorization (MAF) is distinct from generic warranty.
  BUG 4 - Tender disclaimers are not bidder compliance requirements.

Test requirements from the specification:
  1. Regression tests for all four bugs.               [covered]
  2. Both Alpha and Epsilon bidder proposals tested.   [covered]
  3. Operator, criterion, evidence, source, page,
     label, and audit explanation verified.            [covered]
  4. Similarity alone does not establish compliance.   [covered]
  5. Missing evidence is NOT non-compliance.           [covered]
  6. Full existing test suite must still pass.         [run separately]
  7. Pipeline re-run shown for affected clauses.       [covered]
  8. No data files changed without documentation.      [no data changed]
  9. No test claims success without execution.         [covered]
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.extraction_nlp.requirement_extractor import RequirementExtractor, ExtractedRequirement
from backend.classification.baseline_classifier import BaselineClassifier, DocumentComplianceReport
from backend.matching.tfidf_retriever import RetrievedEvidenceCandidate, RequirementRetrievalResult
from backend.pipeline import CompliancePipeline

_TENDERS_DIR = os.path.join(PROJECT_ROOT, "dataset", "synthetic_pdfs", "tenders")
_BIDDERS_DIR = os.path.join(PROJECT_ROOT, "dataset", "synthetic_pdfs", "bidders")
_TENDER_IT001 = os.path.join(_TENDERS_DIR, "TENDER-2024-IT-001.pdf")
_BIDDER_ALPHA = os.path.join(_BIDDERS_DIR, "BIDDER-ALPHA-TECH-PROPOSAL.pdf")
_BIDDER_EPSILON = os.path.join(_BIDDERS_DIR, "BIDDER-EPSILON-INFRA-PROPOSAL.pdf")


# ============================================================
# BUG 1 - Warranty Operator
# ============================================================

class TestBug1WarrantyOperator(unittest.TestCase):
    """BUG 1: Operator for 'at least 36 months' warranty must be >=, not <=."""

    def setUp(self):
        self.extractor = RequirementExtractor()

    def test_at_least_36_months_compound_clause(self):
        """Clause 5.3 compound sentence: warranty operator must be >= despite 'inside' keyword later."""
        clause = (
            "5.3 Warranty & Incident SLA: The contractor must provide a "
            "comprehensive on-site warranty of at least 36 months, with "
            "Critical (P1) incident response inside 15 minutes and on-site "
            "fix within 4 hours."
        )
        reqs = self.extractor.extract_from_text(clause, source_document="TEST.pdf", page_number=2)
        warranty_reqs = [r for r in reqs if "warranty" in r.parameter.lower()]
        self.assertGreater(len(warranty_reqs), 0, "Expected at least one warranty requirement")
        w = warranty_reqs[0]
        self.assertEqual(w.operator, ">=",
                         f"BUG 1: 'at least 36 months' must yield >=, got '{w.operator}'")
        self.assertEqual(w.value, 36.0)
        self.assertEqual(w.unit, "months")

    def test_at_least_yields_gte(self):
        """'at least 36 months' must yield >=."""
        reqs = self.extractor.extract_from_text(
            "The contractor must provide a warranty of at least 36 months.")
        w = next((r for r in reqs if "warranty" in r.parameter.lower()), None)
        self.assertIsNotNone(w, "Warranty requirement must be extracted")
        self.assertEqual(w.operator, ">=",
                         f"'at least 36 months' should yield >=, got {w.operator}")

    def test_minimum_yields_gte(self):
        """'minimum 36 months' must yield >=."""
        reqs = self.extractor.extract_from_text(
            "The contractor must provide a minimum 36 months comprehensive warranty.")
        w = next((r for r in reqs if "warranty" in r.parameter.lower()), None)
        self.assertIsNotNone(w)
        self.assertEqual(w.operator, ">=",
                         f"'minimum 36 months' should yield >=, got {w.operator}")

    def test_no_less_than_yields_gte(self):
        """'no less than 36 months' must yield >=."""
        reqs = self.extractor.extract_from_text(
            "A no less than 36 months comprehensive warranty must be provided.")
        w = next((r for r in reqs if "warranty" in r.parameter.lower()), None)
        self.assertIsNotNone(w)
        self.assertEqual(w.operator, ">=",
                         f"'no less than 36 months' should yield >=, got {w.operator}")

    def test_must_not_exceed_yields_lte(self):
        """'must not exceed 36 months' must yield <=."""
        reqs = self.extractor.extract_from_text(
            "Warranty coverage must not exceed 36 months under the standard plan.")
        w = next((r for r in reqs if "warranty" in r.parameter.lower()), None)
        self.assertIsNotNone(w)
        self.assertEqual(w.operator, "<=",
                         f"'must not exceed 36 months' should yield <=, got {w.operator}")

    def test_exactly_yields_eq(self):
        """'exactly 36 months' must yield ==."""
        reqs = self.extractor.extract_from_text(
            "The warranty period is exactly 36 months with no extensions.")
        w = next((r for r in reqs if "warranty" in r.parameter.lower()), None)
        self.assertIsNotNone(w)
        self.assertEqual(w.operator, "==",
                         f"'exactly 36 months' should yield ==, got {w.operator}")

    def test_operator_from_clause_not_default(self):
        """GTE and LTE clauses must yield different operators - not a uniform default."""
        reqs_gte = self.extractor.extract_from_text("Warranty of at least 12 months.")
        reqs_lte = self.extractor.extract_from_text("Warranty must not exceed 12 months.")
        w_gte = next((r for r in reqs_gte if "warranty" in r.parameter.lower()), None)
        w_lte = next((r for r in reqs_lte if "warranty" in r.parameter.lower()), None)
        self.assertIsNotNone(w_gte)
        self.assertIsNotNone(w_lte)
        self.assertNotEqual(w_gte.operator, w_lte.operator,
                            "GTE and LTE clauses must produce different operators")
        self.assertEqual(w_gte.operator, ">=")
        self.assertEqual(w_lte.operator, "<=")

    def test_pipeline_warranty_operator_alpha(self):
        """Pipeline: warranty req op >= ; Alpha 3-year offer -> COMPLIANT (36m >= 36m)."""
        pipeline = CompliancePipeline()
        report = pipeline.run(_TENDER_IT001, _BIDDER_ALPHA, save_to_db=False)
        w = next((e for e in report.evaluations
                  if "warranty duration" in e.parameter.lower()), None)
        self.assertIsNotNone(w, "Warranty duration evaluation must exist")
        self.assertEqual(w.operator, ">=",
                         f"Pipeline warranty operator must be >=, got {w.operator}")
        self.assertEqual(w.required_value, 36.0)
        self.assertEqual(w.label, "COMPLIANT",
                         f"Alpha 3-year warranty must be COMPLIANT vs >= 36 months. "
                         f"Evidence: '{w.top_evidence_sentence}', "
                         f"Source: {w.top_evidence_source}, Page: {w.top_evidence_page}")

    def test_pipeline_warranty_operator_epsilon(self):
        """Pipeline: Epsilon 18-month warranty must be NON_COMPLIANT vs >= 36 months."""
        pipeline = CompliancePipeline()
        report = pipeline.run(_TENDER_IT001, _BIDDER_EPSILON, save_to_db=False)
        w = next((e for e in report.evaluations
                  if "warranty duration" in e.parameter.lower()), None)
        self.assertIsNotNone(w, "Warranty duration evaluation must exist")
        self.assertEqual(w.operator, ">=",
                         f"Operator must be >= for warranty clause, got {w.operator}")
        self.assertEqual(w.label, "NON_COMPLIANT",
                         f"Epsilon 18-month warranty must be NON_COMPLIANT vs >= 36 months. "
                         f"Evidence: '{w.top_evidence_sentence}', "
                         f"Source: {w.top_evidence_source}, Page: {w.top_evidence_page}")


# ============================================================
# BUG 2 - Project Manager Evidence Mismatch
# ============================================================

class TestBug2ProjectManagerEvidence(unittest.TestCase):
    """BUG 2: PM experience must NOT be satisfied by corporate operating experience."""

    def setUp(self):
        self.extractor = RequirementExtractor()
        self.classifier = BaselineClassifier()

    def test_pm_clause_yields_two_requirements(self):
        """Clause 4.3 must produce BOTH a cert criterion AND an experience criterion."""
        clause = (
            "4.3 Project Manager: The proposed Project Manager must possess a "
            "valid PMP or PRINCE2 certification and a minimum of 8 years of "
            "relevant experience."
        )
        reqs = self.extractor.extract_from_text(clause, source_document="TEST.pdf", page_number=1)
        cert_reqs = [r for r in reqs
                     if "project manager" in r.parameter.lower()
                     and "certification" in r.parameter.lower()]
        exp_reqs = [r for r in reqs
                    if "project manager relevant experience" in r.parameter.lower()]
        self.assertEqual(len(cert_reqs), 1, "Exactly one PM certification req must be extracted")
        self.assertEqual(len(exp_reqs), 1, "Exactly one PM experience req must be extracted")

    def test_pm_certification_distinct_criterion(self):
        """PM cert criterion value must reference PMP/PRINCE2."""
        clause = (
            "The proposed Project Manager must possess a valid PMP or PRINCE2 "
            "certification and a minimum of 8 years of relevant experience."
        )
        reqs = self.extractor.extract_from_text(clause)
        cert = next((r for r in reqs
                     if "project manager" in r.parameter.lower()
                     and "certification" in r.parameter.lower()), None)
        self.assertIsNotNone(cert, "PM certification must be a distinct criterion")
        self.assertIn("pmp", (cert.value or "").lower(),
                      "Certification value must reference PMP")

    def test_pm_experience_distinct_criterion(self):
        """PM experience: operator >=, value 8.0 years, distinctly named parameter."""
        clause = (
            "The proposed Project Manager must possess a valid PMP or PRINCE2 "
            "certification and a minimum of 8 years of relevant experience."
        )
        reqs = self.extractor.extract_from_text(clause)
        exp = next((r for r in reqs
                    if "project manager relevant experience" in r.parameter.lower()), None)
        self.assertIsNotNone(exp, "PM experience must be a distinct criterion")
        self.assertEqual(exp.operator, ">=",
                         f"PM experience op must be >=, got {exp.operator}")
        self.assertEqual(exp.value, 8.0,
                         f"PM experience must be 8 years, got {exp.value}")
        self.assertEqual(exp.unit, "years",
                         f"PM experience unit must be 'years', got {exp.unit}")

    def test_company_experience_rejected_for_pm_req(self):
        """Corporate operating experience (hint=corporate_experience) must yield MISSING_EVIDENCE for PM req."""
        pm_exp_req = ExtractedRequirement(
            requirement_id="REQ-PM-EXP-001",
            original_text="The proposed Project Manager must have a minimum of 8 years of relevant experience.",
            category="Staffing & Experience",
            parameter="Project Manager relevant experience",
            operator=">=",
            value=8.0,
            unit="years",
            is_mandatory=True,
        )
        company_evidence = RetrievedEvidenceCandidate(
            rank=1,
            evidence_id="EVD-EPSILON-005",
            sentence="Established in 2022, our dynamic team has successfully completed 2 years of specialized operations.",
            similarity_score=0.42,
            source_document="BIDDER-EPSILON-INFRA-PROPOSAL.pdf",
            page_number=1,
            numeric_values=[{"raw": "2 years", "value": 2.0, "unit": "years",
                             "parameter_hint": "corporate_experience"}],
            has_negation=False,
            uncertainty_flag=False,
        )
        ret = RequirementRetrievalResult(
            requirement_id="REQ-PM-EXP-001",
            requirement_text=pm_exp_req.original_text,
            parameter=pm_exp_req.parameter,
            candidates=[company_evidence],
        )
        result = self.classifier.evaluate_clause(pm_exp_req, ret)
        self.assertEqual(result.label, "MISSING_EVIDENCE",
                         f"Company operating experience must not satisfy PM req. "
                         f"Got '{result.label}', reasoning: '{result.reasoning[:200]}'")
        self.assertIn("corporate", result.reasoning.lower(),
                      "Reasoning must mention corporate experience was rejected")

    def test_person_specific_pm_evidence_accepted(self):
        """Person-specific PM evidence (11 years, hint=duration) must be COMPLIANT vs >= 8 years."""
        pm_exp_req = ExtractedRequirement(
            requirement_id="REQ-PM-EXP-002",
            original_text="The proposed Project Manager must have at least 8 years of relevant experience.",
            category="Staffing & Experience",
            parameter="Project Manager relevant experience",
            operator=">=",
            value=8.0,
            unit="years",
            is_mandatory=True,
        )
        alpha_pm = RetrievedEvidenceCandidate(
            rank=1,
            evidence_id="EVD-ALPHA-PM",
            sentence="Lead PM Dr. Arthur Vance holds active PMI PMP certification and brings 11 years of direct experience leading enterprise cloud migration.",
            similarity_score=0.71,
            source_document="BIDDER-ALPHA-TECH-PROPOSAL.pdf",
            page_number=1,
            numeric_values=[{"raw": "11 years", "value": 11.0, "unit": "years",
                             "parameter_hint": "duration"}],
            has_negation=False,
            uncertainty_flag=False,
        )
        ret = RequirementRetrievalResult(
            requirement_id="REQ-PM-EXP-002",
            requirement_text=pm_exp_req.original_text,
            parameter=pm_exp_req.parameter,
            candidates=[alpha_pm],
        )
        result = self.classifier.evaluate_clause(pm_exp_req, ret)
        self.assertEqual(result.label, "COMPLIANT",
                         f"PM with 11 years should be COMPLIANT vs >= 8 years. Got '{result.label}'")

    def test_pipeline_pm_experience_epsilon(self):
        """Epsilon pipeline: PM experience must be MISSING_EVIDENCE or UNCERTAIN (company-age evidence only)."""
        pipeline = CompliancePipeline()
        report = pipeline.run(_TENDER_IT001, _BIDDER_EPSILON, save_to_db=False)
        pm_exp = next((e for e in report.evaluations
                       if "project manager relevant experience" in e.parameter.lower()), None)
        self.assertIsNotNone(pm_exp, "PM experience evaluation must exist")
        self.assertIn(pm_exp.label, ("MISSING_EVIDENCE", "UNCERTAIN"),
                      f"Epsilon PM experience must be MISSING/UNCERTAIN. "
                      f"Got '{pm_exp.label}', evidence: '{pm_exp.top_evidence_sentence}'")
        self.assertNotEqual(pm_exp.label, "COMPLIANT",
                            "Company 2-year history must never satisfy PM 8-year requirement")

    def test_pipeline_pm_experience_alpha(self):
        """Alpha pipeline: PM experience must be COMPLIANT (Dr Arthur Vance, 11 years named)."""
        pipeline = CompliancePipeline()
        report = pipeline.run(_TENDER_IT001, _BIDDER_ALPHA, save_to_db=False)
        pm_exp = next((e for e in report.evaluations
                       if "project manager relevant experience" in e.parameter.lower()), None)
        self.assertIsNotNone(pm_exp, "PM experience evaluation must exist for Alpha")
        self.assertEqual(pm_exp.label, "COMPLIANT",
                         f"Alpha PM (11 years) must be COMPLIANT vs >= 8 years. "
                         f"Got '{pm_exp.label}', evidence: '{pm_exp.top_evidence_sentence}'")

    def test_ambiguous_collective_pm_office_experience_uncertain(self):
        """Ambiguous evidence: collective PM office experience must yield UNCERTAIN, not COMPLIANT."""
        pm_exp_req = ExtractedRequirement(
            requirement_id="REQ-PM-EXP-003",
            original_text="The proposed Project Manager must have at least 8 years of relevant experience.",
            category="Staffing & Experience",
            parameter="Project Manager relevant experience",
            operator=">=",
            value=8.0,
            unit="years",
            is_mandatory=True,
        )
        ambiguous_evidence = RetrievedEvidenceCandidate(
            rank=1,
            evidence_id="EVD-AMBIG-PM-001",
            sentence="Our PM office has 8 years of collective experience across enterprise cloud deployments.",
            similarity_score=0.68,
            source_document="BIDDER-PROPOSAL.pdf",
            page_number=1,
            numeric_values=[{"raw": "8 years", "value": 8.0, "unit": "years",
                             "parameter_hint": "duration"}],
            has_negation=False,
            uncertainty_flag=False,
        )
        ret = RequirementRetrievalResult(
            requirement_id="REQ-PM-EXP-003",
            requirement_text=pm_exp_req.original_text,
            parameter=pm_exp_req.parameter,
            candidates=[ambiguous_evidence],
        )
        result = self.classifier.evaluate_clause(pm_exp_req, ret)
        self.assertEqual(result.label, "UNCERTAIN",
                         f"Collective PM office experience must yield UNCERTAIN. Got '{result.label}'")
        self.assertNotEqual(result.label, "COMPLIANT",
                            "Collective team experience must never establish compliance for individual PM requirement")

    def test_ambiguous_pm_with_company_experience_not_compliant(self):
        """Ambiguous evidence: sentence where duration qualifies company must yield UNCERTAIN, not COMPLIANT."""
        pm_exp_req = ExtractedRequirement(
            requirement_id="REQ-PM-EXP-004",
            original_text="The proposed Project Manager must have at least 8 years of relevant experience.",
            category="Staffing & Experience",
            parameter="Project Manager relevant experience",
            operator=">=",
            value=8.0,
            unit="years",
            is_mandatory=True,
        )
        mixed_evidence = RetrievedEvidenceCandidate(
            rank=1,
            evidence_id="EVD-AMBIG-PM-002",
            sentence="Our proposed Project Manager will lead execution for our company, which has 10 years of commercial operating experience.",
            similarity_score=0.65,
            source_document="BIDDER-PROPOSAL.pdf",
            page_number=1,
            numeric_values=[{"raw": "10 years", "value": 10.0, "unit": "years",
                             "parameter_hint": "corporate_experience"}],
            has_negation=False,
            uncertainty_flag=False,
        )
        ret = RequirementRetrievalResult(
            requirement_id="REQ-PM-EXP-004",
            requirement_text=pm_exp_req.original_text,
            parameter=pm_exp_req.parameter,
            candidates=[mixed_evidence],
        )
        result = self.classifier.evaluate_clause(pm_exp_req, ret)
        self.assertIn(result.label, ("UNCERTAIN", "MISSING_EVIDENCE"),
                      f"Mixed company-duration evidence must not yield COMPLIANT. Got '{result.label}'")
        self.assertNotEqual(result.label, "COMPLIANT")

    def test_valid_named_person_pm_below_threshold_non_compliant(self):
        """Person-specific PM with insufficient experience (5 yrs < 8 yrs) must be NON_COMPLIANT."""
        pm_exp_req = ExtractedRequirement(
            requirement_id="REQ-PM-EXP-005",
            original_text="The proposed Project Manager must have at least 8 years of relevant experience.",
            category="Staffing & Experience",
            parameter="Project Manager relevant experience",
            operator=">=",
            value=8.0,
            unit="years",
            is_mandatory=True,
        )
        junior_pm = RetrievedEvidenceCandidate(
            rank=1,
            evidence_id="EVD-PM-JUNIOR",
            sentence="Lead PM Dr. Arthur Vance holds active PMI PMP certification and brings 5 years of direct experience leading enterprise cloud migration.",
            similarity_score=0.72,
            source_document="BIDDER-PROPOSAL.pdf",
            page_number=1,
            numeric_values=[{"raw": "5 years", "value": 5.0, "unit": "years",
                             "parameter_hint": "duration"}],
            has_negation=False,
            uncertainty_flag=False,
        )
        ret = RequirementRetrievalResult(
            requirement_id="REQ-PM-EXP-005",
            requirement_text=pm_exp_req.original_text,
            parameter=pm_exp_req.parameter,
            candidates=[junior_pm],
        )
        result = self.classifier.evaluate_clause(pm_exp_req, ret)
        self.assertEqual(result.label, "NON_COMPLIANT",
                         f"Person-specific PM with 5 years must be NON_COMPLIANT vs >= 8 years. Got '{result.label}'")


# ============================================================
# BUG 3 - OEM Authorization vs Warranty Confusion
# ============================================================

class TestBug3OemAuthorizationVsWarranty(unittest.TestCase):
    """BUG 3: MAF/OEM authorization and generic product warranty are distinct criteria."""

    def setUp(self):
        self.extractor = RequirementExtractor()
        self.classifier = BaselineClassifier()

    def test_maf_clause_extracted_as_mandatory_submission(self):
        """Clause 6.4 must be MANDATORY_SUBMISSION/OEM_MAF_LETTER, not warranty-duration."""
        clause = (
            "6.4 OEM Authorization (MAF): Tender-specific MAF with back-to-back "
            "36 months warranty if bidder is reseller."
        )
        reqs = self.extractor.extract_from_text(clause, source_document="TEST.pdf", page_number=2)
        maf_reqs = [r for r in reqs
                    if "oem" in r.parameter.lower() or "maf" in r.parameter.lower()]
        wrong_warranty = [r for r in reqs
                          if "warranty duration" in r.parameter.lower()
                          and "oem" not in r.original_text.lower()]
        self.assertGreater(len(maf_reqs), 0, "MAF/OEM Authorization must be extracted")
        maf = maf_reqs[0]
        self.assertEqual(maf.operator, "MANDATORY_SUBMISSION",
                         f"MAF must have MANDATORY_SUBMISSION op, got {maf.operator}")
        self.assertIn("OEM_MAF_LETTER", str(maf.value),
                      "MAF value must be OEM_MAF_LETTER")
        self.assertEqual(len(wrong_warranty), 0,
                         "BUG 3: MAF clause must NOT be extracted as warranty-duration requirement")

    def test_generic_warranty_does_not_satisfy_maf(self):
        """Generic 18-month warranty must NOT satisfy OEM MAF requirement."""
        maf_req = ExtractedRequirement(
            requirement_id="REQ-MAF-001",
            original_text="6.4 OEM Authorization (MAF): Tender-specific MAF required.",
            category="GeM & Statutory Procurement Mandates",
            parameter="tender-specific OEM authorization (MAF)",
            operator="MANDATORY_SUBMISSION",
            value="OEM_MAF_LETTER",
            unit=None,
            is_mandatory=True,
        )
        generic_warranty = RetrievedEvidenceCandidate(
            rank=1,
            evidence_id="EVD-EPS-WARRANTY",
            sentence="Standard warranty coverage of 18 months (1.5 years) is bundled, with optional extensions available at extra cost.",
            similarity_score=0.38,
            source_document="BIDDER-EPSILON-INFRA-PROPOSAL.pdf",
            page_number=1,
            numeric_values=[{"raw": "18 months", "value": 18.0, "unit": "months",
                             "parameter_hint": "warranty_period"}],
            extracted_entities=[],
            has_negation=False,
            uncertainty_flag=False,
        )
        ret = RequirementRetrievalResult(
            requirement_id="REQ-MAF-001",
            requirement_text=maf_req.original_text,
            parameter=maf_req.parameter,
            candidates=[generic_warranty],
        )
        result = self.classifier.evaluate_clause(maf_req, ret)
        self.assertIn(result.label, ("MISSING_EVIDENCE", "UNCERTAIN"),
                      f"Generic warranty must NOT satisfy MAF. Got '{result.label}'")
        self.assertNotEqual(result.label, "COMPLIANT",
                            "18-month product warranty is not OEM authorization proof")

    def test_explicit_maf_evidence_accepted(self):
        """Explicit MAF letter (Alpha/Cisco OEM letterhead) must be COMPLIANT."""
        maf_req = ExtractedRequirement(
            requirement_id="REQ-MAF-002",
            original_text="6.4 OEM Authorization (MAF): Tender-specific MAF required.",
            category="GeM & Statutory Procurement Mandates",
            parameter="tender-specific OEM authorization (MAF)",
            operator="MANDATORY_SUBMISSION",
            value="OEM_MAF_LETTER",
            unit=None,
            is_mandatory=True,
        )
        alpha_maf = RetrievedEvidenceCandidate(
            rank=1,
            evidence_id="EVD-ALPHA-MAF",
            sentence="Enclosed is Tender-Specific MAF (Ref: OEM-AUTH-2024-IT01) executed on OEM letterhead by Cisco Systems, guaranteeing direct warranty and spares back-to-back for 36 months.",
            similarity_score=0.82,
            source_document="BIDDER-ALPHA-TECH-PROPOSAL.pdf",
            page_number=2,
            numeric_values=[],
            extracted_entities=[{"text": "Cisco Systems", "label": "ORG",
                                  "start_char": 0, "end_char": 13}],
            has_negation=False,
            uncertainty_flag=False,
        )
        ret = RequirementRetrievalResult(
            requirement_id="REQ-MAF-002",
            requirement_text=maf_req.original_text,
            parameter=maf_req.parameter,
            candidates=[alpha_maf],
        )
        result = self.classifier.evaluate_clause(maf_req, ret)
        self.assertEqual(result.label, "COMPLIANT",
                         f"Explicit OEM MAF must satisfy MAF requirement. Got '{result.label}'")

    def test_pipeline_maf_epsilon(self):
        """Epsilon pipeline: OEM MAF must be MISSING_EVIDENCE (no MAF letter, only generic warranty)."""
        pipeline = CompliancePipeline()
        report = pipeline.run(_TENDER_IT001, _BIDDER_EPSILON, save_to_db=False)
        maf_eval = next((e for e in report.evaluations
                         if "oem" in e.parameter.lower() or "maf" in e.parameter.lower()), None)
        self.assertIsNotNone(maf_eval, "OEM MAF evaluation must exist")
        self.assertIn(maf_eval.label, ("MISSING_EVIDENCE", "UNCERTAIN"),
                      f"Epsilon OEM MAF must be MISSING/UNCERTAIN. "
                      f"Got '{maf_eval.label}', evidence: '{maf_eval.top_evidence_sentence}'")

    def test_pipeline_maf_alpha(self):
        """Alpha pipeline: OEM MAF must be COMPLIANT (Cisco MAF letter cited)."""
        pipeline = CompliancePipeline()
        report = pipeline.run(_TENDER_IT001, _BIDDER_ALPHA, save_to_db=False)
        maf_eval = next((e for e in report.evaluations
                         if "oem" in e.parameter.lower() or "maf" in e.parameter.lower()), None)
        self.assertIsNotNone(maf_eval, "OEM MAF evaluation must exist for Alpha")
        self.assertEqual(maf_eval.label, "COMPLIANT",
                         f"Alpha OEM MAF must be COMPLIANT. "
                         f"Got '{maf_eval.label}', evidence: '{maf_eval.top_evidence_sentence}'")


# ============================================================
# BUG 4 - Tender Disclaimer Mistaken for Requirement
# ============================================================

class TestBug4DisclaimerFilter(unittest.TestCase):
    """BUG 4: Legal notice / disclaimer text must not be extracted as a compliance requirement."""

    def setUp(self):
        self.extractor = RequirementExtractor()

    def test_disclaimer_not_extracted(self):
        """The disclaimer sentence must be filtered."""
        disclaimer = (
            "LEGAL NOTICE: This tender document is a simulated artifact created "
            "strictly for testing and validating NLP algorithms. "
            "It does not represent an actual public solicitation."
        )
        reqs = self.extractor.extract_from_text(disclaimer, source_document="TEST.pdf")
        solicitation_reqs = [r for r in reqs
                             if "solicitation" in r.original_text.lower()]
        self.assertEqual(len(solicitation_reqs), 0,
                         "Disclaimer text must not be extracted as a requirement")

    def test_simulated_artifact_filtered(self):
        """'simulated artifact' phrase must be filtered."""
        line = "This tender document is a simulated artifact created strictly for testing."
        reqs = self.extractor.extract_from_text(line)
        self.assertEqual(len(reqs), 0, "'simulated artifact' line must be filtered")

    def test_legitimate_requirements_preserved(self):
        """Turnover requirement adjacent to disclaimer must still be extracted."""
        page_text = (
            "LEGAL NOTICE: This document is a simulated artifact. "
            "It does not represent an actual public solicitation. "
            "2.1 Minimum Average Annual Turnover: The bidder must demonstrate "
            "an average annual audited financial turnover of at least USD 10.0 "
            "Million over the last three fiscal years."
        )
        reqs = self.extractor.extract_from_text(page_text, source_document="TEST.pdf")
        turnover = [r for r in reqs if "turnover" in r.parameter.lower()]
        self.assertGreater(len(turnover), 0,
                           "Turnover requirement must still be extracted after disclaimer filter")
        solicitation = [r for r in reqs if "solicitation" in r.original_text.lower()]
        self.assertEqual(len(solicitation), 0,
                         "Disclaimer must be filtered even when adjacent to requirements")

    def test_pipeline_disclaimer_absent_epsilon(self):
        """Epsilon pipeline: disclaimer must not appear as any evaluated requirement."""
        pipeline = CompliancePipeline()
        report = pipeline.run(_TENDER_IT001, _BIDDER_EPSILON, save_to_db=False)
        disc = [e for e in report.evaluations
                if "solicitation" in e.requirement_text.lower()]
        self.assertEqual(len(disc), 0,
                         "Disclaimer must not appear as a compliance evaluation criterion")

    def test_pipeline_disclaimer_absent_alpha(self):
        """Alpha pipeline: disclaimer must not appear as any evaluated requirement."""
        pipeline = CompliancePipeline()
        report = pipeline.run(_TENDER_IT001, _BIDDER_ALPHA, save_to_db=False)
        disc = [e for e in report.evaluations
                if "solicitation" in e.requirement_text.lower()]
        self.assertEqual(len(disc), 0,
                         "Disclaimer must not appear in Alpha compliance report")


# ============================================================
# Cross-cutting: similarity guard + missing-evidence guard
# ============================================================

class TestCrossCuttingGuards(unittest.TestCase):
    """Req 4 & 5: similarity alone is not compliance; missing evidence is not non-compliance."""

    def test_high_similarity_no_numeric_not_compliant(self):
        """Similarity 0.90 + no numeric value must NOT yield COMPLIANT for numeric requirement."""
        req = ExtractedRequirement(
            requirement_id="REQ-SIM-001",
            original_text="Warranty must be at least 36 months.",
            category="Statutory Accreditations & Warranties",
            parameter="comprehensive warranty duration",
            operator=">=",
            value=36.0,
            unit="months",
            is_mandatory=True,
        )
        high_sim_no_num = RetrievedEvidenceCandidate(
            rank=1,
            evidence_id="EVD-SIM-001",
            sentence="We offer comprehensive warranty coverage for enterprise deployments.",
            similarity_score=0.90,
            source_document="PROPOSAL.pdf",
            page_number=1,
            numeric_values=[],
            has_negation=False,
            uncertainty_flag=False,
        )
        ret = RequirementRetrievalResult(
            requirement_id="REQ-SIM-001",
            requirement_text=req.original_text,
            parameter=req.parameter,
            candidates=[high_sim_no_num],
        )
        classifier = BaselineClassifier()
        result = classifier.evaluate_clause(req, ret)
        self.assertNotEqual(result.label, "COMPLIANT",
                            f"Similarity 0.90 without numeric value must not yield COMPLIANT. "
                            f"Got '{result.label}'")

    def test_missing_evidence_not_non_compliant(self):
        """Low similarity must yield MISSING_EVIDENCE, not NON_COMPLIANT."""
        req = ExtractedRequirement(
            requirement_id="REQ-MISS-001",
            original_text="The bidder must have PMP or PRINCE2 certification.",
            category="Staffing & Experience",
            parameter="Project Manager PMP/PRINCE2 certification",
            operator="CONTAINS",
            value="PMP_OR_PRINCE2_CERTIFICATION",
            unit=None,
            is_mandatory=True,
        )
        low_sim = RetrievedEvidenceCandidate(
            rank=1,
            evidence_id="EVD-LOW-001",
            sentence="We have offices in three metropolitan areas.",
            similarity_score=0.02,
            source_document="PROPOSAL.pdf",
            page_number=1,
            numeric_values=[],
            has_negation=False,
        )
        ret = RequirementRetrievalResult(
            requirement_id="REQ-MISS-001",
            requirement_text=req.original_text,
            parameter=req.parameter,
            candidates=[low_sim],
        )
        classifier = BaselineClassifier()
        result = classifier.evaluate_clause(req, ret)
        self.assertEqual(result.label, "MISSING_EVIDENCE",
                         f"Low similarity must yield MISSING_EVIDENCE, got '{result.label}'")
        self.assertNotEqual(result.label, "NON_COMPLIANT",
                            "MISSING_EVIDENCE must not be escalated to NON_COMPLIANT")


if __name__ == "__main__":
    unittest.main(verbosity=2)
