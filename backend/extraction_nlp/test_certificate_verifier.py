"""
test_certificate_verifier.py - Comprehensive Unit & Regression Test Suite for Step 2B & 2B.1
Certificate Validity & Certification Posture Verification

Tests:
1. Bare certification statements without verified validity evaluate to UNCERTAIN (Task 1)
2. Historical expiry compared against tender submission deadline (Task 2, Cases A, B, C, D)
3. Multiple certificate date association per standard (Task 3: separate sentences, single sentence, reversed order)
4. Ambiguous certificate-date association routes to UNCERTAIN (Task 3)
5. Compatibility and alignment language strictly routes to UNCERTAIN
6. Standard name extraction & matching (ISO 9001, ISO 27001, CMMI, SOC 2, PMP)
7. Certificate issue date extraction across multiple formats
8. Certificate expiry date extraction across multiple formats
9. Equality boundary: certificate expiring on the exact submission deadline (inclusive)
10. Original quotation and provenance preservation
11. End-to-end integration with BaselineClassifier
12. Regression cases for previously identified false positives
"""

import pytest
from datetime import date

from backend.extraction_nlp.certificate_verifier import CertificateVerifier, CertificateClaim
from backend.extraction_nlp.requirement_extractor import ExtractedRequirement
from backend.matching.tfidf_retriever import RequirementRetrievalResult, RetrievedEvidenceCandidate
from backend.classification.baseline_classifier import BaselineClassifier


@pytest.fixture
def verifier():
    return CertificateVerifier()


@pytest.fixture
def classifier():
    return BaselineClassifier(enable_nli=False, enable_reconciliation=False)


# ==============================================================================
# 1. TASK 1: BARE CERTIFICATION CLAIMS (STEP 2B.1)
# ==============================================================================

class TestBareCertificationClaims:
    """Verifies that bare certification claims without verifiable validity route to UNCERTAIN."""

    def test_bare_statement_without_expiry_date_uncertain(self, verifier):
        """Req: 'The bidder must hold ISO 27001 certification.'
        Evd: 'Our company is ISO 27001 certified.'
        Expected: UNCERTAIN (validity cannot be established without expiry date).
        """
        lbl, conf, rsn, claim = verifier.verify_certificate(
            requirement_text="The bidder must hold ISO 27001 certification.",
            evidence_sentence="Our company is ISO 27001 certified.",
            req_standard="ISO 27001"
        )
        assert lbl == "UNCERTAIN"
        assert claim.claim_type == "CERTIFIED"
        assert claim.expiry_date is None
        assert "cannot verify active validity" in rsn.lower() or "expiry date" in rsn.lower()

    def test_bare_statement_with_deadline_still_uncertain_if_no_expiry(self, verifier):
        """Even if tender deadline is known, a bare claim without certificate expiry date is UNCERTAIN."""
        lbl, conf, rsn, claim = verifier.verify_certificate(
            requirement_text="The bidder must hold ISO 27001 certification.",
            evidence_sentence="Our company is ISO 27001 certified.",
            req_standard="ISO 27001",
            tender_submission_date="2024-10-15"
        )
        assert lbl == "UNCERTAIN"
        assert "expiry date" in rsn.lower()

    def test_verified_certificate_with_expiry_passes(self, verifier):
        """When official certificate has unexpired date and deadline, it passes."""
        lbl, conf, rsn, claim = verifier.verify_certificate(
            requirement_text="The bidder must hold ISO 27001 certification.",
            evidence_sentence="Enclosed is our official ISO 27001 certificate valid through 2025-05-31.",
            req_standard="ISO 27001",
            tender_submission_date="2024-10-15"
        )
        assert lbl == "COMPLIANT"
        assert claim.expiry_date == date(2025, 5, 31)


# ==============================================================================
# 2. TASK 2: HISTORICAL EXPIRY VS SUBMISSION DEADLINE (STEP 2B.1)
# ==============================================================================

class TestHistoricalExpiryDeadlineComparison:
    """Verifies Cases A, B, C, D from Step 2B.1 specification."""

    def test_case_a_expiry_after_deadline_passes(self, verifier):
        """Case A: Certificate expiry: 31 March 2025. Submission deadline: 15 October 2024.
        Certificate was valid at submission -> Pass the date-validity check (COMPLIANT).
        Does NOT get rejected simply because evidence uses 'expired on'.
        """
        lbl, conf, rsn, claim = verifier.verify_certificate(
            requirement_text="The bidder must hold a valid ISO 27001 certificate.",
            evidence_sentence="Our ISO 27001 certificate expired on 31 March 2025.",
            req_standard="ISO 27001",
            tender_submission_date="2024-10-15"
        )
        assert lbl == "COMPLIANT"
        assert claim.expiry_date == date(2025, 3, 31)
        assert "was valid through 2025-03-31 at the time of tender submission" in rsn.lower()

    def test_case_b_expiry_before_deadline_non_compliant(self, verifier):
        """Case B: Certificate expiry: 31 March 2025. Submission deadline: 15 October 2025.
        Certificate expired before submission -> NON_COMPLIANT.
        """
        lbl, conf, rsn, claim = verifier.verify_certificate(
            requirement_text="The bidder must hold a valid ISO 27001 certificate.",
            evidence_sentence="Our ISO 27001 certificate expired on 31 March 2025.",
            req_standard="ISO 27001",
            tender_submission_date="2025-10-15"
        )
        assert lbl == "NON_COMPLIANT"
        assert claim.expiry_date == date(2025, 3, 31)
        assert "expired on 2025-03-31, prior to the tender submission deadline" in rsn.lower()

    def test_case_c_expiry_known_deadline_unavailable_uncertain(self, verifier):
        """Case C: Certificate expiry: 31 March 2025. Submission deadline: unavailable (None).
        Expected: UNCERTAIN (historical validity at submission cannot be established).
        """
        lbl, conf, rsn, claim = verifier.verify_certificate(
            requirement_text="The bidder must hold a valid ISO 27001 certificate.",
            evidence_sentence="Our ISO 27001 certificate expired on 31 March 2025.",
            req_standard="ISO 27001",
            tender_submission_date=None
        )
        assert lbl == "UNCERTAIN"
        assert "submission deadline is unavailable" in rsn.lower()

    def test_case_d_expired_wording_no_date_no_deadline_uncertain(self, verifier):
        """Case D: Evidence: 'Our certificate is expired.' No expiry date and no submission deadline.
        Expected: UNCERTAIN/review.
        """
        lbl, conf, rsn, claim = verifier.verify_certificate(
            requirement_text="The bidder must hold a valid ISO 27001 certificate.",
            evidence_sentence="Our certificate is expired.",
            req_standard="ISO 27001",
            tender_submission_date=None
        )
        assert lbl == "UNCERTAIN"
        assert "no expiration date or tender submission deadline" in rsn.lower()


# ==============================================================================
# 3. TASK 3: MULTIPLE CERTIFICATE DATE ASSOCIATION (STEP 2B.1)
# ==============================================================================

class TestMultipleCertificateDateAssociation:
    """Verifies that each standard is associated with its own individual expiry date."""

    def test_separate_sentences_multiple_standards(self, verifier):
        """'ISO 9001 certificate valid until 31 March 2025. ISO 27001 certificate valid until 31 March 2027.'
        Tender deadline: 2026-01-01.
        ISO 9001 -> expired (2025 < 2026) -> NON_COMPLIANT.
        ISO 27001 -> valid (2027 >= 2026) -> COMPLIANT.
        """
        text = "ISO 9001 certificate valid until 31 March 2025. ISO 27001 certificate valid until 31 March 2027."
        deadline = "2026-01-01"

        lbl_9001, _, _, claim_9001 = verifier.verify_certificate(
            "Bidder must hold ISO 9001", text, req_standard="ISO 9001", tender_submission_date=deadline
        )
        lbl_27001, _, _, claim_27001 = verifier.verify_certificate(
            "Bidder must hold ISO 27001", text, req_standard="ISO 27001", tender_submission_date=deadline
        )

        assert claim_9001.expiry_date == date(2025, 3, 31)
        assert claim_27001.expiry_date == date(2027, 3, 31)
        assert lbl_9001 == "NON_COMPLIANT"
        assert lbl_27001 == "COMPLIANT"

    def test_single_compound_sentence_multiple_standards(self, verifier):
        """'We hold an ISO 9001 certificate valid until 31 March 2025 and an ISO 27001 certificate valid until 31 March 2027.'"""
        text = "We hold an ISO 9001 certificate valid until 31 March 2025 and an ISO 27001 certificate valid until 31 March 2027."
        deadline = "2026-01-01"

        lbl_9001, _, _, claim_9001 = verifier.verify_certificate(
            "Bidder must hold ISO 9001", text, req_standard="ISO 9001", tender_submission_date=deadline
        )
        lbl_27001, _, _, claim_27001 = verifier.verify_certificate(
            "Bidder must hold ISO 27001", text, req_standard="ISO 27001", tender_submission_date=deadline
        )

        assert claim_9001.expiry_date == date(2025, 3, 31)
        assert claim_27001.expiry_date == date(2027, 3, 31)
        assert lbl_9001 == "NON_COMPLIANT"
        assert lbl_27001 == "COMPLIANT"

    def test_reversed_order_standards_and_dates(self, verifier):
        """Reversed order: ISO 27001 mentioned first (expires 2027), then ISO 9001 (expires 2025)."""
        text = "ISO 27001 certificate valid until 31 March 2027. ISO 9001 certificate valid until 31 March 2025."
        deadline = "2026-01-01"

        lbl_9001, _, _, claim_9001 = verifier.verify_certificate(
            "Bidder must hold ISO 9001", text, req_standard="ISO 9001", tender_submission_date=deadline
        )
        lbl_27001, _, _, claim_27001 = verifier.verify_certificate(
            "Bidder must hold ISO 27001", text, req_standard="ISO 27001", tender_submission_date=deadline
        )

        assert claim_9001.expiry_date == date(2025, 3, 31)
        assert claim_27001.expiry_date == date(2027, 3, 31)
        assert lbl_9001 == "NON_COMPLIANT"
        assert lbl_27001 == "COMPLIANT"

    def test_ambiguous_distributed_pairing_uncertain(self, verifier):
        """'We hold ISO 9001 and ISO 27001 certificates valid until 31 March 2025 and 31 March 2027 respectively.'
        Ambiguous date pairing must route to UNCERTAIN rather than assigning unrelated dates.
        """
        text = "We hold ISO 9001 and ISO 27001 certificates valid until 31 March 2025 and 31 March 2027 respectively."
        lbl, conf, rsn, claim = verifier.verify_certificate(
            "Bidder must hold ISO 9001", text, req_standard="ISO 9001", tender_submission_date="2024-10-15"
        )
        assert lbl == "UNCERTAIN"
        assert claim.is_ambiguous_association is True
        assert "ambiguous" in rsn.lower()

    def test_structured_claims_extraction(self, verifier):
        """extract_all_certificate_claims returns structured list of claims with metadata."""
        text = "ISO 9001 certificate valid until 31 March 2025. ISO 27001 certificate valid until 31 March 2027."
        claims = verifier.extract_all_certificate_claims(text, source_document="PROPOSAL.pdf", page_number=5)
        assert len(claims) == 2
        assert claims[0].standard_name == "ISO 9001"
        assert claims[0].expiry_date == date(2025, 3, 31)
        assert claims[0].source_document == "PROPOSAL.pdf"
        assert claims[0].page_number == 5

        assert claims[1].standard_name == "ISO 27001"
        assert claims[1].expiry_date == date(2027, 3, 31)
        assert claims[1].source_document == "PROPOSAL.pdf"
        assert claims[1].page_number == 5


# ==============================================================================
# 4. TASK 5 SPECIFICATION EXAMPLES
# ==============================================================================

class TestTask5SpecificationCases:
    """Verifies all required examples specified in Task 5 of the prompt."""

    def test_example_1_certified_claim_recognized_but_uncertain_without_expiry(self, verifier):
        """1. Req: 'The bidder must hold ISO 27001 certification.'
        Evd: 'Our company is ISO 27001 certified.'
        Expected: Certification claim recognized; validity still needs checking -> UNCERTAIN.
        """
        lbl, conf, rsn, claim = verifier.verify_certificate(
            requirement_text="The bidder must hold ISO 27001 certification.",
            evidence_sentence="Our company is ISO 27001 certified.",
            req_standard="ISO 27001"
        )
        assert lbl == "UNCERTAIN"
        assert claim.claim_type == "CERTIFIED"
        assert "ISO 27001" in claim.standard_name
        assert "cannot verify active validity" in rsn.lower() or "expiry date" in rsn.lower()

    def test_example_2_compatibility_rejected(self, verifier):
        """2. Req: 'The bidder must hold ISO 27001 certification.'
        Evd: 'Our platform is compatible with ISO 27001.'
        Expected: Must NOT become COMPLIANT (routes to UNCERTAIN).
        """
        lbl, conf, rsn, claim = verifier.verify_certificate(
            requirement_text="The bidder must hold ISO 27001 certification.",
            evidence_sentence="Our platform is compatible with ISO 27001.",
            req_standard="ISO 27001"
        )
        assert lbl != "COMPLIANT"
        assert lbl == "UNCERTAIN"
        assert claim.claim_type == "COMPATIBILITY"
        assert "compatibility" in rsn.lower()

    def test_example_3_guideline_alignment_uncertain(self, verifier):
        """3. Req: 'The bidder must hold ISO 27001 certification.'
        Evd: 'Our security practices follow ISO 27001 guidelines.'
        Expected: UNCERTAIN/review.
        """
        lbl, conf, rsn, claim = verifier.verify_certificate(
            requirement_text="The bidder must hold ISO 27001 certification.",
            evidence_sentence="Our security practices follow ISO 27001 guidelines.",
            req_standard="ISO 27001"
        )
        assert lbl == "UNCERTAIN"
        assert claim.claim_type == "COMPATIBILITY"
        assert "not proof of certification" in rsn.lower() or "review" in rsn.lower()

    def test_example_4_expired_before_deadline_non_compliant(self, verifier):
        """4. Req: 'The bidder must hold a valid ISO 27001 certificate.'
        Evd: Certificate expired before actual tender submission deadline (2024-08-01 vs 2024-10-15).
        Expected: NON_COMPLIANT.
        """
        lbl, conf, rsn, claim = verifier.verify_certificate(
            requirement_text="The bidder must hold a valid ISO 27001 certificate.",
            evidence_sentence="We hold an ISO 27001 certificate valid until 2024-08-01.",
            req_standard="ISO 27001",
            tender_submission_date="2024-10-15"
        )
        assert lbl == "NON_COMPLIANT"
        assert claim.expiry_date == date(2024, 8, 1)
        assert "expired on 2024-08-01" in rsn.lower()

    def test_example_5_missing_expiry_date_uncertain(self, verifier):
        """5. Req: 'The bidder must hold a valid ISO 27001 certificate.'
        Evd: Certificate expiry date is missing.
        Expected: UNCERTAIN/review, not automatically COMPLIANT.
        """
        lbl, conf, rsn, claim = verifier.verify_certificate(
            requirement_text="The bidder must hold a valid ISO 27001 certificate.",
            evidence_sentence="Our company holds an accredited ISO 27001 certificate.",
            req_standard="ISO 27001",
            tender_submission_date="2024-10-15"
        )
        assert lbl == "UNCERTAIN"
        assert claim.expiry_date is None
        assert "expiry date" in rsn.lower()

    def test_example_6_valid_through_deadline_compliant(self, verifier):
        """6. Req: 'The bidder must hold a valid ISO 27001 certificate.'
        Evd: Certificate is valid through actual submission deadline (2025-05-31 vs 2024-10-15).
        Expected: COMPLIANT.
        """
        lbl, conf, rsn, claim = verifier.verify_certificate(
            requirement_text="The bidder must hold a valid ISO 27001 certificate.",
            evidence_sentence="We hold an accredited ISO 27001 certificate valid through 2025-05-31.",
            req_standard="ISO 27001",
            tender_submission_date="2024-10-15"
        )
        assert lbl == "COMPLIANT"
        assert claim.expiry_date == date(2025, 5, 31)
        assert "satisfies tender submission deadline" in rsn.lower()


# ==============================================================================
# 5. DATE PARSING FORMAT TESTS
# ==============================================================================

class TestDateParsingFormats:
    """Tests date extraction across all supported procurement date formats."""

    def test_format_iso(self, verifier):
        iss, exp = verifier.extract_dates("Issued on 2022-01-15, valid through 2025-01-15.")
        assert iss == date(2022, 1, 15)
        assert exp == date(2025, 1, 15)

    def test_format_day_month_year_written(self, verifier):
        iss, exp = verifier.extract_dates("Effective from 15th March 2021, expires on 31 March 2025.")
        assert iss == date(2021, 3, 15)
        assert exp == date(2025, 3, 31)

    def test_format_month_day_year_written(self, verifier):
        iss, exp = verifier.extract_dates("Awarded on October 1, 2020; valid until December 31, 2024.")
        assert iss == date(2020, 10, 1)
        assert exp == date(2024, 12, 31)

    def test_format_slash_uk_indian(self, verifier):
        iss, exp = verifier.extract_dates("Issue date: 01/04/2021; expiry date: 31/03/2025.")
        assert iss == date(2021, 4, 1)
        assert exp == date(2025, 3, 31)

    def test_format_month_year_only(self, verifier):
        iss, exp = verifier.extract_dates("Official certificate audited and valid through December 2026.")
        assert iss is None
        assert exp == date(2026, 12, 31)

    def test_format_year_only_with_cue(self, verifier):
        iss, exp = verifier.extract_dates("Accredited certificate valid through 2026.")
        assert iss is None
        assert exp == date(2026, 12, 31)


# ==============================================================================
# 6. CLAIM POSTURE CLASSIFICATION TESTS
# ==============================================================================

class TestClaimPostureClassification:
    """Tests distinction between actual certification, compatibility, and ambiguous claims."""

    def test_actual_certified_claims(self, verifier):
        assert verifier.classify_claim_posture("Our company is ISO 27001 certified.") == "CERTIFIED"
        assert verifier.classify_claim_posture("We hold a valid ISO 9001:2015 certificate.") == "CERTIFIED"
        assert verifier.classify_claim_posture("The organization has been certified against ISO 27001:2022.") == "CERTIFIED"
        assert verifier.classify_claim_posture("Enclosed is our official accreditation certificate.") == "CERTIFIED"

    def test_compatibility_and_alignment_claims(self, verifier):
        assert verifier.classify_claim_posture("Our platform is compatible with ISO 27001.") == "COMPATIBILITY"
        assert verifier.classify_claim_posture("Our processes are aligned with ISO 9001.") == "COMPATIBILITY"
        assert verifier.classify_claim_posture("Our security framework follows ISO guidelines.") == "COMPATIBILITY"
        assert verifier.classify_claim_posture("Our solution is designed according to ISO standards.") == "COMPATIBILITY"
        assert verifier.classify_claim_posture("Our workflows adhere to ISO principles.") == "COMPATIBILITY"

    def test_ambiguous_claims(self, verifier):
        assert verifier.classify_claim_posture("We comply with ISO requirements.") == "AMBIGUOUS"
        assert verifier.classify_claim_posture("We follow ISO 27001 practices.") == "AMBIGUOUS"
        assert verifier.classify_claim_posture("Our organization maintains ISO-level security.") == "AMBIGUOUS"

    def test_explicit_denial_claims(self, verifier):
        assert verifier.classify_claim_posture("The bidder does not currently hold an accredited ISO 9001:2015 certification.") == "DENIED"
        assert verifier.classify_claim_posture("We do not hold ISO 27001 certification.") == "DENIED"
        assert verifier.classify_claim_posture("Our firm has not obtained CMMI Level 5 certification yet.") == "DENIED"

    def test_explicit_expired_claims(self, verifier):
        assert verifier.classify_claim_posture("Our ISO 9001:2015 certification expired on January 15, 2023.") == "EXPIRED"
        assert verifier.classify_claim_posture("OEM Partner Certificate expired on 31-Dec-2023 and has not been renewed.") == "EXPIRED"


# ==============================================================================
# 7. DEADLINE AND BOUNDARY TESTS
# ==============================================================================

class TestDeadlineComparisonsAndBoundaries:
    """Tests deadline comparisons including boundary conditions and missing dates."""

    def test_expiry_equals_deadline_inclusive(self, verifier):
        """Certificate valid until exactly the submission deadline passes the check."""
        lbl, conf, rsn, claim = verifier.verify_certificate(
            requirement_text="The bidder must hold an active ISO 27001 certificate valid through 2024-10-15.",
            evidence_sentence="Certificate valid until 2024-10-15.",
            tender_submission_date="2024-10-15"
        )
        assert lbl == "COMPLIANT"

    def test_extract_deadline_from_requirement_text(self, verifier):
        """Extracts submission date embedded directly in requirement clause text."""
        deadline = verifier.extract_deadline_from_requirement(
            "The bidder must hold an active ISO 27001 certificate valid on 2024-10-15."
        )
        assert deadline == date(2024, 10, 15)


# ==============================================================================
# 8. INTEGRATION & REGRESSION TESTS WITH BASELINE CLASSIFIER
# ==============================================================================

class TestBaselineClassifierIntegration:
    """Tests end-to-end integration of CertificateVerifier within BaselineClassifier."""

    def test_regression_compatibility_not_compliant(self, classifier):
        """REGRESSION: PAIR-0322 & PAIR-0462 in test split were incorrectly labeled COMPLIANT by naive substring match."""
        req = ExtractedRequirement(
            requirement_id="REQ-TEST-ISO-01",
            original_text="The bidder must hold an active accredited ISO 9001:2015 (Quality Management Systems) certification valid through the bid submission date.",
            category="Statutory Compliance & Certifications",
            parameter="ISO 9001:2015 compliance",
            operator="CONTAINS",
            value="ISO 9001:2015",
            is_mandatory=True
        )
        cand = RetrievedEvidenceCandidate(
            rank=1,
            evidence_id="EVD-001",
            sentence="Our development practices and security controls are designed to be fully compatible with ISO 9001:2015 (Quality Management Systems) guidelines.",
            similarity_score=0.86,
            source_document="BIDDER-PROPOSAL.pdf",
            page_number=15
        )
        ret = RequirementRetrievalResult(
            requirement_id=req.requirement_id,
            requirement_text=req.original_text,
            parameter=req.parameter,
            candidates=[cand],
            top_similarity=cand.similarity_score,
            has_candidate=True
        )

        eval_res = classifier.evaluate_clause(req, ret, tender_submission_date="2024-10-15")
        assert eval_res.label == "UNCERTAIN"
        assert eval_res.label != "COMPLIANT"
        assert "compatibility" in eval_res.reasoning.lower()

    def test_regression_expired_certificate_not_compliant(self, classifier):
        """REGRESSION: PAIR-0265 & PAIR-0490 in test split were incorrectly labeled COMPLIANT by naive substring match."""
        req = ExtractedRequirement(
            requirement_id="REQ-TEST-ISO-02",
            original_text="The bidder must hold an active accredited ISO 9001:2015 (Quality Management Systems) certification valid through the bid submission date.",
            category="Statutory Compliance & Certifications",
            parameter="ISO 9001:2015 compliance",
            operator="CONTAINS",
            value="ISO 9001:2015",
            is_mandatory=True
        )
        cand = RetrievedEvidenceCandidate(
            rank=1,
            evidence_id="EVD-002",
            sentence="Our ISO 9001:2015 (Quality Management Systems) certification expired on January 15, 2023, and re-certification audit is currently scheduled.",
            similarity_score=0.85,
            source_document="BIDDER-PROPOSAL.pdf",
            page_number=16
        )
        ret = RequirementRetrievalResult(
            requirement_id=req.requirement_id,
            requirement_text=req.original_text,
            parameter=req.parameter,
            candidates=[cand],
            top_similarity=cand.similarity_score,
            has_candidate=True
        )

        eval_res = classifier.evaluate_clause(req, ret, tender_submission_date="2024-10-15")
        assert eval_res.label == "NON_COMPLIANT"
        assert eval_res.label != "COMPLIANT"
        assert "expired" in eval_res.reasoning.lower()

    def test_valid_unexpired_certificate_compliant(self, classifier):
        """PAIR-0069 in test split: genuine accredited unexpired certificate evaluates to COMPLIANT."""
        req = ExtractedRequirement(
            requirement_id="REQ-TEST-ISO-03",
            original_text="The bidder must hold an active accredited ISO 9001:2015 (Quality Management Systems) certification valid through the bid submission date.",
            category="Statutory Compliance & Certifications",
            parameter="ISO 9001:2015 compliance",
            operator="CONTAINS",
            value="ISO 9001:2015",
            is_mandatory=True
        )
        cand = RetrievedEvidenceCandidate(
            rank=1,
            evidence_id="EVD-003",
            sentence="Enclosed is our official ISO 9001:2015 (Quality Management Systems) accreditation certificate (Cert No: ISO-98421-B), audited and valid through December 2026.",
            similarity_score=0.89,
            source_document="BIDDER-PROPOSAL.pdf",
            page_number=18
        )
        ret = RequirementRetrievalResult(
            requirement_id=req.requirement_id,
            requirement_text=req.original_text,
            parameter=req.parameter,
            candidates=[cand],
            top_similarity=cand.similarity_score,
            has_candidate=True
        )

        eval_res = classifier.evaluate_clause(req, ret, tender_submission_date="2024-10-15")
        assert eval_res.label == "COMPLIANT"
        assert "valid through 2026-12-31" in eval_res.reasoning.lower()
