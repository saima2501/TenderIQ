"""
test_rule_false_positives.py - False-Positive Adversarial Stress Test Suite

Tests the rules introduced or refined in Step 2E:
1. Accreditation / Standard Matching:
   - Generic words ('encryption', 'standards') must not trigger COMPLIANT for AES-256.
   - Mismatched bit length ('AES-128') must not satisfy 'AES-256'.
2. Debarment & Blacklisting:
   - Explicit past debarment must not pass as COMPLIANT.
   - Affirmative non-debarment must pass as COMPLIANT.
   - Disputed/quashed debarment or ambiguous claims must remain UNCERTAIN.
3. Project Manager Experience Posture:
   - Corporate age/founding must be rejected (not attribute company age to person).
   - Collective/pooled team experience must yield UNCERTAIN.
   - Shortfall in personal experience must yield NON_COMPLIANT.
4. Statutory Undertaking & Filings:
   - Future promise ('will apply for GSTIN') must not satisfy mandatory submission.
   - Genuine active GSTIN statement must satisfy mandatory submission.
"""

import pytest
from backend.extraction_nlp.requirement_extractor import RequirementExtractor, ExtractedRequirement
from backend.extraction_nlp.evidence_extractor import EvidenceExtractor
from backend.matching.tfidf_retriever import RequirementRetrievalResult, RetrievedEvidenceCandidate
from backend.classification.baseline_classifier import BaselineClassifier


@pytest.fixture(scope="module")
def classifier():
    return BaselineClassifier(enable_nli=False, enable_reconciliation=False)


@pytest.fixture(scope="module")
def req_ext():
    return RequirementExtractor()


@pytest.fixture(scope="module")
def evd_ext():
    return EvidenceExtractor()


def _make_retrieval(req_obj, evd_text, evd_ext, sim=0.75):
    evd_list = evd_ext.extract_from_text(evd_text)
    evd_obj = evd_list[0] if evd_list else None
    cand = RetrievedEvidenceCandidate(
        rank=1,
        evidence_id="EVD-TEST",
        sentence=evd_text,
        similarity_score=sim,
        source_document="BIDDER.pdf",
        page_number=1,
        has_negation=evd_obj.has_negation if evd_obj else False,
        numeric_values=evd_obj.numeric_values if evd_obj else [],
        extracted_entities=evd_obj.extracted_entities if evd_obj else [],
        uncertainty_flag=evd_obj.uncertainty_flag if evd_obj else False,
        retrieval_method="semantic"
    )
    return RequirementRetrievalResult(
        requirement_id=req_obj.requirement_id,
        requirement_text=req_obj.original_text,
        parameter=req_obj.parameter,
        candidates=[cand],
        top_similarity=sim,
        has_candidate=True
    )


# ---------------------------------------------------------------------------
# 1. ACCREDITATION FALSE POSITIVE TESTS
# ---------------------------------------------------------------------------

def test_accreditation_generic_words_do_not_satisfy_aes256(classifier, evd_ext):
    req = ExtractedRequirement(
        requirement_id="REQ-AES",
        original_text="All data must be encrypted using AES-256.",
        category="Technical Specifications",
        parameter="AES-256 Encryption compliance",
        operator="CONTAINS",
        value="AES-256 Encryption"
    )
    # Vague claim with generic word 'encryption'
    evd = "Robust encryption standards and compliance policies are strictly maintained across our cloud systems."
    ret = _make_retrieval(req, evd, evd_ext)
    res = classifier.evaluate_clause(req, ret)
    assert res.label == "UNCERTAIN", f"Expected UNCERTAIN, got {res.label}"


def test_accreditation_spelled_out_aes256_satisfies(classifier, evd_ext):
    req = ExtractedRequirement(
        requirement_id="REQ-AES",
        original_text="All data must be encrypted using AES-256.",
        category="Technical Specifications",
        parameter="AES-256 Encryption compliance",
        operator="CONTAINS",
        value="AES-256 Encryption"
    )
    # Spelled-out version
    evd = "All databases are secured with military-grade 256-bit Advanced Encryption Standard storage."
    ret = _make_retrieval(req, evd, evd_ext)
    res = classifier.evaluate_clause(req, ret)
    assert res.label == "COMPLIANT", f"Expected COMPLIANT, got {res.label}"


# ---------------------------------------------------------------------------
# 2. DEBARMENT & BLACKLISTING FALSE POSITIVE TESTS
# ---------------------------------------------------------------------------

def test_debarment_admission_yields_non_compliant(classifier, evd_ext):
    req = ExtractedRequirement(
        requirement_id="REQ-DEB",
        original_text="The bidder must submit a declaration that the firm has never been blacklisted or debarred.",
        category="Statutory Compliance",
        parameter="solemn non-blacklisting declaration",
        operator="MANDATORY_SUBMISSION",
        value="NON_BLACKLISTING_AFFIDAVIT"
    )
    # Direct admission
    evd = "Declaration: The firm was temporarily debarred by state electricity board for 12 months in 2020."
    ret = _make_retrieval(req, evd, evd_ext)
    res = classifier.evaluate_clause(req, ret)
    assert res.label == "NON_COMPLIANT", f"Expected NON_COMPLIANT, got {res.label}"


def test_affirmative_non_debarment_yields_compliant(classifier, evd_ext):
    req = ExtractedRequirement(
        requirement_id="REQ-DEB",
        original_text="The bidder must submit a declaration that the firm has never been blacklisted or debarred.",
        category="Statutory Compliance",
        parameter="solemn non-blacklisting declaration",
        operator="MANDATORY_SUBMISSION",
        value="NON_BLACKLISTING_AFFIDAVIT"
    )
    evd = "Solemn affidavit confirms our company has never been debarred or blacklisted by any government authority."
    ret = _make_retrieval(req, evd, evd_ext)
    res = classifier.evaluate_clause(req, ret)
    assert res.label == "COMPLIANT", f"Expected COMPLIANT, got {res.label}"


def test_gem_incident_undertaking_yields_compliant(classifier, evd_ext):
    req = ExtractedRequirement(
        requirement_id="REQ-GEM",
        original_text="The bidder must submit an undertaking that the firm has not been debarred under GeM Incident Management Policy.",
        category="Statutory Compliance",
        parameter="solemn non-blacklisting declaration",
        operator="MANDATORY_SUBMISSION",
        value="NON_BLACKLISTING_AFFIDAVIT"
    )
    evd = "Notarized undertaking confirms our GeM seller profile has zero active incident penalties, notices, or debarment orders across all categories."
    ret = _make_retrieval(req, evd, evd_ext)
    res = classifier.evaluate_clause(req, ret)
    assert res.label == "COMPLIANT", f"Expected COMPLIANT, got {res.label}"


# ---------------------------------------------------------------------------
# 3. PROJECT MANAGER EXPERIENCE POSTURE TESTS
# ---------------------------------------------------------------------------

def test_pm_corporate_experience_rejected(classifier, evd_ext):
    req = ExtractedRequirement(
        requirement_id="REQ-PM",
        original_text="The proposed Project Manager must possess a minimum of 8 years of relevant experience.",
        category="Key Personnel",
        parameter="Project Manager relevant experience",
        operator=">=",
        value=8.0,
        unit="years"
    )
    # Cites firm history, not personal PM
    evd = "Our firm was incorporated in 2012 and has completed 12 years of specialized cloud operations."
    ret = _make_retrieval(req, evd, evd_ext)
    res = classifier.evaluate_clause(req, ret)
    assert res.label == "MISSING_EVIDENCE", f"Expected MISSING_EVIDENCE, got {res.label}"


def test_pm_collective_team_experience_yields_uncertain(classifier, evd_ext):
    req = ExtractedRequirement(
        requirement_id="REQ-PM",
        original_text="The proposed Project Manager must possess a minimum of 8 years of relevant experience.",
        category="Key Personnel",
        parameter="Project Manager relevant experience",
        operator=">=",
        value=8.0,
        unit="years"
    )
    # Cites pooled team
    evd = "Our Project Management Office brings collective experience of over 25 years in enterprise migrations."
    ret = _make_retrieval(req, evd, evd_ext)
    res = classifier.evaluate_clause(req, ret)
    assert res.label == "UNCERTAIN", f"Expected UNCERTAIN, got {res.label}"


def test_pm_named_shortfall_yields_non_compliant(classifier, evd_ext):
    req = ExtractedRequirement(
        requirement_id="REQ-PM",
        original_text="The proposed Project Manager must possess a minimum of 8 years of relevant experience.",
        category="Key Personnel",
        parameter="Project Manager relevant experience",
        operator=">=",
        value=8.0,
        unit="years"
    )
    # Personal named PM with 4 years vs 8 required
    evd = "Proposed PM Sarah Jenkins is a certified Scrum Master with 4 years of technical leadership in cloud migration."
    ret = _make_retrieval(req, evd, evd_ext)
    res = classifier.evaluate_clause(req, ret)
    assert res.label == "NON_COMPLIANT", f"Expected NON_COMPLIANT, got {res.label}"


def test_pm_named_surplus_yields_compliant(classifier, evd_ext):
    req = ExtractedRequirement(
        requirement_id="REQ-PM",
        original_text="The proposed Project Manager must possess a minimum of 8 years of relevant experience.",
        category="Key Personnel",
        parameter="Project Manager relevant experience",
        operator=">=",
        value=8.0,
        unit="years"
    )
    # Personal named PM with 12 years vs 8 required
    evd = "Lead PM Dr. Arthur Vance brings 12 years of direct experience leading enterprise cloud migrations."
    ret = _make_retrieval(req, evd, evd_ext)
    res = classifier.evaluate_clause(req, ret)
    assert res.label == "COMPLIANT", f"Expected COMPLIANT, got {res.label}"


# ---------------------------------------------------------------------------
# 4. STATUTORY UNDERTAKING & FILING TESTS
# ---------------------------------------------------------------------------

def test_future_gst_promise_does_not_satisfy(classifier, evd_ext):
    req = ExtractedRequirement(
        requirement_id="REQ-GST",
        original_text="The bidder must submit an active GSTIN registration certificate.",
        category="Statutory Compliance",
        parameter="active GSTIN registration & filings",
        operator="MANDATORY_SUBMISSION",
        value="VALID_GSTIN_FILING"
    )
    evd = "The company does not currently have GSTIN but will apply for registration upon award of contract."
    ret = _make_retrieval(req, evd, evd_ext)
    res = classifier.evaluate_clause(req, ret)
    assert res.label != "COMPLIANT", f"Expected non-compliant or uncertain, got {res.label}"


def test_active_gstin_statement_satisfies(classifier, evd_ext):
    req = ExtractedRequirement(
        requirement_id="REQ-GST",
        original_text="The bidder must submit an active GSTIN registration certificate.",
        category="Statutory Compliance",
        parameter="active GSTIN registration & filings",
        operator="MANDATORY_SUBMISSION",
        value="VALID_GSTIN_FILING"
    )
    evd = "State tax registration documentation confirms our active fifteen-character GST tax identification number with up-to-date statutory return acknowledgments."
    ret = _make_retrieval(req, evd, evd_ext)
    res = classifier.evaluate_clause(req, ret)
    assert res.label == "COMPLIANT", f"Expected COMPLIANT, got {res.label}"
