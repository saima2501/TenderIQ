"""
test_adversarial_stress.py - Pytest Verification Suite for Step 2F Adversarial Stress Tests
Ensures zero critical safety failures (False Compliant) and checks category-level performance.
"""

import pytest
from backend.evaluation.adversarial_suite import evaluate_adversarial_suite, get_adversarial_cases


@pytest.fixture(scope="module")
def adversarial_report():
    """Runs the 52-case adversarial evaluation suite once for the test module."""
    return evaluate_adversarial_suite()


def test_zero_critical_safety_failures(adversarial_report):
    """Critical Procurement Safeguard: No non-compliant or unverified claim may yield COMPLIANT."""
    assert adversarial_report["false_compliant_count"] == 0, (
        f"Critical safety failure: {adversarial_report['false_compliant_count']} cases mistakenly granted COMPLIANT"
    )


def test_adversarial_accuracy_threshold(adversarial_report):
    """Stress testing overall accuracy should exceed 95%."""
    assert adversarial_report["accuracy"] >= 95.0, (
        f"Expected accuracy >= 95.0%, got {adversarial_report['accuracy']:.2f}%"
    )


def test_all_12_categories_evaluated(adversarial_report):
    """Verifies that all 12 adversarial categories are present and passing."""
    stats = adversarial_report["category_stats"]
    assert len(stats) == 12, f"Expected 12 categories, found {len(stats)}"
    for cat, data in stats.items():
        assert data["total"] > 0, f"Category {cat} has no test cases"
        assert data["failed"] == 0, f"Category {cat} had failures: {data}"


def test_total_case_count():
    """Verifies standard 52 adversarial test cases."""
    cases = get_adversarial_cases()
    assert len(cases) == 52, f"Expected 52 cases, got {len(cases)}"


@pytest.mark.parametrize("case_id,req_text,evd_text,expected_label", [
    ("A", "The MTTR must not exceed 2 hours.", "The system achieves an average MTTR of 90 minutes.", "COMPLIANT"),
    ("B", "The MTTR must not exceed 90 minutes.", "Our platform achieves an average MTTR of 2 hours.", "NON_COMPLIANT"),
    ("C", "The MTTR must not exceed 90 minutes.", "Our platform achieves an average MTTR of 90 minutes.", "COMPLIANT"),
    ("D", "The MTTR must be no more than 90 minutes.", "Our average restoration time is 1.5 hours.", "COMPLIANT"),
    ("E", "The MTTR must be at least 2 hours.", "Our platform MTTR is 90 minutes.", "NON_COMPLIANT"),
])
def test_mttr_directional_numeric_behavior(case_id, req_text, evd_text, expected_label):
    """Directional Numeric Audit: Verifies MTTR evaluations depend strictly on mathematical comparison

    and unit conversion, never on symmetric semantic similarity.
    """
    from backend.extraction_nlp.requirement_extractor import RequirementExtractor
    from backend.extraction_nlp.evidence_extractor import EvidenceExtractor
    from backend.matching.tfidf_retriever import RequirementRetrievalResult, RetrievedEvidenceCandidate
    from backend.classification.baseline_classifier import BaselineClassifier

    req_ext = RequirementExtractor()
    evd_ext = EvidenceExtractor()
    clf = BaselineClassifier(enable_nli=False, enable_reconciliation=False)

    req_objs = req_ext.extract_from_text(req_text)
    assert len(req_objs) > 0, f"Failed to extract requirement from '{req_text}'"
    req_obj = req_objs[0]

    evd_objs = evd_ext.extract_from_text(evd_text)
    assert len(evd_objs) > 0, f"Failed to extract evidence from '{evd_text}'"
    evd_obj = evd_objs[0]

    cand = RetrievedEvidenceCandidate(
        rank=1,
        evidence_id=f"EVD-MTTR-{case_id}",
        sentence=evd_text,
        similarity_score=0.90,
        source_document="BIDDER-PROPOSAL.pdf",
        page_number=1,
        numeric_values=evd_obj.numeric_values
    )
    ret = RequirementRetrievalResult(
        requirement_id=f"REQ-MTTR-{case_id}",
        requirement_text=req_text,
        parameter=req_obj.parameter,
        candidates=[cand],
        top_similarity=0.90,
        has_candidate=True
    )
    res = clf.evaluate_clause(req_obj, ret, bidder_name="Alpha Systems Ltd")

    assert res.label == expected_label, (
        f"Case {case_id} failed: Expected {expected_label}, got {res.label}. "
        f"Numeric comparison: {res.numeric_comparison}. Reasoning: {res.reasoning}"
    )
    assert res.numeric_comparison is not None, f"Case {case_id} failed to trigger numeric comparison"
    assert res.numeric_comparison["is_applicable"] is True


@pytest.mark.parametrize("test_num,req_text,evd_text,expected_label,expected_op", [
    ("1", "The MTTR must be under 2 hours.", "Our average restoration time is 119 minutes.", "COMPLIANT", "<"),
    ("2", "The MTTR must be under 2 hours.", "Our average restoration time is exactly 120 minutes.", "NON_COMPLIANT", "<"),
    ("3", "The MTTR must not exceed 2 hours.", "Our average restoration time is exactly 120 minutes.", "COMPLIANT", "<="),
])
def test_mttr_strict_inequality_boundary_check(test_num, req_text, evd_text, expected_label, expected_op):
    """Boundary Audit: Verifies 'under 2 hours' is interpreted as strict less-than (<),

    whereas 'not exceed 2 hours' is interpreted as less-than-or-equal (<=).
    """
    from backend.extraction_nlp.requirement_extractor import RequirementExtractor
    from backend.extraction_nlp.evidence_extractor import EvidenceExtractor
    from backend.matching.tfidf_retriever import RequirementRetrievalResult, RetrievedEvidenceCandidate
    from backend.classification.baseline_classifier import BaselineClassifier

    req_ext = RequirementExtractor()
    evd_ext = EvidenceExtractor()
    clf = BaselineClassifier(enable_nli=False, enable_reconciliation=False)

    req_obj = req_ext.extract_from_text(req_text)[0]
    assert req_obj.operator == expected_op, (
        f"Test {test_num}: Expected operator '{expected_op}', got '{req_obj.operator}'"
    )

    evd_obj = evd_ext.extract_from_text(evd_text)[0]
    cand = RetrievedEvidenceCandidate(
        rank=1,
        evidence_id=f"EVD-BOUND-{test_num}",
        sentence=evd_text,
        similarity_score=0.90,
        source_document="BIDDER-PROPOSAL.pdf",
        page_number=1,
        numeric_values=evd_obj.numeric_values
    )
    ret = RequirementRetrievalResult(
        requirement_id=f"REQ-BOUND-{test_num}",
        requirement_text=req_text,
        parameter=req_obj.parameter,
        candidates=[cand],
        top_similarity=0.90,
        has_candidate=True
    )
    res = clf.evaluate_clause(req_obj, ret, bidder_name="Alpha Systems Ltd")

    assert res.label == expected_label, (
        f"Test {test_num} failed: Expected {expected_label}, got {res.label}. "
        f"Numeric comparison: {res.numeric_comparison}. Reasoning: {res.reasoning}"
    )
    assert res.numeric_comparison is not None
    assert res.numeric_comparison["operator"] == expected_op
