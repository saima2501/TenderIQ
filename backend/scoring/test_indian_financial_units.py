"""
test_indian_financial_units.py - Comprehensive Unit & Integration Tests for Indian Financial Units
Part of Phase 8: Core NLP Reliability Improvements (Step 2D)

Tests:
1. Deterministic Numeric Comparison for INR, Lakh, and Crore units in NumericComparator.
2. Cross-scale canonical unit conversion (e.g., 150 Lakh == 1.5 Crore, 1000 Lakh == 10 Crore).
3. Decimal values, currency symbols (₹, Rs., Rs, INR, Rupees) and scale variations (Cr, crore, lac, lakh).
4. Currency isolation: USD vs INR returns INCOMPATIBLE_UNITS.
5. Spelled-out and digit-based number extraction in NumberWordNormalizer, RequirementExtractor, and EvidenceExtractor.
6. End-to-end BaselineClassifier integration preserving Step 2C entity-scope safeguards.
"""

import pytest
from backend.scoring.numeric_comparator import NumericComparator
from backend.extraction_nlp.number_normalizer import NumberWordNormalizer
from backend.extraction_nlp.requirement_extractor import RequirementExtractor
from backend.extraction_nlp.evidence_extractor import EvidenceExtractor
from backend.classification.baseline_classifier import BaselineClassifier
from backend.matching.tfidf_retriever import RequirementRetrievalResult, RetrievedEvidenceCandidate


@pytest.fixture
def comparator():
    return NumericComparator()


@pytest.fixture
def normalizer():
    return NumberWordNormalizer()


@pytest.fixture
def req_extractor():
    return RequirementExtractor()


@pytest.fixture
def evd_extractor():
    return EvidenceExtractor()


@pytest.fixture
def classifier():
    return BaselineClassifier(enable_nli=False, enable_reconciliation=False)


# ===========================================================================
# 1. Direct NumericComparator Unit Tests
# ===========================================================================
class TestNumericComparatorIndianUnits:
    """Tests deterministic numeric comparisons with Indian financial units."""

    def test_crore_to_crore_satisfied(self, comparator):
        res = comparator.compare(
            required_val=10.0,
            required_unit="INR Crore",
            operator=">=",
            offered_val=15.0,
            offered_unit="INR Crore"
        )
        assert res.status == "SATISFIED"
        assert res.required_converted == 100_000_000.0
        assert res.offered_converted == 150_000_000.0
        assert res.base_unit == "inr"

    def test_crore_to_crore_violated(self, comparator):
        res = comparator.compare(
            required_val=10.0,
            required_unit="INR Crore",
            operator=">=",
            offered_val=8.0,
            offered_unit="INR Crore"
        )
        assert res.status == "VIOLATED"
        assert res.required_converted == 100_000_000.0
        assert res.offered_converted == 80_000_000.0

    def test_lakh_to_crore_conversion_equality(self, comparator):
        """150 Lakhs equals 1.5 Crores."""
        res = comparator.compare(
            required_val=1.5,
            required_unit="INR Crore",
            operator=">=",
            offered_val=150.0,
            offered_unit="INR Lakh"
        )
        assert res.status == "SATISFIED"
        assert res.required_converted == 15_000_000.0
        assert res.offered_converted == 15_000_000.0

    def test_lakh_to_crore_conversion_satisfied(self, comparator):
        """1500 Lakhs (15 Cr) satisfies 10 Cr requirement."""
        res = comparator.compare(
            required_val=10.0,
            required_unit="INR Crore",
            operator=">=",
            offered_val=1500.0,
            offered_unit="INR Lakh"
        )
        assert res.status == "SATISFIED"
        assert res.offered_converted == 150_000_000.0

    def test_lakh_to_crore_conversion_violated(self, comparator):
        """50 Lakhs (0.5 Cr) fails 1 Cr requirement."""
        res = comparator.compare(
            required_val=1.0,
            required_unit="INR Crore",
            operator=">=",
            offered_val=50.0,
            offered_unit="INR Lakh"
        )
        assert res.status == "VIOLATED"
        assert "shortfall" in res.reasoning.lower()

    def test_rupee_symbol_and_abbreviation_units(self, comparator):
        """Tests variations: ₹ cr, Rs. Lakh, lacs, cr."""
        res = comparator.compare(
            required_val=10.0,
            required_unit="₹ crore",
            operator=">=",
            offered_val=12.0,
            offered_unit="Rs. cr"
        )
        assert res.status == "SATISFIED"
        assert res.base_unit == "inr"

        res_lac = comparator.compare(
            required_val=200.0,
            required_unit="lacs",
            operator=">=",
            offered_val=2.5,
            offered_unit="cr"
        )
        assert res_lac.status == "SATISFIED"

    def test_decimal_values_comparison(self, comparator):
        """Tests decimal values: 2.25 crore vs 250 lakh."""
        res = comparator.compare(
            required_val=2.25,
            required_unit="INR Crore",
            operator="<=",
            offered_val=225.0,
            offered_unit="INR Lakh"
        )
        assert res.status == "SATISFIED"
        assert abs(res.required_converted - res.offered_converted) < 1e-4

    def test_currency_isolation_usd_vs_inr(self, comparator):
        """USD cannot be compared directly with INR without official exchange rate."""
        res = comparator.compare(
            required_val=10.0,
            required_unit="USD Million",
            operator=">=",
            offered_val=100.0,
            offered_unit="INR Crore"
        )
        assert res.status == "INCOMPATIBLE_UNITS"
        assert "mismatch" in res.reasoning.lower()


# ===========================================================================
# 2. NumberWordNormalizer Spelled-Out Numbers Tests
# ===========================================================================
class TestNumberWordNormalizerIndianUnits:
    """Tests spelled-out English expressions for Indian units."""

    def test_written_crore_expressions(self, normalizer):
        res = normalizer.extract_normalized_numbers("The firm has annual turnover of at least ten crore rupees.")
        assert len(res) >= 1
        assert res[0].value == 10.0
        assert res[0].unit == "INR Crore"
        assert res[0].operator == ">="

    def test_written_lakh_expressions(self, normalizer):
        res = normalizer.extract_normalized_numbers("Total experience value of fifty lakh rupees.")
        assert len(res) >= 1
        assert res[0].value == 50.0
        assert res[0].unit == "INR Lakh"

    def test_written_decimal_point_crore(self, normalizer):
        res = normalizer.extract_normalized_numbers("Capital reserves of two point five crore rupees.")
        assert len(res) >= 1
        assert res[0].value == 2.5
        assert res[0].unit == "INR Crore"

    def test_digit_with_symbol_crore(self, normalizer):
        res = normalizer.extract_normalized_numbers("Turnover exceeded ₹15 crore in FY24.")
        assert len(res) >= 1
        assert res[0].value == 15.0
        assert res[0].unit == "INR Crore"


# ===========================================================================
# 3. Requirement & Evidence Extraction Tests
# ===========================================================================
class TestExtractionIndianUnits:
    """Tests requirement and evidence extraction with Indian currency expressions."""

    def test_requirement_extraction_rupee_symbol(self, req_extractor):
        reqs = req_extractor.extract_from_text(
            "The bidder must have an average annual turnover of at least ₹10 crore over the last three years."
        )
        assert len(reqs) == 1
        r = reqs[0]
        assert r.value == 10.0
        assert r.unit == "INR Crore"
        assert r.operator == ">="
        assert "turnover" in r.parameter.lower()

    def test_requirement_extraction_lakh_unit(self, req_extractor):
        reqs = req_extractor.extract_from_text(
            "Minimum turnover of 150 lakh INR required."
        )
        assert len(reqs) == 1
        r = reqs[0]
        assert r.value == 150.0
        assert r.unit == "INR Lakh"
        assert r.operator == ">="

    def test_requirement_extraction_spelled_out_number(self, req_extractor):
        reqs = req_extractor.extract_from_text(
            "The bidder must demonstrate audited financial turnover of at least ten crore rupees."
        )
        assert len(reqs) == 1
        r = reqs[0]
        assert r.value == 10.0
        assert r.unit == "INR Crore"
        assert r.operator == ">="

    def test_evidence_extraction_rupee_symbol_and_lakhs(self, evd_extractor):
        records = evd_extractor.extract_from_text(
            "ABC Technologies Pvt. Ltd. has achieved annual audited turnover of ₹15 crore."
        )
        assert len(records) == 1
        metrics = records[0].numeric_values
        assert len(metrics) >= 1
        assert metrics[0]["value"] == 15.0
        assert metrics[0]["unit"] == "INR Crore"

    def test_evidence_extraction_lakhs_unit(self, evd_extractor):
        records = evd_extractor.extract_from_text(
            "ABC Technologies Pvt. Ltd. has achieved annual audited turnover of 150 lakh rupees."
        )
        assert len(records) == 1
        metrics = records[0].numeric_values
        assert len(metrics) >= 1
        assert metrics[0]["value"] == 150.0
        assert metrics[0]["unit"] == "INR Lakh"


# ===========================================================================
# 4. BaselineClassifier Integration & Entity-Scope Safeguard Preservation
# ===========================================================================
class TestClassifierIntegrationIndianUnits:
    """Verifies that BaselineClassifier compares Indian financial metrics accurately
    while strictly preserving Step 2C entity-scope safeguards."""

    def test_same_entity_turnover_compliant(self, classifier, req_extractor, evd_extractor):
        """Same entity ₹15 crore satisfies ₹10 crore requirement -> COMPLIANT."""
        req = req_extractor.extract_from_text("The bidder must have an average annual turnover of at least ₹10 crore.")[0]
        evd = evd_extractor.extract_from_text("ABC Technologies Pvt. Ltd. has achieved annual audited turnover of ₹15 crore.")[0]

        cand = RetrievedEvidenceCandidate(
            rank=1,
            evidence_id=evd.evidence_id,
            sentence=evd.original_sentence,
            similarity_score=0.92,
            source_document="BIDDER-ABC-PROPOSAL.pdf",
            page_number=1,
            numeric_values=evd.numeric_values
        )
        ret = RequirementRetrievalResult(
            requirement_id=req.requirement_id,
            requirement_text=req.original_text,
            parameter=req.parameter,
            candidates=[cand]
        )
        eval_res = classifier.evaluate_clause(req, ret, bidder_name="ABC Technologies Pvt. Ltd.")
        assert eval_res.label == "COMPLIANT"
        assert "satisfies" in eval_res.reasoning.lower()

    def test_cross_scale_turnover_compliant(self, classifier, req_extractor, evd_extractor):
        """150 Lakhs satisfies 1.5 Crore requirement -> COMPLIANT."""
        req = req_extractor.extract_from_text("The bidder must have an average annual turnover of at least ₹1.5 crore.")[0]
        evd = evd_extractor.extract_from_text("ABC Technologies Pvt. Ltd. has achieved annual audited turnover of 150 lakh rupees.")[0]

        cand = RetrievedEvidenceCandidate(
            rank=1,
            evidence_id=evd.evidence_id,
            sentence=evd.original_sentence,
            similarity_score=0.92,
            source_document="BIDDER-ABC-PROPOSAL.pdf",
            page_number=1,
            numeric_values=evd.numeric_values
        )
        ret = RequirementRetrievalResult(
            requirement_id=req.requirement_id,
            requirement_text=req.original_text,
            parameter=req.parameter,
            candidates=[cand]
        )
        eval_res = classifier.evaluate_clause(req, ret, bidder_name="ABC Technologies Pvt. Ltd.")
        assert eval_res.label == "COMPLIANT"

    def test_same_entity_turnover_non_compliant(self, classifier, req_extractor, evd_extractor):
        """50 Lakhs fails 1 Crore requirement -> NON_COMPLIANT."""
        req = req_extractor.extract_from_text("The bidder must have an average annual turnover of at least ₹1 crore.")[0]
        evd = evd_extractor.extract_from_text("ABC Technologies Pvt. Ltd. has achieved annual audited turnover of 50 lakh rupees.")[0]

        cand = RetrievedEvidenceCandidate(
            rank=1,
            evidence_id=evd.evidence_id,
            sentence=evd.original_sentence,
            similarity_score=0.92,
            source_document="BIDDER-ABC-PROPOSAL.pdf",
            page_number=1,
            numeric_values=evd.numeric_values
        )
        ret = RequirementRetrievalResult(
            requirement_id=req.requirement_id,
            requirement_text=req.original_text,
            parameter=req.parameter,
            candidates=[cand]
        )
        eval_res = classifier.evaluate_clause(req, ret, bidder_name="ABC Technologies Pvt. Ltd.")
        assert eval_res.label == "NON_COMPLIANT"
        assert "fails" in eval_res.reasoning.lower() or "shortfall" in eval_res.reasoning.lower()

    def test_parent_company_turnover_preserves_uncertain(self, classifier, req_extractor, evd_extractor):
        """Parent company turnover of ₹100 crore MUST evaluate to UNCERTAIN, preserving Step 2C."""
        req = req_extractor.extract_from_text("The bidder must have an average annual turnover of at least ₹10 crore.")[0]
        evd = evd_extractor.extract_from_text("ABC Holdings Ltd. (parent company) has achieved annual audited turnover of ₹100 crore.")[0]

        cand = RetrievedEvidenceCandidate(
            rank=1,
            evidence_id=evd.evidence_id,
            sentence=evd.original_sentence,
            similarity_score=0.92,
            source_document="BIDDER-ABC-PROPOSAL.pdf",
            page_number=1,
            numeric_values=evd.numeric_values
        )
        ret = RequirementRetrievalResult(
            requirement_id=req.requirement_id,
            requirement_text=req.original_text,
            parameter=req.parameter,
            candidates=[cand]
        )
        eval_res = classifier.evaluate_clause(req, ret, bidder_name="ABC Technologies Pvt. Ltd.")
        assert eval_res.label == "UNCERTAIN"
        assert "parent company" in eval_res.reasoning.lower()

    def test_subsidiary_turnover_preserves_uncertain(self, classifier, req_extractor, evd_extractor):
        """Subsidiary turnover of ₹100 crore MUST evaluate to UNCERTAIN, preserving Step 2C."""
        req = req_extractor.extract_from_text("The bidder must have an average annual turnover of at least ₹10 crore.")[0]
        evd = evd_extractor.extract_from_text("ABC Services Ltd. (subsidiary) has achieved annual audited turnover of ₹100 crore.")[0]

        cand = RetrievedEvidenceCandidate(
            rank=1,
            evidence_id=evd.evidence_id,
            sentence=evd.original_sentence,
            similarity_score=0.92,
            source_document="BIDDER-ABC-PROPOSAL.pdf",
            page_number=1,
            numeric_values=evd.numeric_values
        )
        ret = RequirementRetrievalResult(
            requirement_id=req.requirement_id,
            requirement_text=req.original_text,
            parameter=req.parameter,
            candidates=[cand]
        )
        eval_res = classifier.evaluate_clause(req, ret, bidder_name="ABC Technologies Pvt. Ltd.")
        assert eval_res.label == "UNCERTAIN"
        assert "subsidiary" in eval_res.reasoning.lower()
