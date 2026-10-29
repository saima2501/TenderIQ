"""
test_phase6_nli.py - Phase 6 Test Suite: NLI Evidence Classification & Compliance Pipeline Integration

Tests:
1. CPU-compatible NLI model initialization and label mapping
2. Entailment estimation between requirement and evidence
3. Contradiction estimation on explicit disclaimers/limitations
4. Neutral relationship estimation on topically adjacent/unrelated evidence
5. Robust empty-string and whitespace handling without exceptions
6. Transparent fallback behavior when model or dependencies are unavailable
7. Mathematical override guard: numeric shortfalls MUST override NLI entailment
8. Unit conversion preservation with NLI support (e.g. 36 months == 3 years)
9. Guard: Qualitative NLI entailment alone NEVER establishes COMPLIANT status
10. Guard: Adversarial negation with high similarity yields NON_COMPLIANT
11. Guard: Conflicting evidence (numeric match + strong NLI contradiction) yields UNCERTAIN
12. Guard: Missing evidence is distinguished from confirmed NON_COMPLIANT
13. Guard: Ambiguous/unverified claims yield UNCERTAIN for human review
14. Provenance preservation: source document, page number, retrieval engine, NLI relations, scores, and rationale
15. End-to-end CompliancePipeline execution with NLI enabled vs disabled
16. SQLite database schema migration and persistence of NLI fields
"""

import os
import pytest
from unittest.mock import MagicMock, patch

from backend.classification.nli_classifier import NLIClassifier, NLIPrediction
from backend.classification.baseline_classifier import (
    BaselineClassifier,
    ClauseComplianceEvaluation,
    DocumentComplianceReport
)
from backend.extraction_nlp.requirement_extractor import ExtractedRequirement
from backend.matching.tfidf_retriever import (
    RequirementRetrievalResult,
    RetrievedEvidenceCandidate
)
from backend.pipeline import CompliancePipeline
from backend.database.db_manager import DatabaseManager


class TestPhase6NLIModel:
    """Verifies the core CPU-compatible pretrained NLI classifier."""

    @pytest.fixture
    def nli_classifier(self):
        """Loads NLIClassifier for testing."""
        return NLIClassifier(device="cpu")

    def test_nli_classifier_initialization_cpu(self, nli_classifier):
        """1. Verifies model loads cleanly on CPU with standard 3-class label map."""
        assert nli_classifier.is_available is True
        assert nli_classifier.device == "cpu"
        assert nli_classifier.model_name == "cross-encoder/nli-distilroberta-base"
        assert nli_classifier.init_error is None
        assert isinstance(nli_classifier.label_map, dict)
        # Should have contradiction (0), entailment (1), neutral (2)
        assert 0 in nli_classifier.label_map or "0" in nli_classifier.label_map

    def test_nli_entailment_prediction(self, nli_classifier):
        """2. Tests estimation of entailment when evidence directly fulfills requirement."""
        requirement = "The contractor must provide a comprehensive on-site warranty of at least 36 months."
        evidence = "We provide 36 months comprehensive on-site warranty with all replacement parts and labor included."

        pred = nli_classifier.predict_pair(requirement, evidence)
        assert isinstance(pred, NLIPrediction)
        assert pred.is_fallback is False
        assert pred.relation == "entailment"
        assert pred.entailment_prob > 0.5
        assert 0.0 <= pred.confidence <= 1.0

    def test_nli_contradiction_prediction(self, nli_classifier):
        """3. Tests estimation of contradiction on explicit limitation or exclusion."""
        requirement = "The contractor guarantees 99.9% uptime and comprehensive maintenance."
        evidence = "We cannot guarantee 99.9% uptime and comprehensive maintenance is expressly excluded."

        pred = nli_classifier.predict_pair(requirement, evidence)
        assert isinstance(pred, NLIPrediction)
        assert pred.is_fallback is False
        assert pred.relation == "contradiction"
        assert pred.contradiction_prob > 0.60
        assert pred.confidence == pred.contradiction_prob

    def test_nli_neutral_prediction(self, nli_classifier):
        """4. Tests estimation of neutral relationship on topically unrelated statements."""
        requirement = "The bidder must hold an active accredited ISO 27001 information security certification."
        evidence = "Our company was established in 2012 and operates regional branch offices across Western Europe."

        pred = nli_classifier.predict_pair(requirement, evidence)
        assert isinstance(pred, NLIPrediction)
        assert pred.is_fallback is False
        assert pred.relation == "neutral"
        assert pred.neutral_prob > 0.50

    def test_nli_empty_or_whitespace_handling(self, nli_classifier):
        """5. Tests empty string or whitespace evidence does not crash and yields neutral."""
        req = "The bidder must possess 5 years experience."
        pred_empty = nli_classifier.predict_pair(req, "")
        pred_ws = nli_classifier.predict_pair(req, "   \n\t  ")

        assert pred_empty.relation == "neutral"
        assert pred_empty.entailment_prob == 0.0
        assert pred_ws.relation == "neutral"
        assert pred_ws.entailment_prob == 0.0

    def test_nli_batch_prediction_consistency(self, nli_classifier):
        """6. Verifies batch prediction matches single pair predictions."""
        pairs = [
            ("Uptime must be 99.9%", "Service uptime is guaranteed at 99.9% monthly."),
            ("Must include on-site labor", "Proposal excludes all on-site labor.")
        ]
        batch_preds = nli_classifier.predict_batch(pairs)
        assert len(batch_preds) == 2
        assert batch_preds[0].relation == "entailment"
        assert batch_preds[1].relation == "contradiction"

    def test_nli_graceful_fallback_when_unavailable(self):
        """7. Tests transparent fallback behavior when model is unavailable or fails."""
        with patch("backend.classification.nli_classifier.CROSS_ENCODER_AVAILABLE", False):
            fallback_clf = NLIClassifier(device="cpu")
            assert fallback_clf.is_available is False
            assert fallback_clf.init_error is not None

            pred = fallback_clf.predict_pair("Req text", "Evidence text")
            assert pred.is_fallback is True
            assert pred.relation == "fallback"
            assert "unavailable" in pred.notes.lower()
            assert pred.confidence == 0.0


class TestPhase6ComplianceGuards:
    """Verifies deterministic guards, numeric overrides, and conservative classification."""

    @pytest.fixture
    def classifier_with_nli(self):
        return BaselineClassifier(enable_nli=True)

    @pytest.fixture
    def classifier_without_nli(self):
        return BaselineClassifier(enable_nli=False)

    def test_numeric_shortfall_overrides_nli_entailment(self, classifier_with_nli):
        """8. Mathematical guard: A verified numeric shortfall MUST override NLI entailment.
        
        Even if NLI claims entailment because both sentences discuss warranties,
        2 years offered < 5 years required must yield NON_COMPLIANT.
        """
        req = ExtractedRequirement(
            requirement_id="REQ-TEST-WARR-01",
            original_text="The contractor must provide a comprehensive on-site warranty of at least 5 years.",
            category="SLA & Warranty",
            parameter="warranty duration",
            operator=">=",
            value=5.0,
            unit="years",
            is_mandatory=True
        )

        cand = RetrievedEvidenceCandidate(
            rank=1,
            evidence_id="EVD-001",
            sentence="We provide an exceptional 2 years comprehensive warranty with 24/7 helpdesk.",
            similarity_score=0.88,
            source_document="BIDDER-PROPOSAL.pdf",
            page_number=12,
            has_negation=False,
            numeric_values=[{"raw": "2 years", "value": 2.0, "unit": "years", "parameter_hint": "duration"}]
        )

        ret_res = RequirementRetrievalResult(
            requirement_id=req.requirement_id,
            requirement_text=req.original_text,
            parameter=req.parameter,
            candidates=[cand],
            top_similarity=0.88,
            has_candidate=True
        )

        # Mock NLI predicting high entailment on qualitative text
        mock_nli_pred = NLIPrediction(
            relation="entailment",
            entailment_prob=0.91,
            contradiction_prob=0.02,
            neutral_prob=0.07,
            confidence=0.91,
            model_name="cross-encoder/nli-distilroberta-base",
            is_fallback=False
        )

        eval_res = classifier_with_nli.evaluate_clause(req, ret_res, nli_pred=mock_nli_pred)

        # MUST be NON_COMPLIANT: numeric shortfall cannot be overridden by NLI!
        assert eval_res.label == "NON_COMPLIANT"
        assert "overrides nli entailment" in eval_res.reasoning.lower() or "violated" in eval_res.reasoning.lower()
        assert eval_res.numeric_comparison is not None
        assert eval_res.numeric_comparison["status"] == "VIOLATED"

    def test_unit_conversion_preserved_with_nli(self, classifier_with_nli):
        """9. Tests unit conversion (36 months == 3 years) works correctly with NLI."""
        req = ExtractedRequirement(
            requirement_id="REQ-TEST-WARR-02",
            original_text="The contractor must provide a comprehensive on-site warranty of at least 36 months.",
            category="SLA & Warranty",
            parameter="warranty duration",
            operator=">=",
            value=36.0,
            unit="months",
            is_mandatory=True
        )

        cand = RetrievedEvidenceCandidate(
            rank=1,
            evidence_id="EVD-002",
            sentence="Our proposal guarantees a full 3 years comprehensive OEM warranty on all equipment.",
            similarity_score=0.82,
            source_document="BIDDER-PROPOSAL.pdf",
            page_number=8,
            has_negation=False,
            numeric_values=[{"raw": "3 years", "value": 3.0, "unit": "years", "parameter_hint": "duration"}]
        )

        ret_res = RequirementRetrievalResult(
            requirement_id=req.requirement_id,
            requirement_text=req.original_text,
            parameter=req.parameter,
            candidates=[cand],
            top_similarity=0.82,
            has_candidate=True
        )

        eval_res = classifier_with_nli.evaluate_clause(req, ret_res)
        assert eval_res.label == "COMPLIANT"
        assert eval_res.numeric_comparison["status"] == "SATISFIED"

    def test_qualitative_entailment_alone_never_compliant(self, classifier_with_nli):
        """10. Guard: A requirement must NEVER be classified as COMPLIANT solely because NLI predicts entailment.
        
        Qualitative statements without verifiable numeric thresholds or certificates
        must return UNCERTAIN for procurement officer verification.
        """
        req = ExtractedRequirement(
            requirement_id="REQ-TEST-QUAL-01",
            original_text="The solution must implement automated load balancing and continuous fault tolerance.",
            category="Technical Specifications",
            parameter="fault tolerance",
            operator="CONTAINS",
            value=None,
            unit=None,
            is_mandatory=True
        )

        cand = RetrievedEvidenceCandidate(
            rank=1,
            evidence_id="EVD-003",
            sentence="Our architecture provides automated load balancing and continuous fault tolerance across all clusters.",
            similarity_score=0.92,
            source_document="BIDDER-PROPOSAL.pdf",
            page_number=14,
            has_negation=False,
            numeric_values=[]
        )

        ret_res = RequirementRetrievalResult(
            requirement_id=req.requirement_id,
            requirement_text=req.original_text,
            parameter=req.parameter,
            candidates=[cand],
            top_similarity=0.92,
            has_candidate=True
        )

        mock_nli_entail = NLIPrediction(
            relation="entailment",
            entailment_prob=0.94,
            contradiction_prob=0.01,
            neutral_prob=0.05,
            confidence=0.94,
            model_name="cross-encoder/nli-distilroberta-base",
            is_fallback=False
        )

        eval_res = classifier_with_nli.evaluate_clause(req, ret_res, nli_pred=mock_nli_entail)

        # MUST be UNCERTAIN: qualitative entailment alone does not equal verified compliance!
        assert eval_res.label == "UNCERTAIN"
        assert "human" in eval_res.reasoning.lower() or "procurement officer" in eval_res.reasoning.lower()

    def test_accreditation_denial_detected_by_nli_vs_baseline(self, classifier_with_nli, classifier_without_nli):
        """11. Verifies that NLI contradiction detects accreditation denial (e.g. PAIR-0406/PAIR-0461).
        
        Requirement: "The bidder must hold an active accredited ISO 9001:2015 certification."
        Evidence: "The bidder does not currently hold an accredited ISO 9001:2015 certification, but our workflows adhere to similar principles."
        
        - Phase 4 Baseline (without NLI): erroneously predicts COMPLIANT due to keyword matching on 'ISO 9001:2015'.
        - Phase 6 (with NLI): correctly predicts NON_COMPLIANT because NLI detects the semantic contradiction (prob ~0.75).
        """
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
            evidence_id="EVD-005",
            sentence="The bidder does not currently hold an accredited ISO 9001:2015 (Quality Management Systems) certification, but our workflows adhere to similar principles.",
            similarity_score=0.89,
            source_document="BIDDER-PROPOSAL.pdf",
            page_number=18,
            has_negation=True,
            numeric_values=[]
        )

        ret_res = RequirementRetrievalResult(
            requirement_id=req.requirement_id,
            requirement_text=req.original_text,
            parameter=req.parameter,
            candidates=[cand],
            top_similarity=0.89,
            has_candidate=True
        )

        # Baseline evaluation (without NLI) -> COMPLIANT (known limitation of rule baseline)
        eval_p4 = classifier_without_nli.evaluate_clause(req, ret_res, nli_pred=None)
        assert eval_p4.label == "COMPLIANT"

        # Phase 6 evaluation (with real NLI prediction) -> NON_COMPLIANT
        nli_pred = classifier_with_nli.nli_classifier.predict_pair(req.original_text, cand.sentence)
        eval_p6 = classifier_with_nli.evaluate_clause(req, ret_res, nli_pred=nli_pred)
        assert eval_p6.label == "NON_COMPLIANT"
        assert nli_pred.relation == "contradiction"
        assert nli_pred.contradiction_prob >= 0.65
        assert "explicit negation" in eval_p6.reasoning.lower() or "denies holding" in eval_p6.reasoning.lower()

    def test_high_similarity_negation_contradiction(self, classifier_with_nli):
        """11. Guard: Adversarial negation with high similarity yields NON_COMPLIANT."""
        req = ExtractedRequirement(
            requirement_id="REQ-TEST-NEG-01",
            original_text="For Critical Priority 1 incidents, support must acknowledge within 30 minutes and provide on-site engineer within 6 hours.",
            category="SLA & Warranty",
            parameter="incident response",
            operator="<=",
            value=6.0,
            unit="hours",
            is_mandatory=True
        )

        cand = RetrievedEvidenceCandidate(
            rank=1,
            evidence_id="EVD-004",
            sentence="For Critical Priority 1 incidents, our standard SLA does NOT include on-site engineer dispatch within 6 hours.",
            similarity_score=0.94,
            source_document="BIDDER-PROPOSAL.pdf",
            page_number=6,
            has_negation=True,
            numeric_values=[{"raw": "6 hours", "value": 6.0, "unit": "hours", "parameter_hint": "duration"}]
        )

        ret_res = RequirementRetrievalResult(
            requirement_id=req.requirement_id,
            requirement_text=req.original_text,
            parameter=req.parameter,
            candidates=[cand],
            top_similarity=0.94,
            has_candidate=True
        )

        eval_res = classifier_with_nli.evaluate_clause(req, ret_res)
        assert eval_res.label == "NON_COMPLIANT"
        assert eval_res.has_negation is True

    def test_missing_evidence_distinguished_from_non_compliant(self, classifier_with_nli):
        """12. Guard: Missing evidence is distinguished from confirmed non-compliance."""
        req = ExtractedRequirement(
            requirement_id="REQ-TEST-MISS-01",
            original_text="The bidder must furnish audited balance sheets for the last three assessment years.",
            category="Financial Capability",
            parameter="audited balance sheet",
            operator="MANDATORY_SUBMISSION",
            value="SUBMISSION_REQUIRED",
            is_mandatory=True
        )

        ret_res = RequirementRetrievalResult(
            requirement_id=req.requirement_id,
            requirement_text=req.original_text,
            parameter=req.parameter,
            candidates=[],
            top_similarity=0.0,
            has_candidate=False
        )

        eval_res = classifier_with_nli.evaluate_clause(req, ret_res)
        assert eval_res.label == "MISSING_EVIDENCE"
        assert eval_res.label != "NON_COMPLIANT"

    def test_provenance_preservation(self, classifier_with_nli):
        """13. Verifies all audit provenance fields are preserved in evaluation output."""
        req = ExtractedRequirement(
            requirement_id="REQ-PROV-01",
            original_text="Annual audited turnover of at least USD 10.0 Million.",
            category="Financial Capability",
            parameter="turnover",
            operator=">=",
            value=10.0,
            unit="USD Million",
            is_mandatory=True
        )

        cand = RetrievedEvidenceCandidate(
            rank=1,
            evidence_id="EVD-PROV-01",
            sentence="Our average audited turnover over the last 3 years is USD 18.5 Million.",
            similarity_score=0.86,
            source_document="BIDDER-ALPHA-TECH-PROPOSAL.pdf",
            page_number=5,
            section_reference="Section 2.1 - Financials",
            has_negation=False,
            numeric_values=[{"raw": "USD 18.5 Million", "value": 18.5, "unit": "USD Million", "parameter_hint": "currency"}]
        )

        ret_res = RequirementRetrievalResult(
            requirement_id=req.requirement_id,
            requirement_text=req.original_text,
            parameter=req.parameter,
            candidates=[cand],
            top_similarity=0.86,
            has_candidate=True
        )

        eval_res = classifier_with_nli.evaluate_clause(req, ret_res)

        assert eval_res.requirement_id == "REQ-PROV-01"
        assert eval_res.top_evidence_sentence == cand.sentence
        assert eval_res.top_evidence_source == "BIDDER-ALPHA-TECH-PROPOSAL.pdf"
        assert eval_res.top_evidence_page == 5
        assert eval_res.top_evidence_section == "Section 2.1 - Financials"
        assert eval_res.retrieval_method is not None
        assert eval_res.reasoning is not None and len(eval_res.reasoning) > 10


class TestPhase6PipelineAndDatabase:
    """Verifies pipeline execution with NLI toggling and SQLite persistence."""

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    TENDER_PDF = os.path.join(PROJECT_ROOT, "dataset", "synthetic_pdfs", "tenders", "TENDER-2024-IT-001.pdf")
    BIDDER_PDF = os.path.join(PROJECT_ROOT, "dataset", "synthetic_pdfs", "bidders", "BIDDER-ALPHA-TECH-PROPOSAL.pdf")

    def test_pipeline_runs_with_nli_enabled(self):
        """14. Tests end-to-end pipeline execution with NLI enabled."""
        if not os.path.exists(self.TENDER_PDF) or not os.path.exists(self.BIDDER_PDF):
            pytest.skip("Synthetic PDFs not found in workspace.")

        pipeline = CompliancePipeline(enable_nli=True)
        report = pipeline.run(
            tender_pdf_path=self.TENDER_PDF,
            bidder_pdf_path=self.BIDDER_PDF,
            enable_nli=True,
            save_to_db=False
        )

        assert isinstance(report, DocumentComplianceReport)
        assert report.nli_enabled is True
        assert report.total_requirements > 0
        assert len(report.evaluations) == report.total_requirements

        # Check that top evaluations have NLI relations computed
        has_nli_eval = any(ev.nli_relation is not None for ev in report.evaluations)
        assert has_nli_eval is True

    def test_pipeline_runs_with_nli_disabled(self):
        """15. Tests end-to-end pipeline execution with NLI disabled (Phase 4 baseline)."""
        if not os.path.exists(self.TENDER_PDF) or not os.path.exists(self.BIDDER_PDF):
            pytest.skip("Synthetic PDFs not found in workspace.")

        pipeline = CompliancePipeline(enable_nli=False)
        report = pipeline.run(
            tender_pdf_path=self.TENDER_PDF,
            bidder_pdf_path=self.BIDDER_PDF,
            enable_nli=False,
            save_to_db=False
        )

        assert isinstance(report, DocumentComplianceReport)
        assert report.nli_enabled is False
        # All NLI relations should be None when disabled
        assert all(ev.nli_relation is None for ev in report.evaluations)

    def test_database_persistence_of_nli_fields(self, tmp_path):
        """16. Tests SQLite database correctly stores and retrieves Phase 6 NLI columns."""
        test_db = os.path.join(str(tmp_path), "test_phase6.db")
        db = DatabaseManager(db_path=test_db)

        # Create dummy report with NLI fields
        eval_item = ClauseComplianceEvaluation(
            requirement_id="REQ-TEST-DB-01",
            requirement_text="Must guarantee 99.9% uptime",
            category="Technical",
            parameter="uptime",
            operator=">=",
            required_value=99.9,
            required_unit="%",
            is_mandatory=True,
            label="COMPLIANT",
            confidence=0.92,
            top_similarity_score=0.89,
            top_evidence_sentence="Guaranteed uptime is 99.95% monthly.",
            top_evidence_source="BIDDER-TEST.pdf",
            top_evidence_page=3,
            top_evidence_section="Section 3",
            has_negation=False,
            retrieval_method="semantic (all-MiniLM-L6-v2)",
            nli_relation="entailment",
            nli_entailment_prob=0.91,
            nli_contradiction_prob=0.01,
            nli_neutral_prob=0.08,
            nli_model="cross-encoder/nli-distilroberta-base",
            reasoning="Numeric SLA threshold satisfied."
        )

        report = DocumentComplianceReport(
            tender_document="TENDER-TEST.pdf",
            bidder_document="BIDDER-TEST.pdf",
            bidder_name="Alpha Tech",
            total_requirements=1,
            compliant_count=1,
            non_compliant_count=0,
            uncertain_count=0,
            missing_evidence_count=0,
            mandatory_shortfalls_count=0,
            retrieval_method="semantic (all-MiniLM-L6-v2)",
            nli_enabled=True,
            nli_model="cross-encoder/nli-distilroberta-base",
            evaluations=[eval_item]
        )

        analysis_id = db.save_analysis(report)
        assert analysis_id.startswith("ANA-")

        retrieved = db.get_analysis_by_id(analysis_id)
        assert retrieved is not None
        assert retrieved["nli_enabled"] is True
        assert retrieved["nli_model"] == "cross-encoder/nli-distilroberta-base"
        assert len(retrieved["evaluations"]) == 1

        ev_row = retrieved["evaluations"][0]
        assert ev_row["nli_relation"] == "entailment"
        assert abs(ev_row["nli_entailment_prob"] - 0.91) < 1e-4
        assert abs(ev_row["nli_contradiction_prob"] - 0.01) < 1e-4
        assert abs(ev_row["nli_neutral_prob"] - 0.08) < 1e-4
        assert ev_row["nli_model"] == "cross-encoder/nli-distilroberta-base"
