"""
test_phase5_semantic.py - Comprehensive Unit & Integration Test Suite for Phase 5
Part of Phase 5: Sentence-BERT Semantic Evidence Retrieval

Tests:
1. Model loading & CPU execution verification.
2. Paraphrase evidence retrieval & top-3 ranking.
3. Full audit traceability (page number, source document, retrieval method).
4. Adversarial negation detection (high similarity disclaimers).
5. Numeric conflict handling (similarity overridden by numeric comparator).
6. Compliance guard: High similarity alone NEVER produces COMPLIANT.
7. MISSING_EVIDENCE vs confirmed NON_COMPLIANT distinction.
8. Graceful TF-IDF fallback when model is unavailable or fails.
9. Offline / missing dependency error transparency.
10. End-to-end pipeline integration & method switching ("semantic" vs "tfidf").
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from backend.matching.semantic_retriever import SemanticRetriever
from backend.matching.tfidf_retriever import TFIDFRetriever
from backend.extraction_nlp.requirement_extractor import ExtractedRequirement
from backend.extraction_nlp.evidence_extractor import ExtractedEvidence
from backend.classification.baseline_classifier import BaselineClassifier
from backend.pipeline import CompliancePipeline


# Helper fixtures & factories
def make_req(req_id: str, text: str, cat: str = "Technical", param: str = "", val=None, unit=None, op=">="):
    return ExtractedRequirement(
        requirement_id=req_id,
        category=cat,
        parameter=param,
        operator=op,
        value=val,
        unit=unit,
        is_mandatory=True,
        page_number=1,
        source_document="TENDER.pdf",
        original_text=text
    )


def make_evd(evd_id: str, sentence: str, page: int = 3, doc: str = "BIDDER.pdf", neg: bool = False, metrics=None):
    return ExtractedEvidence(
        evidence_id=evd_id,
        source_document=doc,
        page_number=page,
        section_reference=f"Section {page}.1",
        original_sentence=sentence,
        has_negation=neg,
        extracted_entities=[],
        numeric_values=metrics or [],
        uncertainty_flag=False,
        review_notes=""
    )


@pytest.fixture(scope="module")
def semantic_retriever():
    """Instantiates SemanticRetriever on CPU once for test module."""
    retriever = SemanticRetriever(
        model_name="all-MiniLM-L6-v2",
        min_similarity_threshold=0.25,
        device="cpu",
        fallback_to_tfidf=True
    )
    return retriever


@pytest.fixture(scope="module")
def classifier():
    """Instantiates BaselineClassifier with Phase 5 semantic threshold."""
    return BaselineClassifier(min_retrieval_threshold=0.08, min_semantic_threshold=0.25)


class TestPhase5SemanticRetrieval:
    """Phase 5 Semantic Evidence Retrieval Test Suite."""

    def test_semantic_retriever_initialization_cpu(self, semantic_retriever):
        """Verify model loads on CPU with valid status and method tag."""
        assert semantic_retriever.is_available is True
        assert semantic_retriever.init_error is None
        assert semantic_retriever.device == "cpu"
        assert "all-MiniLM-L6-v2" in semantic_retriever.retrieval_method_name
        assert "semantic" in semantic_retriever.retrieval_method_name

    def test_paraphrase_retrieval_ranking(self, semantic_retriever):
        """Verify Sentence-BERT matches paraphrased clause despite low lexical token overlap."""
        # Requirement uses formal procurement terminology
        req = make_req(
            "REQ-SLA-01",
            "The vendor must maintain round-the-clock technical helpline and helpdesk support services.",
            param="24x7 Support"
        )

        # Candidates: distractor, exact paraphrase, unrelated
        candidates = [
            make_evd("EVD-01", "The company operates three data centers located in Mumbai and Hyderabad."),
            make_evd("EVD-02", "Our client care team provides 24x7 nonstop continuous assistance via phone and web portal."),
            make_evd("EVD-03", "All server racks are equipped with dual redundant power supply units.")
        ]

        results = semantic_retriever.retrieve_top_k([req], candidates, top_k=3)
        assert len(results) == 1
        top_res = results[0]
        assert top_res.has_candidate is True
        assert len(top_res.candidates) == 3

        # The paraphrase (EVD-02) should be ranked #1
        assert top_res.candidates[0].evidence_id == "EVD-02"
        assert top_res.candidates[0].similarity_score > top_res.candidates[1].similarity_score
        assert top_res.candidates[0].similarity_score >= 0.50

    def test_top_3_ranking_and_score_order(self, semantic_retriever):
        """Verify exactly top-3 candidates are returned in strictly descending similarity order."""
        req = make_req(
            "REQ-WAR-01",
            "Bidder must provide 36 months comprehensive on-site OEM warranty.",
            param="Warranty Duration"
        )
        candidates = [
            make_evd("EVD-A", "We provide standard manufacturer support."),
            make_evd("EVD-B", "Full comprehensive on-site warranty is guaranteed for 36 months."),
            make_evd("EVD-C", "Warranty covers replacement of defective parts within 24 hours."),
            make_evd("EVD-D", "Our corporate headquarters was established in 2012.")
        ]

        results = semantic_retriever.retrieve_top_k([req], candidates, top_k=3)
        cands = results[0].candidates
        assert len(cands) == 3
        assert cands[0].similarity_score >= cands[1].similarity_score >= cands[2].similarity_score
        for c in cands:
            assert 0.0 <= c.similarity_score <= 1.0
            assert c.rank in (1, 2, 3)

    def test_page_and_source_traceability(self, semantic_retriever):
        """Verify evidence metadata (page, filename, method) is preserved with high fidelity."""
        req = make_req("REQ-SEC-01", "Storage systems must use AES-256 encryption at rest.", param="Data Encryption")
        candidates = [
            make_evd("EVD-SEC-12", "All volumes are encrypted at rest with hardware-accelerated AES-256.", page=14, doc="PROPOSAL-GAMMA.pdf")
        ]

        results = semantic_retriever.retrieve_top_k([req], candidates, top_k=1)
        top = results[0].candidates[0]

        assert top.page_number == 14
        assert top.source_document == "PROPOSAL-GAMMA.pdf"
        assert top.section_reference == "Section 14.1"
        assert "semantic" in top.retrieval_method
        assert "all-MiniLM-L6-v2" in top.retrieval_method

    def test_adversarial_negation_retrieval_and_classification(self, semantic_retriever, classifier):
        """Verify high semantic similarity disclaimers are detected as NON_COMPLIANT, not COMPLIANT."""
        req = make_req(
            "REQ-DIS-01",
            "Vendor must provide 24/7 on-site emergency disaster response support.",
            param="Disaster Response"
        )
        # Bidder proposal mentions disaster response with high semantic similarity, but expressly disclaims it
        candidates = [
            make_evd(
                "EVD-NEG-01",
                "Our service level agreement does not include 24/7 on-site emergency disaster response support.",
                neg=True
            )
        ]

        ret_res = semantic_retriever.retrieve_top_k([req], candidates, top_k=1)
        assert ret_res[0].top_similarity >= 0.55  # Strong semantic overlap (> 0.60 actual)

        eval_res = classifier.evaluate_clause(req, ret_res[0])
        assert eval_res.label == "NON_COMPLIANT"
        assert eval_res.has_negation is True
        assert "negation" in eval_res.reasoning.lower() or "disclaims" in eval_res.reasoning.lower()

    def test_numeric_conflict_overrides_high_similarity(self, semantic_retriever, classifier):
        """Verify numeric comparator overrides high similarity when offered metric violates requirement."""
        req = make_req(
            "REQ-NUM-01",
            "The contractor must provide a comprehensive on-site warranty of at least 5 years.",
            param="Warranty Duration",
            val=5.0,
            unit="years",
            op=">="
        )
        # Bidder states high-similarity sentence but offers only 2 years
        candidates = [
            make_evd(
                "EVD-NUM-01",
                "We provide a comprehensive on-site warranty of 2 years post-commissioning.",
                metrics=[{"raw": "2 years", "value": 2.0, "unit": "years", "confidence": 0.95}]
            )
        ]

        ret_res = semantic_retriever.retrieve_top_k([req], candidates, top_k=1)
        assert ret_res[0].top_similarity >= 0.55  # High similarity (> 0.60 actual)

        eval_res = classifier.evaluate_clause(req, ret_res[0])
        # Numeric violation must govern: 2 years < 5 years
        assert eval_res.label == "NON_COMPLIANT"
        assert eval_res.numeric_comparison is not None
        assert eval_res.numeric_comparison["status"] == "VIOLATED"

    def test_similarity_alone_never_establishes_compliance(self, semantic_retriever, classifier):
        """CRITICAL: High semantic similarity on vague/promotional claims must NEVER label COMPLIANT."""
        req = make_req(
            "REQ-QUAL-01",
            "The cloud architecture must demonstrate proven high reliability and zero downtime capability.",
            param="Reliability Architecture"
        )
        # Promotional statement with zero empirical metrics or third-party audits
        candidates = [
            make_evd(
                "EVD-PROMO-01",
                "Alpha Cloud demonstrates proven high reliability and unmatched zero downtime architecture for all clients."
            )
        ]

        ret_res = semantic_retriever.retrieve_top_k([req], candidates, top_k=1)
        assert ret_res[0].top_similarity >= 0.75  # Very high semantic similarity

        eval_res = classifier.evaluate_clause(req, ret_res[0])
        # MUST NOT BE COMPLIANT solely because similarity is high
        assert eval_res.label != "COMPLIANT"
        assert eval_res.label == "UNCERTAIN"
        assert "evaluator" in eval_res.reasoning.lower() or "human" in eval_res.reasoning.lower() or "vague" in eval_res.reasoning.lower()

    def test_missing_evidence_distinct_from_non_compliant(self, semantic_retriever, classifier):
        """Verify MISSING_EVIDENCE is emitted when similarity is below threshold, not false NON_COMPLIANT."""
        req = make_req(
            "REQ-MISS-01",
            "Bidder must submit ISO 14001:2015 Environmental Management certification.",
            param="ISO 14001"
        )
        # Completely unrelated sentence
        candidates = [
            make_evd("EVD-UNRELATED", "Our office lunch cafeteria was remodeled in August 2023.")
        ]

        ret_res = semantic_retriever.retrieve_top_k([req], candidates, top_k=1)
        # Semantic similarity for unrelated sentence is low (< 0.25)
        eval_res = classifier.evaluate_clause(req, ret_res[0])

        assert eval_res.label == "MISSING_EVIDENCE"
        assert eval_res.label != "NON_COMPLIANT"
        assert "no statement" in eval_res.reasoning.lower() or "missing" in eval_res.reasoning.lower()

    def test_fallback_to_tfidf_when_model_fails(self):
        """Verify graceful delegation to TF-IDF when semantic model is unavailable or fails."""
        # Create retriever with forced unavailable state
        retriever = SemanticRetriever(
            model_name="all-MiniLM-L6-v2",
            min_similarity_threshold=0.25,
            fallback_to_tfidf=True
        )
        # Simulate model load failure
        retriever.is_available = False
        retriever.init_error = "Mock CPU out of memory error"

        req = make_req("REQ-FALLBACK", "Minimum turnover of 50 Crore INR required.", param="Turnover")
        candidates = [
            make_evd("EVD-FB", "Annual turnover exceeds 60 Crore INR in audited statements.")
        ]

        results = retriever.retrieve_top_k([req], candidates, top_k=1)
        assert len(results) == 1
        top_cand = results[0].candidates[0]

        # Must transparently reflect fallback method
        assert "tfidf_fallback" in top_cand.retrieval_method
        assert "Mock CPU out of memory" in top_cand.retrieval_method
        assert "Retrieved via TF-IDF fallback" in top_cand.review_notes

    def test_offline_or_missing_dependency_behavior(self):
        """Verify explicit RuntimeError is raised when fallback is disabled and model cannot load."""
        retriever = SemanticRetriever(
            model_name="all-MiniLM-L6-v2",
            fallback_to_tfidf=False
        )
        retriever.is_available = False
        retriever.init_error = "sentence_transformers library not installed in offline environment"

        req = make_req("REQ-ERR", "Test requirement")
        candidates = [make_evd("EVD-ERR", "Test evidence")]

        with pytest.raises(RuntimeError) as exc_info:
            retriever.retrieve_top_k([req], candidates, top_k=1)

        assert "SentenceRetriever is unavailable" in str(exc_info.value) or "cannot encode" in str(exc_info.value) or "TF-IDF fallback is disabled" in str(exc_info.value)

    def test_pipeline_e2e_with_semantic_and_tfidf_selection(self, tmp_path):
        """Verify CompliancePipeline executes end-to-end with both engines and records method."""
        db_file = os.path.join(tmp_path, "test_pipeline.db")
        pipeline = CompliancePipeline(db_path=db_file)

        # Use synthetic test PDFs from dataset
        tender_pdf = os.path.join(
            os.path.dirname(__file__), "..", "dataset", "synthetic_pdfs", "tenders", "TENDER-2024-IT-001.pdf"
        )
        bidder_pdf = os.path.join(
            os.path.dirname(__file__), "..", "dataset", "synthetic_pdfs", "bidders", "BIDDER-ALPHA-TECH-PROPOSAL.pdf"
        )

        if not os.path.exists(tender_pdf) or not os.path.exists(bidder_pdf):
            pytest.skip("Synthetic PDFs not generated in dataset/synthetic_pdfs/.")

        # 1. Run with Sentence-BERT Semantic Retrieval
        report_semantic = pipeline.run(tender_pdf, bidder_pdf, retrieval_method="semantic", save_to_db=True)
        assert "semantic" in report_semantic.retrieval_method
        assert report_semantic.total_requirements > 0
        assert report_semantic.evaluations[0].retrieval_method == report_semantic.retrieval_method

        # 2. Run with TF-IDF Keyword Baseline
        report_tfidf = pipeline.run(tender_pdf, bidder_pdf, retrieval_method="tfidf", save_to_db=True)
        assert "tfidf" in report_tfidf.retrieval_method
        assert report_tfidf.total_requirements == report_semantic.total_requirements

        # Verify database recorded both analyses with distinct methods
        db_rows = pipeline.db.list_analyses()
        assert len(db_rows) == 2
        methods_in_db = [r["retrieval_method"] for r in db_rows]
        assert any("semantic" in m for m in methods_in_db)
        assert any("tfidf" in m for m in methods_in_db)
