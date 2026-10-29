"""
test_demo_blockers.py - Verification and regression tests for the four demo blockers:
1. Multi-document bidder package ingestion in pipeline and API
2. Contextual financial figure extraction (_parse_usd_millions)
3. Phase 7 dashboard visibility and conflict reporting
4. Bidder proposal and annexure package selector isolation
"""

import os
import pytest
from fastapi.testclient import TestClient

from backend.pipeline import (
    ComplianceVerificationPipeline,
    DEFAULT_BIDDER_ANNEXURES,
)
from backend.reconciliation.cross_document_reconciler import (
    _parse_usd_millions,
    CrossDocumentReconciler,
)
from backend.api.app import app
from backend.database.db_manager import DatabaseManager


# ---------------------------------------------------------------------------
# BLOCKER 2: FINANCIAL FIGURE EXTRACTION REGRESSION TESTS
# ---------------------------------------------------------------------------

class TestFinancialFigureExtraction:
    """Regression tests for _parse_usd_millions in cross_document_reconciler.py."""

    def test_audited_vs_disputed_parenthetical_claim(self):
        """USD 6.2M audited figure must be extracted instead of USD 14.5M disputed note."""
        text = (
            "Audited balance sheet: Registered annual turnover is USD 6.2 Million. "
            "(Note: Narrative claims of USD 14.5 Million are currently subject to ongoing audit review)."
        )
        val = _parse_usd_millions(text)
        assert val == 6.2, f"Expected 6.2, got {val}"

    def test_disputed_narrative_claim_in_parentheses_ignored(self):
        """Parenthetical claims with disputed / unverified narrative must be excluded."""
        text = (
            "Audited turnover: USD 8.0 Million (earlier marketing claims of USD 25.0 Million were unverified)."
        )
        val = _parse_usd_millions(text)
        assert val == 8.0, f"Expected 8.0, got {val}"

    def test_ordinary_valid_turnover_statements(self):
        """Standard valid turnover statements should parse cleanly."""
        assert _parse_usd_millions("Our annual audited turnover is USD 18.5 Million.") == 18.5
        assert _parse_usd_millions("Audited revenue of $22.4M for FY2023.") == 22.4
        assert _parse_usd_millions("The certified accounts show USD 6.2M turnover.") == 6.2
        assert _parse_usd_millions("Annual turnover of USD 32 Million as certified by auditors.") == 32.0

    def test_unrelated_numbers_ignored(self):
        """Numbers for headcount, uptime, SLA or calendar years must not be parsed as turnover."""
        assert _parse_usd_millions("We have 150 full-time engineers.") is None
        assert _parse_usd_millions("We guarantee 99.95% system availability as of 2023.") is None
        assert _parse_usd_millions("The project team delivered 4 enterprise deployments in 2022.") is None

    def test_multiple_conflicting_audited_figures_returns_uncertain(self):
        """Multiple conflicting audited figures must return None (conservative review signal)."""
        text = (
            "Audited financial statements report USD 6.0 Million turnover, whereas the certified "
            "bank statement indicates USD 12.0 Million turnover."
        )
        val = _parse_usd_millions(text)
        assert val is None, f"Expected None (uncertain review signal), got {val}"

    def test_empty_and_narrative_without_figures(self):
        """Statements with no figures return None."""
        assert _parse_usd_millions("") is None
        assert _parse_usd_millions("Our organization maintains healthy financial liquidity.") is None


# ---------------------------------------------------------------------------
# BLOCKER 1 & 4: MULTI-DOCUMENT PIPELINE AND BIDDER ISOLATION
# ---------------------------------------------------------------------------

class TestMultiDocumentPipeline:
    """Integration tests for multi-document package ingestion in ComplianceVerificationPipeline."""

    @pytest.fixture
    def pdf_paths(self):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        tenders_dir = os.path.join(project_root, "dataset", "synthetic_pdfs", "tenders")
        bidders_dir = os.path.join(project_root, "dataset", "synthetic_pdfs", "bidders")
        tender = os.path.join(tenders_dir, "TENDER-2024-IT-001.pdf")
        alpha = os.path.join(bidders_dir, "BIDDER-ALPHA-TECH-PROPOSAL.pdf")
        beta = os.path.join(bidders_dir, "BIDDER-BETA-HEALTH-PROPOSAL.pdf")
        annexure = os.path.join(bidders_dir, "ANNEXURE-FINANCIAL-STATEMENTS.pdf")
        return {
            "tender": tender,
            "alpha": alpha,
            "beta": beta,
            "annexure": annexure,
        }

    def test_multi_document_package_analysis_detects_financial_conflict(self, pdf_paths):
        """Proposal + Annexure analysis detects $14.5M proposal claim vs $6.2M audited annexure."""
        pipeline = ComplianceVerificationPipeline()
        report = pipeline.run(
            tender_pdf_path=pdf_paths["tender"],
            bidder_pdf_path=pdf_paths["alpha"],
            annexure_pdf_paths=[pdf_paths["annexure"]],
            retrieval_method="tfidf",
            enable_nli=False,
            enable_reconciliation=True,
        )

        assert "Alpha" in report.bidder_name, f"Unexpected bidder {report.bidder_name}"
        assert "ANNEXURE-FINANCIAL-STATEMENTS.pdf" in report.annexure_documents

        # Find the turnover requirement evaluation (parameter: 'audited financial turnover')
        req_turnover = next((e for e in report.evaluations if "turnover" in (e.parameter or "").lower()), None)
        assert req_turnover is not None, "Audited turnover requirement not found in report evaluations"

        # Audited figure is $6.2M < $10.0M threshold -> Must be NON_COMPLIANT with cross_doc_conflict=True
        assert req_turnover.cross_doc_conflict is True, "Expected cross_doc_conflict to be True"
        assert req_turnover.cross_doc_primary_value == 14.5
        assert req_turnover.cross_doc_authoritative_value == 6.2
        assert req_turnover.cross_doc_authoritative_source == "ANNEXURE-FINANCIAL-STATEMENTS.pdf"
        assert req_turnover.label == "NON_COMPLIANT"

        # Evidence candidates must retain source document provenance
        sources = {
            (c.get("source_document") if isinstance(c, dict) else getattr(c, "source_document", None))
            for c in req_turnover.candidates
        }
        sources.discard(None)
        assert len(sources) >= 1, "Candidates must record source_document provenance"

    def test_single_pdf_backward_compatibility(self, pdf_paths):
        """Passing annexure_pdf_paths=[] runs in single-document mode without cross-document conflicts."""
        pipeline = ComplianceVerificationPipeline()
        report = pipeline.run(
            tender_pdf_path=pdf_paths["tender"],
            bidder_pdf_path=pdf_paths["alpha"],
            annexure_pdf_paths=[],  # Explicitly empty: single-PDF mode
            retrieval_method="tfidf",
            enable_nli=False,
            enable_reconciliation=True,
        )

        assert report.annexure_documents == []
        req_turnover = next((e for e in report.evaluations if "turnover" in (e.parameter or "").lower()), None)
        assert req_turnover is not None
        # In single document mode without annexure, proposal claims $14.5M which meets >= $10M threshold
        assert req_turnover.cross_doc_conflict is False
        assert req_turnover.label == "COMPLIANT"

    def test_wrong_bidder_isolation_raises_error(self, pdf_paths):
        """Attempting to attach Alpha's annexure to Beta's proposal must raise ValueError."""
        pipeline = ComplianceVerificationPipeline()
        with pytest.raises(ValueError, match="Cross-bidder isolation error"):
            pipeline.run(
                tender_pdf_path=pdf_paths["tender"],
                bidder_pdf_path=pdf_paths["beta"],
                annexure_pdf_paths=[pdf_paths["annexure"]],  # Alpha's annexure attached to Beta
                retrieval_method="tfidf",
            )

    def test_auto_association_default_mapping(self, pdf_paths):
        """When annexure_pdf_paths is None, default package mapping automatically associates Alpha's annexure."""
        pipeline = ComplianceVerificationPipeline()
        report = pipeline.run(
            tender_pdf_path=pdf_paths["tender"],
            bidder_pdf_path=pdf_paths["alpha"],
            annexure_pdf_paths=None,  # Auto-associate
            retrieval_method="tfidf",
            enable_nli=False,
            enable_reconciliation=True,
        )
        assert "ANNEXURE-FINANCIAL-STATEMENTS.pdf" in report.annexure_documents
        req_turnover = next((e for e in report.evaluations if "turnover" in (e.parameter or "").lower()), None)
        assert req_turnover is not None
        assert req_turnover.cross_doc_conflict is True


# ---------------------------------------------------------------------------
# BLOCKER 3 & 4: API ENDPOINTS AND DASHBOARD VISIBILITY
# ---------------------------------------------------------------------------

class TestApiAndDashboard:
    """Integration tests for FastAPI endpoints, package selection, and dashboard UI."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_bidders_endpoint_excludes_annexures(self, client):
        """GET /api/documents/bidders must return only proposals, not annexures."""
        res = client.get("/api/documents/bidders")
        assert res.status_code == 200
        bidders = res.json()
        assert isinstance(bidders, list)
        for b in bidders:
            assert not b.startswith("ANNEXURE-"), f"Annexure found in bidders list: {b}"
            assert b.startswith("BIDDER-"), f"Non-bidder found in bidders list: {b}"

    def test_packages_endpoint_structure(self, client):
        """GET /api/documents/packages returns proposals, annexures, and mapping."""
        res = client.get("/api/documents/packages")
        assert res.status_code == 200
        data = res.json()
        assert "proposals" in data
        assert "annexures" in data
        assert "package_mapping" in data
        assert "BIDDER-ALPHA-TECH-PROPOSAL.pdf" in data["package_mapping"]
        assert "ANNEXURE-FINANCIAL-STATEMENTS.pdf" in data["package_mapping"]["BIDDER-ALPHA-TECH-PROPOSAL.pdf"]

    def test_api_analyze_with_package_returns_conflict(self, client):
        """POST /api/analyze with bidder proposal and annexure returns cross_doc_conflict."""
        payload = {
            "tender_filename": "TENDER-2024-IT-001.pdf",
            "bidder_filename": "BIDDER-ALPHA-TECH-PROPOSAL.pdf",
            "annexure_filenames": ["ANNEXURE-FINANCIAL-STATEMENTS.pdf"],
            "retrieval_method": "tfidf",
            "enable_nli": False,
        }
        res = client.post("/api/analyze", json=payload)
        assert res.status_code == 200
        data = res.json()

        assert "ANNEXURE-FINANCIAL-STATEMENTS.pdf" in data["annexure_documents"]
        req_turnover = next((e for e in data["evaluations"] if "turnover" in (e.get("parameter") or "").lower()), None)
        assert req_turnover is not None, "Turnover evaluation not returned in API response"
        assert req_turnover["cross_doc_conflict"] is True
        assert req_turnover["cross_doc_authoritative_value"] == 6.2
        assert req_turnover["label"] == "NON_COMPLIANT"

    def test_api_analyze_wrong_bidder_isolation_returns_400(self, client):
        """POST /api/analyze with mismatched annexure returns HTTP 400."""
        payload = {
            "tender_filename": "TENDER-2024-IT-001.pdf",
            "bidder_filename": "BIDDER-BETA-HEALTH-PROPOSAL.pdf",
            "annexure_filenames": ["ANNEXURE-FINANCIAL-STATEMENTS.pdf"],
            "retrieval_method": "tfidf",
            "enable_nli": False,
        }
        res = client.post("/api/analyze", json=payload)
        assert res.status_code == 400
        assert "Cross-bidder isolation error" in res.json()["detail"]

    def test_dashboard_html_contains_phase7_and_conflict_elements(self, client):
        """GET / serves HTML containing Phase 7 metadata, conflict badges, and package controls."""
        res = client.get("/")
        assert res.status_code == 200
        html = res.text

        # Header and Phase 7 metadata
        assert "Phase 7 Multi-Document Platform" in html
        assert "Phase 7: Multi-Document Reconciliation" in html

        # UI elements for package selection and conflicts
        assert "Package Annexures" in html
        assert "CROSS-DOCUMENT CONFLICT" in html
        assert "onBidderSelectionChanged" in html
        assert "val-conflict" in html


# ---------------------------------------------------------------------------
# DATABASE PERSISTENCE TESTS
# ---------------------------------------------------------------------------

class TestDatabasePersistence:
    """Verify that annexure_documents and cross-document conflict details persist in SQLite."""

    def test_db_persistence_roundtrip(self, tmp_path):
        db_file = str(tmp_path / "test_compliance.db")
        db = DatabaseManager(db_path=db_file)

        # Mock an evaluation with cross-doc conflict and annexures
        eval_dict = {
            "requirement_id": "REQ-001",
            "category": "FINANCIAL",
            "requirement_text": "USD 10 Million turnover",
            "is_mandatory": True,
            "parameter": "annual_turnover_usd",
            "operator": ">=",
            "required_value": 10.0,
            "required_unit": "USD_MILLION",
            "label": "NON_COMPLIANT",
            "confidence_score": 0.95,
            "top_evidence_sentence": "Registered annual turnover is USD 6.2 Million.",
            "top_evidence_source": "ANNEXURE-FINANCIAL-STATEMENTS.pdf",
            "top_evidence_page": 1,
            "top_similarity_score": 0.88,
            "has_negation": False,
            "numeric_comparison": {"operator": ">=", "required": 10.0, "extracted": 6.2, "status": "FAIL"},
            "reasoning": "Cross-document discrepancy detected.",
            "candidates": [],
            "retrieval_method": "tfidf",
            "nli_relation": "contradiction",
            "cross_doc_conflict": True,
            "cross_doc_conflict_type": "numeric_magnitude",
            "cross_doc_conflict_severity": "HARD",
            "cross_doc_primary_source": "BIDDER-ALPHA-TECH-PROPOSAL.pdf",
            "cross_doc_primary_value": 14.5,
            "cross_doc_authoritative_source": "ANNEXURE-FINANCIAL-STATEMENTS.pdf",
            "cross_doc_authoritative_value": 6.2,
            "cross_doc_authoritative_page": 1,
            "cross_doc_reasoning": "Proposal claims $14.5M vs audited $6.2M.",
        }

        report_dict = {
            "tender_document": "TENDER-IT-INFRA-2024.pdf",
            "bidder_document": "BIDDER-ALPHA-TECH-PROPOSAL.pdf",
            "bidder_name": "ALPHA TECH",
            "total_requirements": 1,
            "compliant_count": 0,
            "non_compliant_count": 1,
            "uncertain_count": 0,
            "missing_evidence_count": 0,
            "mandatory_shortfalls_count": 1,
            "retrieval_method": "tfidf",
            "nli_enabled": False,
            "annexure_documents": ["ANNEXURE-FINANCIAL-STATEMENTS.pdf"],
            "evaluations": [eval_dict],
            "disclaimer": "Academic demo notice",
        }

        analysis_id = db.save_analysis(report_dict)
        assert analysis_id is not None

        saved = db.get_analysis_by_id(analysis_id)
        assert saved is not None
        assert saved["annexure_documents"] == ["ANNEXURE-FINANCIAL-STATEMENTS.pdf"]
        assert len(saved["evaluations"]) == 1
        ev = saved["evaluations"][0]
        assert ev["cross_doc_conflict"] is True
        assert ev["cross_doc_authoritative_value"] == 6.2
        assert ev["cross_doc_primary_value"] == 14.5
