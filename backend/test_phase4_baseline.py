"""
test_phase4_baseline.py - Automated Unit & Integration Test Suite for Phase 4 Baseline Demo
Part of Phase 4: Baseline Demo Pipeline

Covers:
1. TF-IDF Top-3 Candidate Evidence Retrieval
2. Deterministic Numeric Comparison & Unit Conversions (Gbps/Mbps, years/months, percentages, operators)
3. Transparent Baseline Classification (COMPLIANT, NON_COMPLIANT, UNCERTAIN, MISSING_EVIDENCE)
4. SQLite Database Persistence & CRUD Operations
5. End-to-End Compliance Pipeline (PDF -> Extract -> Retrieve -> Compare -> Classify -> DB)
6. FastAPI REST Endpoints & Web Dashboard Delivery
"""

import os
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient

from backend.matching.tfidf_retriever import (
    TFIDFRetriever,
    RetrievedEvidenceCandidate,
    RequirementRetrievalResult
)
from backend.scoring.numeric_comparator import NumericComparator, NumericComparisonResult
from backend.classification.baseline_classifier import BaselineClassifier, ClauseComplianceEvaluation, DocumentComplianceReport
from backend.database.db_manager import DatabaseManager
from backend.pipeline import CompliancePipeline
from backend.api.app import app
from backend.extraction_nlp.requirement_extractor import ExtractedRequirement
from backend.extraction_nlp.evidence_extractor import ExtractedEvidence


class TestPhase4BaselinePipeline(unittest.TestCase):
    """Automated unit and integration test suite for Phase 4."""

    @classmethod
    def setUpClass(cls):
        cls.retriever = TFIDFRetriever()
        cls.comparator = NumericComparator()
        cls.classifier = BaselineClassifier()
        cls.tenders_dir = os.path.join(PROJECT_ROOT, "dataset", "synthetic_pdfs", "tenders")
        cls.bidders_dir = os.path.join(PROJECT_ROOT, "dataset", "synthetic_pdfs", "bidders")
        cls.client = TestClient(app)

    # -------------------------------------------------------------
    # 1. Test TF-IDF Candidate Evidence Retrieval
    # -------------------------------------------------------------
    def test_tfidf_retriever(self):
        """Verify TF-IDF retrieval tokenization, ranking, and top-3 candidate selection."""
        requirements = [
            ExtractedRequirement(
                requirement_id="REQ-001",
                original_text="The hosting environment must guarantee an uptime SLA of 99.95%.",
                category="Technical Specifications",
                parameter="monthly uptime SLA",
                operator=">=",
                value=99.95,
                unit="%",
                is_mandatory=True
            ),
            ExtractedRequirement(
                requirement_id="REQ-002",
                original_text="The contractor must supply a 36 months warranty.",
                category="Statutory Accreditations & Warranties",
                parameter="comprehensive warranty duration",
                operator=">=",
                value=36.0,
                unit="months",
                is_mandatory=True
            )
        ]

        evidence_records = [
            ExtractedEvidence(
                evidence_id="EVD-001",
                original_sentence="Our platform contractually commits to a minimum monthly uptime SLA of 99.99%.",
                numeric_values=[{"raw": "99.99%", "value": 99.99, "unit": "%"}],
                page_number=1
            ),
            ExtractedEvidence(
                evidence_id="EVD-002",
                original_sentence="We provide an all-inclusive 3 years warranty on all installed equipment.",
                numeric_values=[{"raw": "3 years", "value": 3.0, "unit": "years"}],
                page_number=1
            ),
            ExtractedEvidence(
                evidence_id="EVD-003",
                original_sentence="The company was incorporated in 2016 with headquarters in the capital city.",
                numeric_values=[],
                page_number=1
            ),
            ExtractedEvidence(
                evidence_id="EVD-004",
                original_sentence="Network throughput supports 10 Gbps peak load capacity.",
                numeric_values=[{"raw": "10 Gbps", "value": 10.0, "unit": "Gbps"}],
                page_number=2
            )
        ]

        results = self.retriever.retrieve_top_k(requirements, evidence_records, top_k=3)
        self.assertEqual(len(results), 2)

        # Requirement 1 (Uptime SLA) should rank EVD-001 at Rank 1
        res1 = results[0]
        self.assertEqual(res1.requirement_id, "REQ-001")
        self.assertEqual(len(res1.candidates), 3)
        self.assertEqual(res1.candidates[0].evidence_id, "EVD-001")
        self.assertGreater(res1.candidates[0].similarity_score, 0.20)

        # Requirement 2 (Warranty) should rank EVD-002 at Rank 1
        res2 = results[1]
        self.assertEqual(res2.requirement_id, "REQ-002")
        self.assertEqual(res2.candidates[0].evidence_id, "EVD-002")
        self.assertGreater(res2.candidates[0].similarity_score, 0.15)

    # -------------------------------------------------------------
    # 2. Test Numeric Comparator with Unit Conversions & Operators
    # -------------------------------------------------------------
    def test_numeric_comparator(self):
        """Verify deterministic mathematical comparison across unit conversions and operators."""
        # A. At least 36 months (offered: 3 years) -> Unit conversion SATISFIED
        res1 = self.comparator.compare(
            required_val=36.0, required_unit="months", operator=">=",
            offered_val=3.0, offered_unit="years"
        )
        self.assertEqual(res1.status, "SATISFIED")
        self.assertEqual(res1.required_converted, 36.0)
        self.assertEqual(res1.offered_converted, 36.0)

        # B. 1 Gbps vs 1,000 Mbps -> Unit conversion SATISFIED
        res2 = self.comparator.compare(
            required_val=1.0, required_unit="Gbps", operator=">=",
            offered_val=1000.0, offered_unit="Mbps"
        )
        self.assertEqual(res2.status, "SATISFIED")

        # C. Must not exceed 5% (offered: 4.5% -> SATISFIED; offered: 6.0% -> VIOLATED)
        res3_pass = self.comparator.compare(
            required_val=5.0, required_unit="%", operator="<=",
            offered_val=4.5, offered_unit="%"
        )
        self.assertEqual(res3_pass.status, "SATISFIED")

        res3_fail = self.comparator.compare(
            required_val=5.0, required_unit="%", operator="<=",
            offered_val=6.0, offered_unit="%"
        )
        self.assertEqual(res3_fail.status, "VIOLATED")

        # D. Exactly 10 units (offered: 10 units -> SATISFIED; offered: 9 units -> VIOLATED)
        res4_pass = self.comparator.compare(
            required_val=10.0, required_unit="units", operator="==",
            offered_val=10.0, offered_unit="units"
        )
        self.assertEqual(res4_pass.status, "SATISFIED")

        res4_fail = self.comparator.compare(
            required_val=10.0, required_unit="units", operator="==",
            offered_val=9.0, offered_unit="units"
        )
        self.assertEqual(res4_fail.status, "VIOLATED")

        # E. Numerical boundary case: USD 9.95M vs >= USD 10.0M -> VIOLATED
        res5 = self.comparator.compare(
            required_val=10.0, required_unit="USD Million", operator=">=",
            offered_val=9.95, offered_unit="USD Million"
        )
        self.assertEqual(res5.status, "VIOLATED")

        # F. Incompatible units (Duration vs Bandwidth) -> INCOMPATIBLE_UNITS
        res6 = self.comparator.compare(
            required_val=36.0, required_unit="months", operator=">=",
            offered_val=10.0, offered_unit="Gbps"
        )
        self.assertEqual(res6.status, "INCOMPATIBLE_UNITS")

    # -------------------------------------------------------------
    # 3. Test Transparent Baseline Classifier
    # -------------------------------------------------------------
    def test_baseline_classifier(self):
        """Verify compliance label assignment without equating similarity to compliance."""
        # A. COMPLIANT on satisfied threshold
        req_uptime = ExtractedRequirement(
            requirement_id="REQ-01",
            original_text="Uptime must be at least 99.95%.",
            category="Technical",
            parameter="uptime SLA",
            operator=">=",
            value=99.95,
            unit="%"
        )
        cand_compliant = RetrievedEvidenceCandidate(
            rank=1,
            evidence_id="EVD-01",
            sentence="We commit to 99.99% monthly uptime availability.",
            similarity_score=0.85,
            source_document="PROPOSAL.pdf",
            page_number=1,
            numeric_values=[{"raw": "99.99%", "value": 99.99, "unit": "%"}],
            has_negation=False
        )
        ret1 = RequirementRetrievalResult(
            requirement_id="REQ-01",
            requirement_text=req_uptime.original_text,
            parameter=req_uptime.parameter,
            candidates=[cand_compliant]
        )
        eval1 = self.classifier.evaluate_clause(req_uptime, ret1)
        self.assertEqual(eval1.label, "COMPLIANT")

        # B. NON_COMPLIANT on numeric shortfall (offered 99.80% vs required 99.95%)
        cand_shortfall = RetrievedEvidenceCandidate(
            rank=1,
            evidence_id="EVD-02",
            sentence="Our hosting SLA is designed for 99.80% availability.",
            similarity_score=0.82,
            source_document="PROPOSAL.pdf",
            page_number=1,
            numeric_values=[{"raw": "99.80%", "value": 99.80, "unit": "%"}],
            has_negation=False
        )
        ret2 = RequirementRetrievalResult(
            requirement_id="REQ-01",
            requirement_text=req_uptime.original_text,
            parameter=req_uptime.parameter,
            candidates=[cand_shortfall]
        )
        eval2 = self.classifier.evaluate_clause(req_uptime, ret2)
        self.assertEqual(eval2.label, "NON_COMPLIANT")
        self.assertIn("fails threshold", eval2.reasoning.lower())

        # C. NON_COMPLIANT on explicit negation (does not guarantee)
        cand_negation = RetrievedEvidenceCandidate(
            rank=1,
            evidence_id="EVD-03",
            sentence="The production environment does not guarantee a minimum monthly uptime SLA of 99.95%.",
            similarity_score=0.91,
            source_document="PROPOSAL.pdf",
            page_number=1,
            numeric_values=[{"raw": "99.95%", "value": 99.95, "unit": "%"}],
            has_negation=True
        )
        ret3 = RequirementRetrievalResult(
            requirement_id="REQ-01",
            requirement_text=req_uptime.original_text,
            parameter=req_uptime.parameter,
            candidates=[cand_negation]
        )
        eval3 = self.classifier.evaluate_clause(req_uptime, ret3)
        self.assertEqual(eval3.label, "NON_COMPLIANT")
        self.assertTrue(eval3.has_negation)

        # D. UNCERTAIN on promotional claim
        cand_vague = RetrievedEvidenceCandidate(
            rank=1,
            evidence_id="EVD-04",
            sentence="Our team has decades of collective domain knowledge in high uptime architectures.",
            similarity_score=0.45,
            source_document="PROPOSAL.pdf",
            page_number=1,
            numeric_values=[],
            uncertainty_flag=True
        )
        ret4 = RequirementRetrievalResult(
            requirement_id="REQ-01",
            requirement_text=req_uptime.original_text,
            parameter=req_uptime.parameter,
            candidates=[cand_vague]
        )
        eval4 = self.classifier.evaluate_clause(req_uptime, ret4)
        self.assertEqual(eval4.label, "UNCERTAIN")

        # E. MISSING_EVIDENCE on low similarity
        cand_low = RetrievedEvidenceCandidate(
            rank=1,
            evidence_id="EVD-05",
            sentence="General office location is in the north technology corridor.",
            similarity_score=0.03,
            source_document="PROPOSAL.pdf",
            page_number=1
        )
        ret5 = RequirementRetrievalResult(
            requirement_id="REQ-01",
            requirement_text=req_uptime.original_text,
            parameter=req_uptime.parameter,
            candidates=[cand_low]
        )
        eval5 = self.classifier.evaluate_clause(req_uptime, ret5)
        self.assertEqual(eval5.label, "MISSING_EVIDENCE")

    # -------------------------------------------------------------
    # 4. Test SQLite Database Manager Integration
    # -------------------------------------------------------------
    def test_database_sqlite_integration(self):
        """Verify SQLite saving, listing, and retrieving complete evaluations."""
        import uuid
        db_path = os.path.join(tempfile.gettempdir(), f"test_db_{uuid.uuid4().hex}.db")

        try:
            db = DatabaseManager(db_path=db_path)

            mock_eval = ClauseComplianceEvaluation(
                requirement_id="REQ-TEST-01",
                requirement_text="Turnover must be at least USD 10.0 Million.",
                category="Financial Capability",
                parameter="audited financial turnover",
                operator=">=",
                required_value=10.0,
                required_unit="USD Million",
                is_mandatory=True,
                label="COMPLIANT",
                confidence=0.92,
                top_similarity_score=0.88,
                top_evidence_sentence="Audited turnover registered at USD 14.5 Million.",
                top_evidence_source="BIDDER.pdf",
                top_evidence_page=1,
                has_negation=False,
                reasoning="Offered USD 14.5M satisfies threshold >= USD 10.0M."
            )

            mock_report = DocumentComplianceReport(
                tender_document="TENDER-TEST.pdf",
                bidder_document="BIDDER-TEST.pdf",
                bidder_name="Test Systems Ltd.",
                total_requirements=1,
                compliant_count=1,
                non_compliant_count=0,
                uncertain_count=0,
                missing_evidence_count=0,
                mandatory_shortfalls_count=0,
                evaluations=[mock_eval]
            )

            analysis_id = db.save_analysis(mock_report)
            self.assertTrue(analysis_id.startswith("ANA-"))

            # Test list
            analyses = db.list_analyses()
            self.assertEqual(len(analyses), 1)
            self.assertEqual(analyses[0]["id"], analysis_id)
            self.assertEqual(analyses[0]["bidder_name"], "Test Systems Ltd.")

            # Test get by ID
            full_record = db.get_analysis_by_id(analysis_id)
            self.assertIsNotNone(full_record)
            self.assertEqual(len(full_record["evaluations"]), 1)
            self.assertEqual(full_record["evaluations"][0]["requirement_id"], "REQ-TEST-01")
            self.assertEqual(full_record["evaluations"][0]["label"], "COMPLIANT")
            self.assertEqual(full_record["evaluations"][0]["required_value"], 10.0)

        finally:
            try:
                if os.path.exists(db_path):
                    os.remove(db_path)
            except Exception:
                pass

    # -------------------------------------------------------------
    # 5. Test End-to-End Compliance Pipeline
    # -------------------------------------------------------------
    def test_compliance_pipeline_e2e(self):
        """Verify full end-to-end execution on synthetic tender and bidder documents."""
        import uuid
        tender_file = os.path.join(self.tenders_dir, "TENDER-2024-IT-001.pdf")
        bidder_file = os.path.join(self.bidders_dir, "BIDDER-ALPHA-TECH-PROPOSAL.pdf")

        db_path = os.path.join(tempfile.gettempdir(), f"test_e2e_{uuid.uuid4().hex}.db")

        try:
            test_pipeline = CompliancePipeline(db_path=db_path)
            report = test_pipeline.run(tender_file, bidder_file, save_to_db=True)

            self.assertIsInstance(report, DocumentComplianceReport)
            self.assertEqual(report.tender_document, "TENDER-2024-IT-001.pdf")
            self.assertEqual(report.bidder_document, "BIDDER-ALPHA-TECH-PROPOSAL.pdf")
            self.assertGreater(report.total_requirements, 5)
            self.assertGreater(report.compliant_count, 0)

            # Check that an analysis_id was assigned and saved to database
            analysis_id = getattr(report, "analysis_id", None)
            self.assertIsNotNone(analysis_id)

            db = DatabaseManager(db_path=db_path)
            saved = db.get_analysis_by_id(analysis_id)
            self.assertIsNotNone(saved)
            self.assertEqual(saved["total_requirements"], report.total_requirements)
        finally:
            try:
                if os.path.exists(db_path):
                    os.remove(db_path)
            except Exception:
                pass

    # -------------------------------------------------------------
    # 6. Test FastAPI Endpoints & UI
    # -------------------------------------------------------------
    def test_fastapi_endpoints(self):
        """Verify FastAPI REST endpoints and HTML web dashboard."""
        # A. Health endpoint
        res_health = self.client.get("/api/health")
        self.assertEqual(res_health.status_code, 200)
        data_health = res_health.json()
        self.assertEqual(data_health["status"], "healthy")
        self.assertEqual(data_health["phase"], 4)

        # B. Document list endpoints
        res_tenders = self.client.get("/api/documents/tenders")
        self.assertEqual(res_tenders.status_code, 200)
        tenders = res_tenders.json()
        self.assertIn("TENDER-2024-IT-001.pdf", tenders)

        res_bidders = self.client.get("/api/documents/bidders")
        self.assertEqual(res_bidders.status_code, 200)
        bidders = res_bidders.json()
        self.assertIn("BIDDER-ALPHA-TECH-PROPOSAL.pdf", bidders)

        # C. Analyze endpoint
        res_analyze = self.client.post("/api/analyze", json={
            "tender_filename": "TENDER-2024-IT-001.pdf",
            "bidder_filename": "BIDDER-ALPHA-TECH-PROPOSAL.pdf"
        })
        self.assertEqual(res_analyze.status_code, 200)
        analysis_data = res_analyze.json()
        self.assertIn("analysis_id", analysis_data)
        self.assertGreater(analysis_data["total_requirements"], 0)
        self.assertIn("evaluations", analysis_data)

        # D. Analyses history endpoint
        res_history = self.client.get("/api/analyses")
        self.assertEqual(res_history.status_code, 200)
        history = res_history.json()
        self.assertGreater(len(history), 0)

        # E. Analysis detail endpoint
        analysis_id = analysis_data["analysis_id"]
        res_detail = self.client.get(f"/api/analyses/{analysis_id}")
        self.assertEqual(res_detail.status_code, 200)
        detail = res_detail.json()
        self.assertEqual(detail["id"], analysis_id)

        # F. Web Dashboard UI
        res_ui = self.client.get("/")
        self.assertEqual(res_ui.status_code, 200)
        self.assertIn("text/html", res_ui.headers["content-type"])
        self.assertIn("NLP Procurement Tender & Bidder Compliance Analysis", res_ui.text)


if __name__ == "__main__":
    unittest.main()
