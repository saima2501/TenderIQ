"""
test_startup_seeding.py - Unit & Integration Tests for Idempotent Startup Demo Seeding
Verifies:
1. Startup seeding executes when database has zero records.
2. Startup seeding does not produce duplicate records on repeated execution.
3. Existing database records are never overwritten or modified.
4. Seeding failures are handled gracefully without corrupting database or raising unhandled errors.
5. FastAPI health check endpoint returns 200 OK with expected schema.
"""

import os
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from backend.api.app import app, seed_initial_demo_if_empty
from backend.database.db_manager import DatabaseManager
from backend.pipeline import CompliancePipeline


@pytest.fixture
def empty_db(tmp_path):
    """Provides an isolated, empty SQLite DatabaseManager instance."""
    db_file = str(tmp_path / "empty_test.db")
    return DatabaseManager(db_path=db_file)


def test_1_seeding_on_empty_database(empty_db):
    """1. Seeding executes successfully when database contains zero analyses."""
    assert len(empty_db.list_analyses()) == 0

    # Execute seeding with real pipeline (TF-IDF mode)
    seeded = seed_initial_demo_if_empty(database=empty_db)
    assert seeded is True

    analyses = empty_db.list_analyses()
    assert len(analyses) == 1

    first = analyses[0]
    assert first["tender_filename"] == "TENDER-2024-IT-001.pdf"
    assert first["bidder_filename"] == "BIDDER-ALPHA-TECH-PROPOSAL.pdf"
    assert first["total_requirements"] > 0

    # Verify evaluations were persisted in SQLite
    details = empty_db.get_analysis_by_id(first["id"])
    assert details is not None
    assert len(details.get("evaluations", [])) > 0


def test_2_no_duplicate_seeding_on_repeated_startup(empty_db):
    """2. Calling seeding a second time does not duplicate analyses or evaluations."""
    # First seeding
    seeded_first = seed_initial_demo_if_empty(database=empty_db)
    assert seeded_first is True
    initial_count = len(empty_db.list_analyses())
    assert initial_count == 1

    # Second execution (simulating service restart or second worker)
    seeded_second = seed_initial_demo_if_empty(database=empty_db)
    assert seeded_second is False

    # Count must remain exactly 1
    assert len(empty_db.list_analyses()) == initial_count


def test_3_existing_records_not_overwritten(empty_db):
    """3. If database already contains a record, seeding skips immediately without modifying it."""
    # Manually insert a mock analysis
    mock_report = {
        "tender_filename": "EXISTING-TENDER.pdf",
        "bidder_filename": "EXISTING-BIDDER.pdf",
        "bidder_name": "Pre-Existing Vendor",
        "overall_status": "COMPLIANT",
        "total_requirements": 1,
        "compliant_count": 1,
        "non_compliant_count": 0,
        "uncertain_count": 0,
        "missing_evidence_count": 0,
        "mandatory_shortfalls_count": 0,
        "retrieval_method": "tfidf",
        "nli_enabled": False,
        "nli_model": None,
        "annexure_documents": [],
        "evaluations": [],
    }
    custom_id = empty_db.save_analysis(mock_report)
    assert len(empty_db.list_analyses()) == 1

    # Attempt seeding
    seeded = seed_initial_demo_if_empty(database=empty_db)
    assert seeded is False

    # Verify only original record exists and is untouched
    records = empty_db.list_analyses()
    assert len(records) == 1
    assert records[0]["id"] == custom_id
    assert records[0]["bidder_name"] == "Pre-Existing Vendor"


def test_4_seeding_failure_handling(empty_db):
    """4. If pipeline raises an error during seeding, it is caught gracefully without crashing."""
    mock_pipeline = MagicMock(spec=CompliancePipeline)
    mock_pipeline.run.side_effect = RuntimeError("Simulated pipeline extraction failure")

    seeded = seed_initial_demo_if_empty(database=empty_db, nlp_pipeline=mock_pipeline)
    assert seeded is False
    assert len(empty_db.list_analyses()) == 0


def test_5_api_health_endpoint():
    """5. API health endpoint returns 200 OK and valid health schema."""
    client = TestClient(app)
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "tenders_available" in data
    assert "bidders_available" in data


def test_6_memory_safety_and_disabled_model_configuration():
    """6. CompliancePipeline with models disabled does not initialize SBERT or NLI models."""
    safe_pipe = CompliancePipeline(
        enable_semantic=False,
        enable_nli=False,
        default_retrieval_method="tfidf",
    )
    assert safe_pipe.semantic_retriever is None
    assert safe_pipe.classifier.nli_classifier is None
    assert safe_pipe.retriever == safe_pipe.tfidf_retriever

    # Verify TF-IDF retrieval still works end-to-end
    demo_tender = os.path.join("dataset", "synthetic_pdfs", "tenders", "TENDER-2024-IT-001.pdf")
    demo_bidder = os.path.join("dataset", "synthetic_pdfs", "bidders", "BIDDER-ALPHA-TECH-PROPOSAL.pdf")
    demo_annexure = [os.path.join("dataset", "synthetic_pdfs", "bidders", "ANNEXURE-FINANCIAL-STATEMENTS.pdf")]
    report = safe_pipe.run(
        tender_pdf_path=demo_tender,
        bidder_pdf_path=demo_bidder,
        annexure_pdf_paths=demo_annexure,
        retrieval_method="tfidf",
        enable_nli=False,
        save_to_db=False,
    )
    assert report is not None
    assert len(report.evaluations) > 0
    assert report.retrieval_method.lower().startswith("tfidf")
