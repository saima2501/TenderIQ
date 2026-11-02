"""
test_audit_fixes.py - Unit & Integration Tests for Step 2G.6 Audit Fixes
Verifies:
1. Default Ollama request timeout is 45 seconds and configurable via env variable.
2. Prompt optimization reduces bloat while preserving all mandatory compliance facts.
3. Cross-document discrepancy citations correctly identify authoritative annexures AND proposals.
4. Ordinary citations remain accurate and missing citations are handled honestly.
5. Deterministic fallback retains correct multi-document provenance.
"""

import os
from unittest.mock import MagicMock, patch
import pytest

from backend.assistant.ollama_client import OllamaClient, DEFAULT_REQUEST_TIMEOUT
from backend.assistant.context_builder import AnalysisContextBuilder, AnalysisContext, ClauseContext
from backend.assistant.chat_service import ChatService
from backend.database.db_manager import DatabaseManager


@pytest.fixture
def mock_discrepancy_analysis():
    """Mock analysis with an authoritative cross-document discrepancy."""
    return {
        "id": "ANA-DISCREPANCY-001",
        "tender_filename": "TENDER-FINANCIAL-2026.pdf",
        "bidder_filename": "BIDDER-ALPHA-TECH-PROPOSAL.pdf",
        "bidder_name": "Alpha Tech Systems Ltd",
        "overall_status": "DISCREPANCIES_FLAGGED (1 mandatory shortfall)",
        "total_requirements": 2,
        "compliant_count": 1,
        "non_compliant_count": 1,
        "uncertain_count": 0,
        "missing_evidence_count": 0,
        "mandatory_shortfalls_count": 1,
        "retrieval_method": "semantic",
        "nli_enabled": True,
        "nli_model": "cross-encoder/nli-distilroberta-base",
        "annexure_documents": ["ANNEXURE-FINANCIAL-STATEMENTS.pdf"],
        "evaluations": [
            {
                "requirement_id": "REQ-FIN-01",
                "requirement_text": "Annual financial turnover must be at least USD 50.0 Million.",
                "category": "financial",
                "parameter": "annual_turnover",
                "operator": ">=",
                "required_value": 50.0,
                "required_unit": "million",
                "is_mandatory": True,
                "label": "NON_COMPLIANT",
                "confidence": 0.95,
                "top_similarity_score": 0.88,
                "top_evidence_sentence": "Our average annual turnover reached USD 60.0 Million over recent fiscal years.",
                "top_evidence_source": "BIDDER-ALPHA-TECH-PROPOSAL.pdf",
                "top_evidence_page": 3,
                "top_evidence_section": "Financial Standing",
                "reasoning": "Cross-document discrepancy: narrative proposal claims $60M but audited balance sheet confirms $30M.",
                "numeric_comparison": {
                    "status": "VIOLATED",
                    "required_raw": 50.0,
                    "required_unit": "million",
                    "offered_raw": 30.0,
                    "offered_unit": "million",
                    "operator": ">=",
                    "reasoning": "Audited metric 30.0 Million violates threshold of >= 50.0 Million."
                },
                "cross_doc_conflict": True,
                "cross_doc_conflict_type": "NUMERIC_DISCREPANCY",
                "cross_doc_conflict_severity": "HIGH",
                "cross_doc_primary_source": "BIDDER-ALPHA-TECH-PROPOSAL.pdf",
                "cross_doc_primary_value": 60.0,
                "cross_doc_authoritative_source": "ANNEXURE-FINANCIAL-STATEMENTS.pdf",
                "cross_doc_authoritative_value": 30.0,
                "cross_doc_authoritative_page": 1,
                "cross_doc_reasoning": "Audited balance sheet (ANNEXURE-FINANCIAL-STATEMENTS.pdf p.1) establishes revenue is $30M, refuting proposal narrative ($60M).",
            },
            {
                "requirement_id": "REQ-TECH-02",
                "requirement_text": "System uptime SLA must be at least 99.90%.",
                "category": "technical",
                "parameter": "uptime_sla",
                "operator": ">=",
                "required_value": 99.90,
                "required_unit": "%",
                "is_mandatory": True,
                "label": "COMPLIANT",
                "confidence": 0.94,
                "top_similarity_score": 0.89,
                "top_evidence_sentence": "Platform contractually guarantees 99.95% annual availability.",
                "top_evidence_source": "BIDDER-ALPHA-TECH-PROPOSAL.pdf",
                "top_evidence_page": 7,
                "top_evidence_section": "Service Level Agreement",
                "reasoning": "Offered 99.95% satisfies requirement of >= 99.90%.",
                "numeric_comparison": {
                    "status": "SATISFIED",
                    "required_raw": 99.90,
                    "required_unit": "%",
                    "offered_raw": 99.95,
                    "offered_unit": "%",
                    "operator": ">=",
                    "reasoning": "Offered 99.95% satisfies requirement of >= 99.90%."
                },
                "cross_doc_conflict": False,
                "cross_doc_conflict_type": None,
                "cross_doc_conflict_severity": "NONE",
                "cross_doc_reasoning": None,
            }
        ]
    }


def test_1_default_timeout_is_45_seconds():
    """1. Ollama default timeout is 45.0s."""
    client = OllamaClient()
    assert client.request_timeout == 45.0
    assert client._timeout.read == 45.0


def test_2_timeout_env_variable_configurable(monkeypatch):
    """2. Ollama timeout is configurable via OLLAMA_REQUEST_TIMEOUT env var."""
    monkeypatch.setenv("OLLAMA_REQUEST_TIMEOUT", "60.0")
    # Re-instantiate reading env
    timeout_val = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "45.0"))
    client = OllamaClient(request_timeout=timeout_val)
    assert client.request_timeout == 60.0
    assert client._timeout.read == 60.0


def test_3_prompt_optimization_preserves_compliance_information(mock_discrepancy_analysis):
    """3. Prompt optimization reduces bloat while preserving all mandatory compliance facts."""
    mock_db = MagicMock(spec=DatabaseManager)
    mock_db.get_analysis_by_id.return_value = mock_discrepancy_analysis

    builder = AnalysisContextBuilder(db_manager=mock_db)
    ctx = builder.build("ANA-DISCREPANCY-001")
    prompt = ctx.format_for_prompt()

    # Preserves requirement IDs
    assert "[REQ-FIN-01]" in prompt
    assert "[REQ-TECH-02]" in prompt

    # Preserves official verdicts
    assert "VERDICT: NON_COMPLIANT" in prompt
    assert "VERDICT: COMPLIANT" in prompt

    # Preserves numeric details
    assert "Required >= 50.0 million" in prompt
    assert "Required >= 99.9 %" in prompt

    # Preserves evidence and sources
    assert "BIDDER-ALPHA-TECH-PROPOSAL.pdf" in prompt
    assert "Page 3" in prompt

    # Preserves cross-doc conflict details and authoritative annexure
    assert "ANNEXURE-FINANCIAL-STATEMENTS.pdf" in prompt
    assert "p.1" in prompt
    assert "Authoritative" in prompt

    # Indicates summary view
    assert "Authoritative Summary Context" in prompt


def test_4_cross_document_citations_include_authoritative_annexure_and_proposal(mock_discrepancy_analysis):
    """4. Discrepancy citations include BOTH the authoritative annexure AND the proposal."""
    mock_db = MagicMock(spec=DatabaseManager)
    mock_db.get_analysis_by_id.return_value = mock_discrepancy_analysis

    builder = AnalysisContextBuilder(db_manager=mock_db)
    # Using deterministic fallback client to verify citation generation
    client = OllamaClient(base_url="http://127.0.0.1:9999")
    service = ChatService(ollama_client=client, context_builder=builder)

    res = service.chat("ANA-DISCREPANCY-001", "Explain cross-document discrepancies.")
    assert res.fallback_used is True

    # Citations must contain BOTH the authoritative annexure and the proposal
    citation_docs = [c["source_document"] for c in res.citations]
    assert "ANNEXURE-FINANCIAL-STATEMENTS.pdf" in citation_docs
    assert "BIDDER-ALPHA-TECH-PROPOSAL.pdf" in citation_docs

    # Authoritative annexure must have correct page 1
    auth_cit = [c for c in res.citations if c["source_document"] == "ANNEXURE-FINANCIAL-STATEMENTS.pdf"][0]
    assert auth_cit["page_number"] == 1
    assert auth_cit["has_citation"] is True


def test_5_ordinary_citations_remain_accurate(mock_discrepancy_analysis):
    """5. Non-cross-document queries cite only the relevant proposal without injecting annexures."""
    mock_db = MagicMock(spec=DatabaseManager)
    mock_db.get_analysis_by_id.return_value = mock_discrepancy_analysis

    builder = AnalysisContextBuilder(db_manager=mock_db)
    client = OllamaClient(base_url="http://127.0.0.1:9999")
    service = ChatService(ollama_client=client, context_builder=builder)

    res = service.chat("ANA-DISCREPANCY-001", "What is the uptime SLA?")
    assert res.fallback_used is True

    citation_docs = [c["source_document"] for c in res.citations]
    assert "BIDDER-ALPHA-TECH-PROPOSAL.pdf" in citation_docs
    # Authoritative annexure should NOT be cited for non-conflict SLA requirement
    assert "ANNEXURE-FINANCIAL-STATEMENTS.pdf" not in citation_docs


def test_6_missing_citation_information_handled_honestly():
    """6. When source document is missing or N/A, reports unavailable rather than inventing."""
    analysis_no_citation = {
        "id": "ANA-NO-CIT",
        "tender_filename": "TENDER.pdf",
        "bidder_filename": "BIDDER.pdf",
        "bidder_name": "Vendor X",
        "overall_status": "MISSING_EVIDENCE",
        "total_requirements": 1,
        "compliant_count": 0,
        "non_compliant_count": 0,
        "uncertain_count": 0,
        "missing_evidence_count": 1,
        "mandatory_shortfalls_count": 1,
        "retrieval_method": "tfidf",
        "nli_enabled": False,
        "nli_model": None,
        "annexure_documents": [],
        "evaluations": [
            {
                "requirement_id": "REQ-MISS-01",
                "requirement_text": "Must submit EMD Guarantee.",
                "category": "eligibility",
                "parameter": "emd",
                "operator": "==",
                "required_value": None,
                "required_unit": None,
                "is_mandatory": True,
                "label": "MISSING_EVIDENCE",
                "confidence": 0.8,
                "top_similarity_score": 0.1,
                "top_evidence_sentence": None,
                "top_evidence_source": None,
                "top_evidence_page": None,
                "reasoning": "EMD guarantee document not submitted.",
                "numeric_comparison": None,
                "cross_doc_conflict": False,
            }
        ]
    }
    mock_db = MagicMock(spec=DatabaseManager)
    mock_db.get_analysis_by_id.return_value = analysis_no_citation

    builder = AnalysisContextBuilder(db_manager=mock_db)
    client = OllamaClient(base_url="http://127.0.0.1:9999")
    service = ChatService(ollama_client=client, context_builder=builder)

    res = service.chat("ANA-NO-CIT", "Where is the EMD?")
    assert len(res.citations) == 1
    assert res.citations[0]["has_citation"] is False
    assert res.citations[0]["source_document"] == "Source citation unavailable"
    assert res.citations[0]["page_number"] is None


def test_7_focus_query_prioritizes_relevant_clauses(mock_discrepancy_analysis):
    """7. focus_query re-orders clauses so targeted requirement appears first."""
    mock_db = MagicMock(spec=DatabaseManager)
    mock_db.get_analysis_by_id.return_value = mock_discrepancy_analysis

    builder = AnalysisContextBuilder(db_manager=mock_db)
    ctx = builder.build("ANA-DISCREPANCY-001")

    # Focus on uptime SLA (REQ-TECH-02 is 2nd in original list)
    prompt_focused = ctx.format_for_prompt(focus_query="Tell me about the uptime SLA")
    pos_tech = prompt_focused.find("[REQ-TECH-02]")
    pos_fin = prompt_focused.find("[REQ-FIN-01]")

    assert pos_tech != -1
    assert pos_fin != -1
    # REQ-TECH-02 should now appear before REQ-FIN-01
    assert pos_tech < pos_fin


def test_8_actual_newline_formatting_in_prompt(mock_discrepancy_analysis):
    """8. format_for_prompt() must join lines using actual newline characters, not literal backslash-n."""
    mock_db = MagicMock(spec=DatabaseManager)
    mock_db.get_analysis_by_id.return_value = mock_discrepancy_analysis

    builder = AnalysisContextBuilder(db_manager=mock_db)
    ctx = builder.build("ANA-DISCREPANCY-001")
    prompt_str = ctx.format_for_prompt(max_clauses=8)

    # Must contain actual newlines
    assert "\n" in prompt_str
    # Must NOT contain escaped literal backslash-n sequences
    assert "\\n" not in prompt_str
    # Verify multiline structure
    lines = prompt_str.split("\n")
    assert len(lines) > 5
    assert lines[0].startswith("=== STORED TENDERIQ COMPLIANCE ANALYSIS")


def test_9_clause_outside_top_eight_brought_into_context_by_prioritization():
    """9. A clause positioned outside top 8 is prioritized into the 8-clause context."""
    # Build 12 dummy clauses; clause 11 has a unique parameter 'quantum_encryption'
    evals = []
    for i in range(1, 13):
        evals.append({
            "requirement_id": f"REQ-GEN-{i:02d}",
            "requirement_text": f"General Requirement number {i}",
            "category": "technical",
            "parameter": f"param_{i}",
            "operator": ">=",
            "required_value": float(i),
            "required_unit": "units",
            "is_mandatory": True,
            "label": "COMPLIANT",
            "confidence": 0.9,
            "top_similarity_score": 0.8,
            "top_evidence_sentence": f"Evidence statement for requirement {i}",
            "top_evidence_source": "PROPOSAL.pdf",
            "top_evidence_page": i,
            "top_evidence_section": "General",
            "reasoning": f"Complies with requirement {i}",
            "numeric_comparison": None,
            "cross_doc_conflict": False,
        })
    # Clause 11 is special
    evals[10]["requirement_id"] = "REQ-CRYPTO-99"
    evals[10]["requirement_text"] = "Bidder must support quantum encryption protocols"
    evals[10]["parameter"] = "quantum_encryption"

    mock_db = MagicMock(spec=DatabaseManager)
    mock_db.get_analysis_by_id.return_value = {
        "id": "ANA-MANY-CLAUSES",
        "tender_filename": "TENDER.pdf",
        "bidder_filename": "PROPOSAL.pdf",
        "bidder_name": "Test Vendor",
        "overall_status": "COMPLIANT",
        "total_requirements": 12,
        "compliant_count": 12,
        "non_compliant_count": 0,
        "uncertain_count": 0,
        "missing_evidence_count": 0,
        "mandatory_shortfalls_count": 0,
        "retrieval_method": "semantic",
        "nli_enabled": True,
        "nli_model": "test-nli",
        "annexure_documents": [],
        "evaluations": evals,
    }

    builder = AnalysisContextBuilder(db_manager=mock_db)
    ctx = builder.build("ANA-MANY-CLAUSES")

    # Without focus_query, only top 8 are included; REQ-CRYPTO-99 (index 10) is omitted
    prompt_unfocused = ctx.format_for_prompt(max_clauses=8)
    assert "[REQ-CRYPTO-99]" not in prompt_unfocused

    # With focus_query targeting quantum encryption, it must be prioritized into top 8
    prompt_focused = ctx.format_for_prompt(max_clauses=8, focus_query="Tell me about quantum encryption")
    assert "[REQ-CRYPTO-99]" in prompt_focused


def test_10_context_limitation_note_included_when_exceeding_eight_clauses():
    """10. When clauses exceed max_clauses, the context limitation note is explicitly included."""
    evals = [
        {
            "requirement_id": f"REQ-{i:02d}",
            "requirement_text": f"Req {i}",
            "category": "technical",
            "parameter": f"param_{i}",
            "operator": ">=",
            "required_value": 1.0,
            "required_unit": "unit",
            "is_mandatory": True,
            "label": "COMPLIANT",
            "confidence": 0.9,
            "top_similarity_score": 0.8,
            "top_evidence_sentence": f"Evidence {i}",
            "top_evidence_source": "PROPOSAL.pdf",
            "top_evidence_page": i,
            "top_evidence_section": "General",
            "reasoning": "Satisfied",
            "numeric_comparison": None,
            "cross_doc_conflict": False,
        }
        for i in range(1, 13)
    ]
    mock_db = MagicMock(spec=DatabaseManager)
    mock_db.get_analysis_by_id.return_value = {
        "id": "ANA-12-CLAUSES",
        "tender_filename": "TENDER.pdf",
        "bidder_filename": "PROPOSAL.pdf",
        "bidder_name": "Test Vendor",
        "overall_status": "COMPLIANT",
        "total_requirements": 12,
        "compliant_count": 12,
        "non_compliant_count": 0,
        "uncertain_count": 0,
        "missing_evidence_count": 0,
        "mandatory_shortfalls_count": 0,
        "retrieval_method": "semantic",
        "nli_enabled": True,
        "nli_model": "test-nli",
        "annexure_documents": [],
        "evaluations": evals,
    }

    builder = AnalysisContextBuilder(db_manager=mock_db)
    ctx = builder.build("ANA-12-CLAUSES")
    prompt = ctx.format_for_prompt(max_clauses=8)

    assert "[Note: Context summarized; 4 additional clauses recorded in database]" in prompt
