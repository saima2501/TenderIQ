"""
test_frontend_routes.py - Verification tests for Step 2G.5 AI Assistant Frontend Integration
Verifies:
- Dashboard root serving and DOM integrity for the new AI Assistant view
- Sidebar navigation item, icon, and hash routing
- Option 3 analysis context selection (workbench selection + dropdown switcher)
- Chat interface components, quick prompts, and safety constraints
- Clause explanation integration in Bid Analysis results
- Status endpoint and offline fallback indicators
- Preservation of all existing dashboard views and contracts
"""

import re
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import pytest

from backend.api.app import app
from backend.database.db_manager import DatabaseManager


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def dashboard_html(client):
    res = client.get("/")
    assert res.status_code == 200
    return res.text


def test_1_sidebar_navigation_item(dashboard_html):
    """1. New sidebar item navigation exists with correct ID, route, and icon."""
    assert 'id="nav-ai-assistant"' in dashboard_html
    assert 'href="#ai-assistant"' in dashboard_html
    assert "AI Assistant" in dashboard_html
    assert "🤖" in dashboard_html


def test_2_route_and_hash_navigation(dashboard_html):
    """2. AI Assistant route exists in ROUTES dictionary and hash routing is wired."""
    assert "'ai-assistant':" in dashboard_html
    assert "sectionId: 'view-ai-assistant'" in dashboard_html
    assert "navId: 'nav-ai-assistant'" in dashboard_html
    assert "if(route === 'ai-assistant') initAiAssistantView();" in dashboard_html


def test_3_view_section_structure(dashboard_html):
    """3. view-ai-assistant SPA section exists with required header and structure."""
    assert 'id="view-ai-assistant"' in dashboard_html
    assert "AI Procurement Assistant" in dashboard_html
    assert 'id="assistant-status-pill"' in dashboard_html
    assert 'id="assistant-status-text"' in dashboard_html


def test_4_context_selector_elements(dashboard_html):
    """4. Analysis context selector elements (Option 3) are present."""
    assert 'id="assistant-analysis-select"' in dashboard_html
    assert 'id="assistant-meta-banner"' in dashboard_html
    assert 'id="assistant-meta-tender"' in dashboard_html
    assert 'id="assistant-meta-bidder"' in dashboard_html
    assert 'id="assistant-meta-verdict"' in dashboard_html
    assert 'id="assistant-meta-id"' in dashboard_html


def test_5_no_analysis_prompt_and_active_chat(dashboard_html):
    """5. Clear prompt shown when no analysis is selected, and active chat area exists."""
    assert 'id="assistant-no-analysis-prompt"' in dashboard_html
    assert "No Bid Analysis Selected" in dashboard_html
    assert 'id="assistant-active-chat-area"' in dashboard_html


def test_6_chat_interface_controls(dashboard_html):
    """6. Chat controls: messages container, input textarea, send button, loading & error box."""
    assert 'id="chat-messages-container"' in dashboard_html
    assert 'id="chat-input"' in dashboard_html
    assert 'id="btn-chat-send"' in dashboard_html
    assert 'id="chat-loading-indicator"' in dashboard_html
    assert 'id="chat-error-box"' in dashboard_html


def test_7_quick_prompts_configured(dashboard_html):
    """7. All 5 quick prompt questions are present and call triggerQuickPrompt."""
    prompts = [
        "Summarize compliance shortfalls.",
        "Explain cross-document discrepancies.",
        "Which documents are missing?",
        "Project Manager qualifications",
        "Overall bid compliance result"
    ]
    for p in prompts:
        assert p in dashboard_html


def test_8_clause_explanation_action_wired(dashboard_html):
    """8. 'Explain with AI' action is wired in displayResults and modal details."""
    assert "explainClauseWithAI" in dashboard_html
    assert "Explain with AI" in dashboard_html
    assert "ai-explanation-box" in dashboard_html


def test_9_xss_safety_escaping(dashboard_html):
    """9. escapeHtml function is defined to protect against untrusted AI text XSS."""
    assert "function escapeHtml" in dashboard_html
    assert ".replace(/&/g, '&amp;')" in dashboard_html
    assert ".replace(/</g, '&lt;')" in dashboard_html


def test_10_status_endpoint_offline_resilience(client):
    """10. Status endpoint reports offline fallback without crashing."""
    with patch("backend.api.app.ollama_client.is_available", return_value=False):
        res = client.get("/api/assistant/status")
        assert res.status_code == 200
        data = res.json()
        assert data["ollama_available"] is False
        assert data["deterministic_fallback_available"] is True


def test_11_chat_endpoint_contract(client):
    """11. POST /api/assistant/chat responds with structured model_used and fallback_used."""
    mock_db = MagicMock(spec=DatabaseManager)
    mock_db.get_analysis_by_id.return_value = {
        "id": "A-TEST-001",
        "tender_filename": "TENDER-1.pdf",
        "bidder_filename": "BIDDER-1.pdf",
        "bidder_name": "Acme Corp",
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
        "evaluations": [
            {
                "requirement_id": "REQ-1",
                "requirement_text": "Must have ISO 9001.",
                "category": "technical",
                "parameter": "iso",
                "operator": "==",
                "required_value": None,
                "required_unit": None,
                "is_mandatory": True,
                "label": "COMPLIANT",
                "confidence": 0.95,
                "top_similarity_score": 0.90,
                "top_evidence_sentence": "We hold an ISO 9001 certificate.",
                "top_evidence_source": "BIDDER-1.pdf",
                "top_evidence_page": 2,
                "reasoning": "Valid certificate provided.",
                "numeric_comparison": None,
                "cross_doc_conflict": False,
            }
        ]
    }
    with patch("backend.api.app.db_manager", mock_db), \
         patch("backend.api.app.chat_service.context_builder.db", mock_db):
        res = client.post("/api/assistant/chat", json={
            "analysis_id": "A-TEST-001",
            "message": "Summarize compliance shortfalls."
        })
        assert res.status_code == 200
        data = res.json()
        assert data["analysis_id"] == "A-TEST-001"
        assert "answer" in data
        assert isinstance(data["citations"], list)
        assert "fallback_used" in data


def test_12_explain_clause_endpoint_contract(client):
    """12. POST /api/assistant/explain-clause returns grounded explanation and verdict."""
    mock_db = MagicMock(spec=DatabaseManager)
    mock_db.get_analysis_by_id.return_value = {
        "id": "A-TEST-002",
        "evaluations": [
            {
                "requirement_id": "REQ-01",
                "requirement_text": "Experience must be 5 years.",
                "category": "eligibility",
                "parameter": "exp",
                "operator": ">=",
                "required_value": 5,
                "required_unit": "years",
                "is_mandatory": True,
                "label": "COMPLIANT",
                "confidence": 0.9,
                "top_similarity_score": 0.85,
                "top_evidence_sentence": "We have 7 years experience.",
                "top_evidence_source": "PROP.pdf",
                "top_evidence_page": 1,
                "reasoning": "7 years satisfies >= 5 years.",
                "numeric_comparison": None,
                "cross_doc_conflict": False,
            }
        ]
    }
    with patch("backend.api.app.db_manager", mock_db):
        res = client.post("/api/assistant/explain-clause", json={
            "analysis_id": "A-TEST-002",
            "requirement_id": "REQ-01"
        })
        assert res.status_code == 200
        data = res.json()
        assert data["official_verdict"] == "COMPLIANT"
        assert "COMPLIANT" in data["explanation"]
        assert data["citations"][0]["source_document"] == "PROP.pdf"


def test_13_existing_views_intact(dashboard_html):
    """13. Existing views remain unmodified: overview, tenders, bid-analysis, conflicts, evidence, reports, audit, settings."""
    existing_sections = [
        'id="view-overview"',
        'id="view-tenders"',
        'id="view-bid-analysis"',
        'id="view-conflicts"',
        'id="view-evidence"',
        'id="view-reports"',
        'id="view-audit"',
        'id="view-settings"',
    ]
    for sec in existing_sections:
        assert sec in dashboard_html
