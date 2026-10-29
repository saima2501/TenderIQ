"""
app.py - FastAPI REST Service & Interactive Web Dashboard
Part of Phase 4: Baseline Demo Pipeline

Endpoints:
- GET /api/health: Service status & metadata
- GET /api/documents/tenders: Lists available synthetic tender PDFs
- GET /api/documents/bidders: Lists available synthetic bidder proposal PDFs
- POST /api/analyze: Runs end-to-end extraction, retrieval, numeric checks & classification
- GET /api/analyses: Lists historical analyses stored in SQLite
- GET /api/analyses/{analysis_id}: Details for a specific historical analysis
- GET /: Interactive HTML5 web application dashboard for local demo
"""

import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from backend.pipeline import CompliancePipeline, DEFAULT_BIDDER_ANNEXURES
from backend.database.db_manager import DatabaseManager

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TENDERS_DIR = os.path.join(PROJECT_ROOT, "dataset", "synthetic_pdfs", "tenders")
BIDDERS_DIR = os.path.join(PROJECT_ROOT, "dataset", "synthetic_pdfs", "bidders")

app = FastAPI(
    title="AI-Powered GeM Procurement Compliance NLP Pipeline",
    description="Phase 7 Multi-Document Evidence Reconciliation Demo API & Evaluation Dashboard",
    version="1.0.0"
)

# Enable CORS for local React/Vite development or standalone browsers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = CompliancePipeline()
db_manager = DatabaseManager()


class AnalysisRequest(BaseModel):
    tender_filename: str
    bidder_filename: str
    annexure_filenames: Optional[List[str]] = None
    retrieval_method: Optional[str] = "semantic"
    enable_nli: Optional[bool] = True


@app.get("/api/health")
def health():
    """Health check endpoint."""
    nli_clf = getattr(pipeline.classifier, "nli_classifier", None)
    return {
        "status": "healthy",
        "service": "NLP Procurement Compliance Platform",
        "phase": 4,
        "phase_5_semantic": True,
        "phase_6_nli": True,
        "phase_7_cross_doc_reconciliation": True,
        "current_phase": 7,
        "semantic_retriever": {
            "available": pipeline.semantic_retriever.is_available,
            "model_name": pipeline.semantic_retriever.model_name,
            "device": pipeline.semantic_retriever.device,
            "error": pipeline.semantic_retriever.init_error
        },
        "nli_classifier": {
            "available": getattr(nli_clf, "is_available", False) if nli_clf else False,
            "model_name": getattr(nli_clf, "model_name", None),
            "device": getattr(nli_clf, "device", None),
            "enabled": pipeline.classifier.enable_nli,
            "error": getattr(nli_clf, "init_error", None)
        },
        "cross_doc_reconciler": {
            "enabled": getattr(pipeline.classifier, "enable_reconciliation", True),
            "available": getattr(pipeline.classifier, "reconciler", None) is not None,
        },
        "tfidf_available": True,
        "database": "SQLite (compliance_demo.db)",
        "tenders_available": len(os.listdir(TENDERS_DIR)) if os.path.exists(TENDERS_DIR) else 0,
        "bidders_available": len(os.listdir(BIDDERS_DIR)) if os.path.exists(BIDDERS_DIR) else 0
    }


@app.get("/api/documents/tenders")
def get_tenders():
    """Returns list of synthetic tender PDFs available in workspace."""
    if not os.path.exists(TENDERS_DIR):
        return []
    files = [f for f in os.listdir(TENDERS_DIR) if f.lower().endswith(".pdf")]
    return sorted(files)


@app.get("/api/documents/bidders")
def get_bidders():
    """Returns list of actual bidder proposal PDFs (excluding separate annexures)."""
    if not os.path.exists(BIDDERS_DIR):
        return []
    # Filter to main proposal documents only (Blocker 4)
    files = [
        f for f in os.listdir(BIDDERS_DIR)
        if f.lower().endswith(".pdf") and not f.upper().startswith("ANNEXURE")
    ]
    return sorted(files)


@app.get("/api/documents/packages")
def get_packages():
    """Returns proposals, all available annexures, and deterministic package mappings."""
    if not os.path.exists(BIDDERS_DIR):
        return {"proposals": [], "annexures": [], "package_mapping": {}}
    all_files = [f for f in os.listdir(BIDDERS_DIR) if f.lower().endswith(".pdf")]
    proposals = sorted([f for f in all_files if not f.upper().startswith("ANNEXURE")])
    annexures = sorted([f for f in all_files if f.upper().startswith("ANNEXURE")])
    return {
        "proposals": proposals,
        "annexures": annexures,
        "package_mapping": DEFAULT_BIDDER_ANNEXURES
    }


@app.post("/api/analyze")
def run_analysis(request: AnalysisRequest):
    """Executes full end-to-end extraction, matching, and multi-document classification."""
    tender_path = os.path.join(TENDERS_DIR, request.tender_filename)
    bidder_path = os.path.join(BIDDERS_DIR, request.bidder_filename)

    if not os.path.exists(tender_path):
        raise HTTPException(status_code=404, detail=f"Tender PDF not found: {request.tender_filename}")
    if not os.path.exists(bidder_path):
        raise HTTPException(status_code=404, detail=f"Bidder PDF not found: {request.bidder_filename}")

    # Build and validate annexure paths (Blocker 1 & Blocker 4)
    annex_names = request.annexure_filenames
    if annex_names is None:
        annex_names = DEFAULT_BIDDER_ANNEXURES.get(request.bidder_filename, [])

    annexure_paths: List[str] = []
    for a_name in annex_names:
        # Cross-bidder isolation verification: Do not mix documents from different bidders
        for other_bidder, allowed_annexes in DEFAULT_BIDDER_ANNEXURES.items():
            if other_bidder != request.bidder_filename and a_name in allowed_annexes:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cross-bidder isolation error: Annexure '{a_name}' belongs to '{other_bidder}', not '{request.bidder_filename}'. Cannot mix documents across different bidders."
                )
        a_path = os.path.join(BIDDERS_DIR, a_name)
        if not os.path.exists(a_path):
            raise HTTPException(status_code=404, detail=f"Annexure PDF not found: {a_name}")
        annexure_paths.append(a_path)

    try:
        report = pipeline.run(
            tender_pdf_path=tender_path,
            bidder_pdf_path=bidder_path,
            annexure_pdf_paths=annexure_paths,
            retrieval_method=request.retrieval_method,
            enable_nli=request.enable_nli,
            save_to_db=True
        )
        report_dict = report.to_dict()
        report_dict["analysis_id"] = getattr(report, "analysis_id", "N/A")
        return report_dict
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.get("/api/analyses")
def list_analyses():
    """Returns all past analysis records from SQLite."""
    return db_manager.list_analyses()


@app.get("/api/analyses/{analysis_id}")
def get_analysis_details(analysis_id: str):
    """Retrieves full analysis report and clause evaluations from SQLite."""
    res = db_manager.get_analysis_by_id(analysis_id)
    if not res:
        raise HTTPException(status_code=404, detail=f"Analysis not found: {analysis_id}")
    return res



# -------------------------------------------------------------
# Dashboard Statistics Aggregation Endpoint
# -------------------------------------------------------------
@app.get("/api/dashboard-stats")
def dashboard_stats():
    """Aggregates KPI and compliance data from all historical analyses."""
    analyses = db_manager.list_analyses()
    total_analyses = len(analyses)
    unique_bidders = len(set(a.get("bidder_filename", "") for a in analyses))
    total_compliant     = sum(a.get("compliant_count",        0) for a in analyses)
    total_non_compliant = sum(a.get("non_compliant_count",    0) for a in analyses)
    total_uncertain     = sum(a.get("uncertain_count",        0) for a in analyses)
    total_missing       = sum(a.get("missing_evidence_count", 0) for a in analyses)
    total_clauses = total_compliant + total_non_compliant + total_uncertain + total_missing
    need_review   = total_uncertain + total_missing
    cross_doc_count = 0
    review_queue_items = []
    for a in analyses[:10]:
        details = db_manager.get_analysis_by_id(a["id"])
        if details:
            for ev in details.get("evaluations", []):
                if ev.get("cross_doc_conflict"):
                    cross_doc_count += 1
                if ev.get("label") in ("NON_COMPLIANT","UNCERTAIN","MISSING_EVIDENCE") and len(review_queue_items) < 5:
                    review_queue_items.append({
                        "analysis_id":      a["id"],
                        "tender_filename":  a.get("tender_filename",""),
                        "bidder_filename":  a.get("bidder_filename",""),
                        "requirement_id":   ev.get("requirement_id",""),
                        "requirement_text": ev.get("requirement_text","")[:80],
                        "label":            ev.get("label",""),
                        "severity":         ev.get("cross_doc_conflict_severity","Medium") or "Medium",
                        "cross_doc_conflict": ev.get("cross_doc_conflict", False),
                    })
    return {
        "total_analyses":        total_analyses,
        "unique_bidders":        unique_bidders,
        "need_review":           need_review,
        "cross_doc_conflicts":   cross_doc_count,
        "total_clauses":         total_clauses,
        "compliant_count":       total_compliant,
        "non_compliant_count":   total_non_compliant,
        "uncertain_count":       total_uncertain,
        "missing_evidence_count":total_missing,
        "review_queue":          review_queue_items,
        "recent_analyses":       analyses[:8],
    }


# -------------------------------------------------------------
# Embedded Web Dashboard for Instant Local Demo
# -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index_page():
    """Serves the TenderIQ interactive dashboard for the Phase 7 Reconciliation demo."""
    html_file = os.path.join(os.path.dirname(__file__), "dashboard.html")
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<!DOCTYPE html><html><body><h1>TenderIQ</h1></body></html>")
