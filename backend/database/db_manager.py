"""
db_manager.py - SQLite Database Storage for Procurement Compliance Analyses
Part of Phase 4: Baseline Demo Pipeline

Stores demo analyses and their itemized requirement evaluations in a local
SQLite database (compliance_demo.db).
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_FILE_PATH = os.path.join(os.path.dirname(__file__), "compliance_demo.db")


class DatabaseManager:
    """Manages SQLite database connections and CRUD operations for compliance analyses."""

    def __init__(self, db_path: str = DB_FILE_PATH):
        self.db_path = db_path
        self._initialize_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a sqlite3 connection with dict-like row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize_schema(self):
        """Creates the database schema if tables do not exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Analyses Summary Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    tender_filename TEXT NOT NULL,
                    bidder_filename TEXT NOT NULL,
                    bidder_name TEXT NOT NULL,
                    total_requirements INTEGER NOT NULL,
                    compliant_count INTEGER NOT NULL,
                    non_compliant_count INTEGER NOT NULL,
                    uncertain_count INTEGER NOT NULL,
                    missing_evidence_count INTEGER NOT NULL,
                    mandatory_shortfalls_count INTEGER NOT NULL,
                    overall_status TEXT NOT NULL,
                    retrieval_method TEXT DEFAULT 'tfidf',
                    nli_enabled INTEGER DEFAULT 1,
                    nli_model TEXT,
                    annexure_documents TEXT
                )
            """)

            # Safe migration for existing SQLite databases
            try:
                cursor.execute("ALTER TABLE analyses ADD COLUMN annexure_documents TEXT")
            except Exception:
                pass

            # 2. Clause Evaluations Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS clause_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id TEXT NOT NULL,
                    requirement_id TEXT NOT NULL,
                    requirement_text TEXT NOT NULL,
                    category TEXT NOT NULL,
                    parameter TEXT NOT NULL,
                    operator TEXT NOT NULL,
                    required_value REAL,
                    required_unit TEXT,
                    is_mandatory INTEGER,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    top_similarity_score REAL NOT NULL,
                    top_evidence_sentence TEXT,
                    top_evidence_source TEXT,
                    top_evidence_page INTEGER,
                    top_evidence_section TEXT,
                    has_negation INTEGER NOT NULL,
                    retrieval_method TEXT DEFAULT 'tfidf',
                    nli_relation TEXT,
                    nli_entailment_prob REAL,
                    nli_contradiction_prob REAL,
                    nli_neutral_prob REAL,
                    nli_model TEXT,
                    numeric_comparison_json TEXT,
                    reasoning TEXT NOT NULL,
                    candidates_json TEXT,
                    cross_doc_conflict INTEGER DEFAULT 0,
                    cross_doc_conflict_type TEXT,
                    cross_doc_conflict_severity TEXT DEFAULT 'NONE',
                    cross_doc_primary_source TEXT,
                    cross_doc_primary_value REAL,
                    cross_doc_authoritative_source TEXT,
                    cross_doc_authoritative_value REAL,
                    cross_doc_authoritative_page INTEGER,
                    cross_doc_reasoning TEXT,
                    FOREIGN KEY (analysis_id) REFERENCES analyses (id) ON DELETE CASCADE
                )
            """)

            # 3. Graceful migration for existing database files
            try:
                cursor.execute("ALTER TABLE analyses ADD COLUMN retrieval_method TEXT DEFAULT 'tfidf'")
            except sqlite3.OperationalError:
                pass
            try:
                cursor.execute("ALTER TABLE analyses ADD COLUMN nli_enabled INTEGER DEFAULT 1")
            except sqlite3.OperationalError:
                pass
            try:
                cursor.execute("ALTER TABLE analyses ADD COLUMN nli_model TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                cursor.execute("ALTER TABLE clause_evaluations ADD COLUMN retrieval_method TEXT DEFAULT 'tfidf'")
            except sqlite3.OperationalError:
                pass
            for col, ctype in [
                ("nli_relation", "TEXT"),
                ("nli_entailment_prob", "REAL"),
                ("nli_contradiction_prob", "REAL"),
                ("nli_neutral_prob", "REAL"),
                ("nli_model", "TEXT"),
                ("cross_doc_conflict", "INTEGER DEFAULT 0"),
                ("cross_doc_conflict_type", "TEXT"),
                ("cross_doc_conflict_severity", "TEXT DEFAULT 'NONE'"),
                ("cross_doc_primary_source", "TEXT"),
                ("cross_doc_primary_value", "REAL"),
                ("cross_doc_authoritative_source", "TEXT"),
                ("cross_doc_authoritative_value", "REAL"),
                ("cross_doc_authoritative_page", "INTEGER"),
                ("cross_doc_reasoning", "TEXT"),
            ]:
                try:
                    cursor.execute(f"ALTER TABLE clause_evaluations ADD COLUMN {col} {ctype}")
                except sqlite3.OperationalError:
                    pass

            conn.commit()

    def save_analysis(self, report: Any) -> str:
        """Saves a DocumentComplianceReport to SQLite and returns the generated analysis_id."""
        analysis_id = f"ANA-{uuid.uuid4().hex[:8].upper()}"
        now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        def _get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        # Determine overall preliminary posture (strictly informative, not final qualification)
        mand_shortfalls = _get(report, "mandatory_shortfalls_count", 0)
        overall_status = "COMPLIANT_PRELIMINARY" if mand_shortfalls == 0 else f"DISCREPANCIES_FLAGGED ({mand_shortfalls} mandatory shortfalls)"
        retrieval_method = _get(report, "retrieval_method", "tfidf")
        nli_enabled = 1 if _get(report, "nli_enabled", True) else 0
        nli_model = _get(report, "nli_model", None)

        annexure_docs = _get(report, "annexure_documents", []) or []
        annexure_docs_json = json.dumps(list(annexure_docs))

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Insert summary row
            cursor.execute("""
                INSERT INTO analyses (
                    id, created_at, tender_filename, bidder_filename, bidder_name,
                    total_requirements, compliant_count, non_compliant_count,
                    uncertain_count, missing_evidence_count, mandatory_shortfalls_count,
                    overall_status, retrieval_method, nli_enabled, nli_model,
                    annexure_documents
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                analysis_id,
                now_iso,
                getattr(report, "tender_document", "TENDER.pdf"),
                getattr(report, "bidder_document", "BIDDER.pdf"),
                getattr(report, "bidder_name", "Unknown Bidder"),
                getattr(report, "total_requirements", 0),
                getattr(report, "compliant_count", 0),
                getattr(report, "non_compliant_count", 0),
                getattr(report, "uncertain_count", 0),
                getattr(report, "missing_evidence_count", 0),
                mand_shortfalls,
                overall_status,
                retrieval_method,
                nli_enabled,
                nli_model,
                annexure_docs_json
            ))

            # Insert itemized evaluations
            for ev in _get(report, "evaluations", []):
                req_val = _get(ev, "required_value", None)
                cursor.execute("""
                    INSERT INTO clause_evaluations (
                        analysis_id, requirement_id, requirement_text, category, parameter,
                        operator, required_value, required_unit, is_mandatory, label,
                        confidence, top_similarity_score, top_evidence_sentence,
                        top_evidence_source, top_evidence_page, top_evidence_section,
                        has_negation, retrieval_method, nli_relation, nli_entailment_prob,
                        nli_contradiction_prob, nli_neutral_prob, nli_model,
                        numeric_comparison_json, reasoning, candidates_json,
                        cross_doc_conflict, cross_doc_conflict_type, cross_doc_conflict_severity,
                        cross_doc_primary_source, cross_doc_primary_value,
                        cross_doc_authoritative_source, cross_doc_authoritative_value,
                        cross_doc_authoritative_page, cross_doc_reasoning
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    analysis_id,
                    _get(ev, "requirement_id", ""),
                    _get(ev, "requirement_text", ""),
                    _get(ev, "category", ""),
                    _get(ev, "parameter", ""),
                    _get(ev, "operator", ">="),
                    float(req_val) if req_val is not None else None,
                    _get(ev, "required_unit", None),
                    1 if _get(ev, "is_mandatory", False) else 0,
                    _get(ev, "label", "UNCERTAIN"),
                    _get(ev, "confidence", 0.0),
                    _get(ev, "top_similarity_score", 0.0),
                    _get(ev, "top_evidence_sentence", None),
                    _get(ev, "top_evidence_source", None),
                    _get(ev, "top_evidence_page", None),
                    _get(ev, "top_evidence_section", None),
                    1 if _get(ev, "has_negation", False) else 0,
                    _get(ev, "retrieval_method", retrieval_method),
                    _get(ev, "nli_relation", None),
                    _get(ev, "nli_entailment_prob", None),
                    _get(ev, "nli_contradiction_prob", None),
                    _get(ev, "nli_neutral_prob", None),
                    _get(ev, "nli_model", None),
                    json.dumps(_get(ev, "numeric_comparison", None)),
                    _get(ev, "reasoning", ""),
                    json.dumps(_get(ev, "candidates", [])),
                    1 if _get(ev, "cross_doc_conflict", False) else 0,
                    _get(ev, "cross_doc_conflict_type", None),
                    _get(ev, "cross_doc_conflict_severity", "NONE"),
                    _get(ev, "cross_doc_primary_source", None),
                    _get(ev, "cross_doc_primary_value", None),
                    _get(ev, "cross_doc_authoritative_source", None),
                    _get(ev, "cross_doc_authoritative_value", None),
                    _get(ev, "cross_doc_authoritative_page", None),
                    _get(ev, "cross_doc_reasoning", None),
                ))

            conn.commit()

        return analysis_id

    def list_analyses(self) -> List[Dict[str, Any]]:
        """Returns all recorded analyses sorted by creation date descending."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM analyses ORDER BY created_at DESC")
            rows = cursor.fetchall()
            results = []
            for row in rows:
                r = dict(row)
                if r.get("annexure_documents"):
                    try:
                        r["annexure_documents"] = json.loads(r["annexure_documents"])
                    except Exception:
                        r["annexure_documents"] = []
                else:
                    r["annexure_documents"] = []
                results.append(r)
            return results

    def get_analysis_by_id(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves complete analysis with all evaluated requirements and evidence."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,))
            summary_row = cursor.fetchone()
            if not summary_row:
                return None

            result = dict(summary_row)
            if "nli_enabled" in result:
                result["nli_enabled"] = bool(result["nli_enabled"])
            if result.get("annexure_documents"):
                try:
                    result["annexure_documents"] = json.loads(result["annexure_documents"])
                except Exception:
                    result["annexure_documents"] = []
            else:
                result["annexure_documents"] = []

            cursor.execute("""
                SELECT * FROM clause_evaluations
                WHERE analysis_id = ?
                ORDER BY id ASC
            """, (analysis_id,))
            eval_rows = cursor.fetchall()

            evaluations = []
            for er in eval_rows:
                d = dict(er)
                # Parse JSON fields
                d["numeric_comparison"] = json.loads(d["numeric_comparison_json"]) if d.get("numeric_comparison_json") else None
                d["candidates"] = json.loads(d["candidates_json"]) if d.get("candidates_json") else []
                d["is_mandatory"] = bool(d["is_mandatory"])
                d["has_negation"] = bool(d["has_negation"])
                d["cross_doc_conflict"] = bool(d.get("cross_doc_conflict", False))
                evaluations.append(d)

            result["evaluations"] = evaluations
            return result
