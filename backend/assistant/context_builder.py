"""
context_builder.py - Grounded Analysis Context Builder
Part of Step 2G.4: AI Chatbot Backend Implementation

Retrieves stored bid compliance analysis results from the database for a specific
analysis_id and formats them into structured, isolated context for AI assistance.
Guarantees strict analysis and bidder isolation.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import logging

from backend.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class ClauseContext:
    """Detailed evaluation context for a single tender requirement."""
    requirement_id: str
    requirement_text: str
    category: str
    parameter: str
    operator: str
    required_value: Optional[float]
    required_unit: Optional[str]
    is_mandatory: bool
    official_verdict: str                  # Immutable label from database (COMPLIANT, NON_COMPLIANT, etc.)
    confidence: float
    top_similarity_score: float
    top_evidence_sentence: Optional[str]
    source_document: Optional[str]
    page_number: Optional[int]
    section: Optional[str]
    reasoning: str
    numeric_comparison: Optional[Dict[str, Any]]
    cross_doc_conflict: bool
    cross_doc_conflict_type: Optional[str]
    cross_doc_conflict_severity: Optional[str]
    cross_doc_reasoning: Optional[str]
    cross_doc_primary_source: Optional[str] = None
    cross_doc_authoritative_source: Optional[str] = None
    cross_doc_authoritative_page: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisContext:
    """Complete, isolated compliance context for an entire bid analysis."""
    analysis_id: str
    tender_filename: str
    bidder_filename: str
    bidder_name: str
    overall_status: str
    total_requirements: int
    compliant_count: int
    non_compliant_count: int
    uncertain_count: int
    missing_evidence_count: int
    mandatory_shortfalls_count: int
    retrieval_method: str
    nli_enabled: bool
    nli_model: Optional[str]
    annexure_documents: List[str]
    clauses: List[ClauseContext] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_clause(self, requirement_id: str) -> Optional[ClauseContext]:
        """Finds clause by requirement_id (case-insensitive)."""
        target = requirement_id.strip().upper()
        for c in self.clauses:
            if c.requirement_id.strip().upper() == target:
                return c
        return None

    def format_summary(self) -> str:
        """Generates a concise textual summary of analysis status."""
        return (
            f"Analysis ID: {self.analysis_id}\n"
            f"Tender Document: {self.tender_filename}\n"
            f"Bidder: {self.bidder_name} (File: {self.bidder_filename})\n"
            f"Official Overall Status: {self.overall_status}\n"
            f"Requirements Breakdown: Total {self.total_requirements} | "
            f"Compliant: {self.compliant_count} | Non-Compliant: {self.non_compliant_count} | "
            f"Uncertain: {self.uncertain_count} | Missing Evidence: {self.missing_evidence_count} | "
            f"Mandatory Shortfalls: {self.mandatory_shortfalls_count}"
        )

    def format_for_prompt(self, max_clauses: int = 40, focus_query: Optional[str] = None) -> str:
        """Formats the analysis context into dense, optimized prompt text.

        Preserves all official verdicts, numeric checks, discrepancies, evidence quotes,
        and citations while eliminating redundant whitespace and formatting bloat.
        """
        lines = [
            "=== STORED TENDERIQ COMPLIANCE ANALYSIS (Authoritative Summary Context) ===",
            self.format_summary(),
            "",
            "--- ITEM-BY-ITEM EVALUATION RECORDS (Concise Audit View) ---",
        ]

        # Prioritize clauses matching focus_query if provided
        clauses_to_format = list(self.clauses)
        if focus_query:
            fq_upper = focus_query.upper()

            def _is_relevant(c: ClauseContext) -> bool:
                if c.requirement_id.upper() in fq_upper:
                    return True
                if c.parameter:
                    p_up = c.parameter.upper()
                    if p_up in fq_upper or p_up.replace("_", " ") in fq_upper:
                        return True
                    # Check individual parameter words of length >= 3
                    words = [w for w in p_up.split("_") if len(w) >= 3]
                    if words and all(w in fq_upper for w in words):
                        return True
                if c.cross_doc_conflict and ("CONFLICT" in fq_upper or "DISCREPANC" in fq_upper):
                    return True
                if c.official_verdict == "MISSING_EVIDENCE" and "MISSING" in fq_upper:
                    return True
                # Check requirement text phrase overlap
                req_words = [w.strip(".,;:()") for w in c.requirement_text.upper().split() if len(w) >= 4]
                matching_words = [w for w in req_words if w in fq_upper and w not in {"MUST", "BE", "AT", "LEAST", "WITH", "THAT", "SHALL", "HAVE"}]
                if len(matching_words) >= 2:
                    return True
                return False

            relevant = [c for c in clauses_to_format if _is_relevant(c)]
            others = [c for c in clauses_to_format if c not in relevant]
            clauses_to_format = (relevant + others)[:max_clauses]
        else:
            clauses_to_format = clauses_to_format[:max_clauses]

        for c in clauses_to_format:
            mand_str = "MANDATORY" if c.is_mandatory else "OPTIONAL"
            lines.append(f"[{c.requirement_id}] ({c.parameter or 'general'}, {mand_str}) -> VERDICT: {c.official_verdict}")
            lines.append(f"  Requirement: \"{c.requirement_text}\"")
            if c.reasoning:
                lines.append(f"  Audit Rationale: {c.reasoning}")

            if c.top_evidence_sentence:
                citation_doc = c.source_document or "Source unavailable"
                citation_page = f", Page {c.page_number}" if c.page_number is not None else ""
                lines.append(f"  Evidence: \"{c.top_evidence_sentence}\" [Source: {citation_doc}{citation_page}]")
            else:
                lines.append("  Evidence: [NO EVIDENCE SUBMITTED OR LOCATED]")

            if c.numeric_comparison and isinstance(c.numeric_comparison, dict):
                st = c.numeric_comparison.get("status", "")
                rq = c.numeric_comparison.get("required_raw")
                of = c.numeric_comparison.get("offered_raw")
                un = c.numeric_comparison.get("required_unit") or ""
                op = c.numeric_comparison.get("operator") or ""
                lines.append(f"  Numeric Check: Required {op} {rq} {un}, Offered {of} {un} -> {st}")

            if c.cross_doc_conflict:
                auth_doc = c.cross_doc_authoritative_source or "Audited Annexure"
                auth_p = f" p.{c.cross_doc_authoritative_page}" if c.cross_doc_authoritative_page else ""
                prim_doc = c.cross_doc_primary_source or c.source_document or "Proposal"
                rsn = c.cross_doc_reasoning or c.reasoning
                lines.append(f"  Cross-Doc Conflict: Primary ({prim_doc}) vs Authoritative ({auth_doc}{auth_p}) -> Discrepancy: {rsn}")

            lines.append("")

        if len(self.clauses) > max_clauses:
            lines.append(f"[Note: Context summarized; {len(self.clauses) - max_clauses} additional clauses recorded in database]")

        return "\n".join(lines)


class AnalysisContextBuilder:
    """Retrieves and builds isolated compliance context from the database."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()

    def build(self, analysis_id: str) -> Optional[AnalysisContext]:
        """Loads analysis by ID and returns AnalysisContext, or None if not found.

        Enforces strict isolation: queries only the rows corresponding to analysis_id.
        """
        if not analysis_id or not isinstance(analysis_id, str):
            logger.warning("Invalid analysis_id provided: %r", analysis_id)
            return None

        clean_id = analysis_id.strip()
        data = self.db.get_analysis_by_id(clean_id)
        if not data:
            logger.info("Analysis ID not found in database: %s", clean_id)
            return None

        # Build itemized clauses
        raw_evals = data.get("evaluations", [])
        clause_list: List[ClauseContext] = []

        for ev in raw_evals:
            c = ClauseContext(
                requirement_id=str(ev.get("requirement_id", "REQ-UNKNOWN")),
                requirement_text=str(ev.get("requirement_text", "")),
                category=str(ev.get("category", "")),
                parameter=str(ev.get("parameter", "")),
                operator=str(ev.get("operator", ">=")),
                required_value=ev.get("required_value"),
                required_unit=ev.get("required_unit"),
                is_mandatory=bool(ev.get("is_mandatory", False)),
                official_verdict=str(ev.get("label", "UNCERTAIN")).strip(),
                confidence=float(ev.get("confidence", 0.0) or 0.0),
                top_similarity_score=float(ev.get("top_similarity_score", 0.0) or 0.0),
                top_evidence_sentence=ev.get("top_evidence_sentence"),
                source_document=ev.get("top_evidence_source"),
                page_number=ev.get("top_evidence_page"),
                section=ev.get("top_evidence_section"),
                reasoning=str(ev.get("reasoning", "")),
                numeric_comparison=ev.get("numeric_comparison"),
                cross_doc_conflict=bool(ev.get("cross_doc_conflict", False)),
                cross_doc_conflict_type=ev.get("cross_doc_conflict_type"),
                cross_doc_conflict_severity=ev.get("cross_doc_conflict_severity"),
                cross_doc_reasoning=ev.get("cross_doc_reasoning"),
                cross_doc_primary_source=ev.get("cross_doc_primary_source"),
                cross_doc_authoritative_source=ev.get("cross_doc_authoritative_source"),
                cross_doc_authoritative_page=ev.get("cross_doc_authoritative_page"),
            )
            clause_list.append(c)

        return AnalysisContext(
            analysis_id=clean_id,
            tender_filename=str(data.get("tender_filename", "TENDER.pdf")),
            bidder_filename=str(data.get("bidder_filename", "BIDDER.pdf")),
            bidder_name=str(data.get("bidder_name", "Unknown Bidder")),
            overall_status=str(data.get("overall_status", "UNCERTAIN")),
            total_requirements=int(data.get("total_requirements", len(clause_list))),
            compliant_count=int(data.get("compliant_count", 0)),
            non_compliant_count=int(data.get("non_compliant_count", 0)),
            uncertain_count=int(data.get("uncertain_count", 0)),
            missing_evidence_count=int(data.get("missing_evidence_count", 0)),
            mandatory_shortfalls_count=int(data.get("mandatory_shortfalls_count", 0)),
            retrieval_method=str(data.get("retrieval_method", "tfidf")),
            nli_enabled=bool(data.get("nli_enabled", True)),
            nli_model=data.get("nli_model"),
            annexure_documents=list(data.get("annexure_documents") or []),
            clauses=clause_list,
        )
