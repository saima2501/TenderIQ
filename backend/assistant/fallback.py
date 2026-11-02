"""
fallback.py - Deterministic Rule-Based Fallback Explainer
Part of Step 2G.3: Local AI Explanation Service

Generates audit-ready, professional explanations directly from deterministic
classifier reasoning and metadata when local Ollama is offline or times out.
Requires zero external models or network access.
"""

from typing import Dict, Any, Optional, Tuple, List


class DeterministicFallbackExplainer:
    """Generates structured, grounded explanations directly from pipeline results."""

    def explain(
        self,
        requirement_id: str,
        requirement_text: str,
        official_verdict: str,
        reasoning: str,
        top_evidence_sentence: Optional[str] = None,
        source_document: Optional[str] = None,
        page_number: Optional[int] = None,
        numeric_comparison: Optional[Dict[str, Any]] = None,
        cross_doc_conflict: bool = False,
        cross_doc_reasoning: Optional[str] = None,
    ) -> Tuple[str, str, List[Dict[str, Any]]]:
        """Generates (explanation, evidence_summary, citations)."""
        # 1. Build Citations List
        citations: List[Dict[str, Any]] = []
        if source_document and source_document != "N/A":
            citations.append({
                "source_document": source_document,
                "page_number": page_number,
                "has_citation": True
            })
            citation_str = f"[{source_document}, Page {page_number}]" if page_number is not None else f"[{source_document}]"
        else:
            citations.append({
                "source_document": "Source citation unavailable",
                "page_number": None,
                "has_citation": False
            })
            citation_str = "[Source citation unavailable]"

        # 2. Build Evidence Summary
        evidence_clean = (top_evidence_sentence or "").strip()
        if evidence_clean:
            evidence_summary = f'"{evidence_clean}" {citation_str}'
        else:
            evidence_summary = "No supporting evidence was submitted or located in the bid proposal package."

        # 3. Build Grounded Explanation text by Verdict Class
        engine_reasoning = reasoning.strip() if reasoning else ""

        if official_verdict == "COMPLIANT":
            if numeric_comparison and numeric_comparison.get("status") == "SATISFIED":
                comp_rsn = numeric_comparison.get("reasoning", "")
                explanation = (
                    f"The TenderIQ compliance engine determined this requirement is COMPLIANT. "
                    f"The bidder's submitted evidence confirms satisfaction of the specified threshold: {comp_rsn} "
                    f"Verified from document {citation_str}."
                )
            elif engine_reasoning:
                explanation = (
                    f"The TenderIQ compliance engine determined this requirement is COMPLIANT. "
                    f"{engine_reasoning} "
                    f"Verified from document {citation_str}."
                )
            else:
                explanation = (
                    f"The TenderIQ compliance engine determined this requirement is COMPLIANT. "
                    f"The bidder submitted qualifying evidence that satisfies the mandatory tender requirement. "
                    f"Document citation: {citation_str}."
                )

        elif official_verdict == "NON_COMPLIANT":
            if cross_doc_conflict and cross_doc_reasoning:
                explanation = (
                    f"The TenderIQ compliance engine determined this requirement is NON_COMPLIANT due to an irreconcilable cross-document discrepancy. "
                    f"{cross_doc_reasoning} "
                    f"Evaluated against evidence cited in {citation_str}."
                )
            elif numeric_comparison and numeric_comparison.get("status") == "VIOLATED":
                comp_rsn = numeric_comparison.get("reasoning", "")
                explanation = (
                    f"The TenderIQ compliance engine determined this requirement is NON_COMPLIANT due to a numeric shortfall. "
                    f"{comp_rsn} "
                    f"Extracted from proposal documentation {citation_str}."
                )
            elif engine_reasoning:
                explanation = (
                    f"The TenderIQ compliance engine determined this requirement is NON_COMPLIANT. "
                    f"{engine_reasoning} "
                    f"Evidence citation: {citation_str}."
                )
            else:
                explanation = (
                    f"The TenderIQ compliance engine determined this requirement is NON_COMPLIANT. "
                    f"The submitted proposal evidence fails to satisfy the mandatory tender condition. "
                    f"Citing: {citation_str}."
                )

        elif official_verdict == "UNCERTAIN":
            if engine_reasoning:
                explanation = (
                    f"The TenderIQ compliance engine marked this requirement as UNCERTAIN for human procurement review. "
                    f"{engine_reasoning} "
                    f"Relevant passage identified in {citation_str}."
                )
            else:
                explanation = (
                    f"The TenderIQ compliance engine marked this requirement as UNCERTAIN for procurement auditor review. "
                    f"The submitted evidence lacks independent third-party verification, clear registration details, or legal entity authorization. "
                    f"Citing: {citation_str}."
                )

        elif official_verdict == "MISSING_EVIDENCE":
            explanation = (
                f"The TenderIQ compliance engine marked this requirement as MISSING_EVIDENCE. "
                f"The required mandatory document or undertaking was not found in the submitted bid package. "
                f"This finding requires procurement clarification or formal document submission, rather than automatic technical disqualification."
            )

        else:
            # Fallback for any unknown status
            explanation = (
                f"The TenderIQ compliance engine assigned the status '{official_verdict}'. "
                f"{engine_reasoning} Citing: {citation_str}."
            )

        return explanation, evidence_summary, citations
