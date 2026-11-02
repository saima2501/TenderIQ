"""
explanation_service.py - Local AI Compliance Explanation Service
Part of Step 2G.3: Local AI Explanation Service

Coordinates Ollama Qwen model and deterministic fallback to provide explainable,
grounded compliance summaries. The service is strictly explanatory; it CANNOT
modify or assign compliance verdicts.
"""

import re
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Union

from backend.assistant.ollama_client import OllamaClient
from backend.assistant.prompts import SYSTEM_EXPLANATION_PROMPT, build_clause_prompt
from backend.assistant.fallback import DeterministicFallbackExplainer

logger = logging.getLogger(__name__)

SANCTIONED_VERDICTS = {"COMPLIANT", "NON_COMPLIANT", "UNCERTAIN", "MISSING_EVIDENCE"}


@dataclass
class ExplanationResult:
    """Structured response from the Explanation Service."""
    official_verdict: str                  # Immutable ground truth from compliance engine
    requirement_id: str
    explanation: str                       # Plain professional English explanation
    evidence_summary: str                  # Summary of cited evidence or notice of absence
    citations: List[Dict[str, Any]]        # [{source_document, page_number, has_citation}]
    model_used: str                        # "qwen2.5:1.5b" or "deterministic-fallback"
    fallback_used: bool                    # True if deterministic fallback generated explanation
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExplanationService:
    """Local AI explanation service with deterministic offline fallback."""

    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        fallback_explainer: Optional[DeterministicFallbackExplainer] = None,
    ):
        self.client = ollama_client or OllamaClient()
        self.fallback = fallback_explainer or DeterministicFallbackExplainer()

    def _extract_field(self, obj: Union[Dict[str, Any], Any], key: str, default: Any = None) -> Any:
        """Safely extracts field from either a dict or an object."""
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _validate_ai_response(self, text: str, official_verdict: str) -> bool:
        """Validates that Qwen's response is safe, well-formed, and does not dispute the official verdict.

        Rejects responses that:
        - Are empty or excessively short
        - Attempt to override the official verdict with a conflicting verdict
        - Contain common hallucinations or first-person adjudications ("I have determined this is compliant")
        """
        if not text or len(text.strip()) < 20:
            return False

        upper_text = text.upper()

        # Check for verdict conflict: If official is NON_COMPLIANT, but model claims COMPLIANT (without negation)
        if official_verdict in ("NON_COMPLIANT", "UNCERTAIN", "MISSING_EVIDENCE"):
            if "I HAVE DETERMINED THAT THIS BIDDER IS COMPLIANT" in upper_text:
                logger.warning("Rejected AI explanation: attempted to override official verdict.")
                return False
            if "I CONCLUDE THE BIDDER IS FULLY COMPLIANT" in upper_text:
                logger.warning("Rejected AI explanation: first-person authority override.")
                return False

        # Reject explicit contradiction of official verdict
        conflicting_labels = SANCTIONED_VERDICTS - {official_verdict}
        for wrong_lbl in conflicting_labels:
            # Match patterns like: "is actually COMPLIANT" or "should be COMPLIANT"
            if re.search(rf"\b(?:is\s+actually|should\s+be|verdict\s+is)\s+{wrong_lbl}\b", upper_text):
                logger.warning("Rejected AI explanation: disputed official verdict with '%s'.", wrong_lbl)
                return False

        return True

    def explain_clause(self, clause_data: Union[Dict[str, Any], Any]) -> ExplanationResult:
        """Generates an explanation for a single evaluated tender requirement.

        MANDATORY SAFETY GUARANTEE:
        The 'official_verdict' field is copied directly from the supplied pipeline data.
        The AI model is never queried for a verdict and cannot change it.
        """
        # 1. Extract Immutable Ground Truth
        raw_label = str(self._extract_field(clause_data, "label", "UNCERTAIN")).strip()
        official_verdict = raw_label if raw_label in SANCTIONED_VERDICTS else "UNCERTAIN"

        req_id = str(self._extract_field(clause_data, "requirement_id", "REQ-UNKNOWN"))
        req_text = str(self._extract_field(clause_data, "requirement_text", ""))
        reasoning = str(self._extract_field(clause_data, "reasoning", ""))
        evidence_sent = self._extract_field(clause_data, "top_evidence_sentence", None)
        source_doc = self._extract_field(clause_data, "top_evidence_source", None)
        page_num = self._extract_field(clause_data, "top_evidence_page", None)
        num_comp = self._extract_field(clause_data, "numeric_comparison", None)
        cross_doc = bool(self._extract_field(clause_data, "cross_doc_conflict", False))
        cross_doc_rsn = self._extract_field(clause_data, "cross_doc_reasoning", None)
        conf = float(self._extract_field(clause_data, "confidence", 1.0) or 1.0)

        # 2. Build Deterministic Baseline Components (always computed for citations & fallback)
        fb_explanation, fb_evidence_summary, fb_citations = self.fallback.explain(
            requirement_id=req_id,
            requirement_text=req_text,
            official_verdict=official_verdict,
            reasoning=reasoning,
            top_evidence_sentence=evidence_sent,
            source_document=source_doc,
            page_number=page_num,
            numeric_comparison=num_comp,
            cross_doc_conflict=cross_doc,
            cross_doc_reasoning=cross_doc_rsn,
        )

        # 3. Attempt Local Qwen Generation via Ollama
        generated_explanation: Optional[str] = None
        try:
            prompt = build_clause_prompt(
                requirement_id=req_id,
                requirement_text=req_text,
                official_verdict=official_verdict,
                reasoning=reasoning,
                top_evidence_sentence=evidence_sent,
                source_document=source_doc,
                page_number=page_num,
                numeric_comparison=num_comp,
                cross_doc_conflict=cross_doc,
                cross_doc_reasoning=cross_doc_rsn,
            )

            raw_ai_text = self.client.generate(
                prompt=prompt,
                system=SYSTEM_EXPLANATION_PROMPT,
                temperature=0.1,
                max_tokens=300,
            )

            if raw_ai_text and self._validate_ai_response(raw_ai_text, official_verdict):
                generated_explanation = raw_ai_text.strip()
        except Exception as exc:
            logger.warning("Error during Qwen explanation generation: %s", exc)
            generated_explanation = None

        # 4. Construct Final Safe Explanation Result
        if generated_explanation:
            return ExplanationResult(
                official_verdict=official_verdict,
                requirement_id=req_id,
                explanation=generated_explanation,
                evidence_summary=fb_evidence_summary,
                citations=fb_citations,
                model_used=self.client.model_name,
                fallback_used=False,
                confidence=conf,
            )

        # Safe Fallback
        return ExplanationResult(
            official_verdict=official_verdict,
            requirement_id=req_id,
            explanation=fb_explanation,
            evidence_summary=fb_evidence_summary,
            citations=fb_citations,
            model_used="deterministic-fallback",
            fallback_used=True,
            confidence=conf,
        )
