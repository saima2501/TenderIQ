"""
chat_service.py - Grounded Conversational AI Chat Service
Part of Step 2G.4: AI Chatbot Backend Implementation

Answers evaluator questions about a specific TenderIQ bid analysis using stored
compliance results and evidence as the sole source of truth.
Enforces verdict immutability, prompt injection resistance, and 100% offline fallback.
"""

import re
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Union

from backend.assistant.ollama_client import OllamaClient
from backend.assistant.context_builder import AnalysisContextBuilder, AnalysisContext, ClauseContext
from backend.assistant.prompts import SYSTEM_CHAT_PROMPT, build_chat_prompt

logger = logging.getLogger(__name__)

MAX_USER_MESSAGE_LENGTH = 1000
MAX_HISTORY_LENGTH = 10
MAX_HISTORY_MSG_LENGTH = 1000
SANCTIONED_VERDICTS = {"COMPLIANT", "NON_COMPLIANT", "UNCERTAIN", "MISSING_EVIDENCE"}


class AnalysisNotFoundError(Exception):
    """Raised when the requested analysis_id is not found in the database."""
    pass


class InvalidChatInputError(ValueError):
    """Raised when user message or history fails validation."""
    pass


@dataclass
class ChatResult:
    """Structured response from the Chat Service."""
    analysis_id: str
    answer: str
    citations: List[Dict[str, Any]]
    model_used: str
    fallback_used: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ChatService:
    """Conversational assistant service grounded strictly in stored analysis results."""

    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        context_builder: Optional[AnalysisContextBuilder] = None,
    ):
        self.ollama_client = ollama_client or OllamaClient()
        self.context_builder = context_builder or AnalysisContextBuilder()

    def validate_inputs(
        self,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        """Validates user message and conversation history limits and formats."""
        if not user_message or not user_message.strip():
            raise InvalidChatInputError("User message cannot be empty or whitespace.")

        if len(user_message) > MAX_USER_MESSAGE_LENGTH:
            raise InvalidChatInputError(
                f"User message exceeds maximum length of {MAX_USER_MESSAGE_LENGTH} characters."
            )

        if history:
            if len(history) > MAX_HISTORY_LENGTH:
                raise InvalidChatInputError(
                    f"Conversation history exceeds maximum limit of {MAX_HISTORY_LENGTH} messages."
                )

            for idx, msg in enumerate(history):
                if not isinstance(msg, dict):
                    raise InvalidChatInputError(f"History entry at index {idx} must be a dictionary.")
                role = msg.get("role", "")
                if role not in ("user", "assistant"):
                    raise InvalidChatInputError(f"Invalid history role '{role}' at index {idx}. Must be 'user' or 'assistant'.")
                content = msg.get("content", "")
                if len(content) > MAX_HISTORY_MSG_LENGTH:
                    raise InvalidChatInputError(
                        f"History message at index {idx} exceeds {MAX_HISTORY_MSG_LENGTH} characters."
                    )

    def _validate_ai_chat_response(self, text: str, context: AnalysisContext) -> bool:
        """Validates that Qwen's response is safe, well-formed, and does not contradict stored facts.

        Rejects output if:
        - It is empty or excessively short (< 20 characters).
        - It claims first-person authority override ("I decided", "I have approved").
        - It claims an overall status that directly contradicts the official overall status.
        """
        if not text or len(text.strip()) < 20:
            return False

        upper_text = text.upper()

        # Reject first-person authority claims
        first_person_patterns = [
            "I HAVE DETERMINED",
            "I HAVE DECIDED",
            "I HAVE APPROVED",
            "I CONCLUDE THE BIDDER IS FULLY COMPLIANT",
            "IN MY OPINION THE BIDDER PASSES",
        ]
        for pat in first_person_patterns:
            if pat in upper_text:
                logger.warning("Rejected AI chat response: first-person authority claim.")
                return False

        # Reject direct contradiction of overall status if mentioned
        if context.overall_status == "NON_COMPLIANT":
            if "OVERALL STATUS IS COMPLIANT" in upper_text or "OVERALL VERDICT IS COMPLIANT" in upper_text:
                logger.warning("Rejected AI chat response: contradicted overall status.")
                return False

        return True

    def _extract_citations_for_query(
        self,
        context: AnalysisContext,
        query: str,
        answer_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Collects genuine citations from stored clauses matching the query or answer.

        When cross-document discrepancies are discussed, emits both the authoritative
        annexure citation and the primary proposal document citation.
        """
        combined_text = f"{query} {answer_text or ''}".upper()
        citations: List[Dict[str, Any]] = []
        seen_keys = set()

        # 1. Search for specific requirement IDs mentioned (e.g. REQ-001)
        found_clauses: List[ClauseContext] = []
        for clause in context.clauses:
            if clause.requirement_id.upper() in combined_text:
                found_clauses.append(clause)

        # 2. Check for cross-doc conflict / discrepancy inquiries
        is_conflict_query = any(w in combined_text for w in ["CONFLICT", "DISCREPANC", "CONTRADICTION", "CROSS-DOC"])
        if is_conflict_query:
            for clause in context.clauses:
                if clause.cross_doc_conflict and clause not in found_clauses:
                    found_clauses.append(clause)

        # 3. If no requirement ID mentioned and not conflict query, match keywords
        if not found_clauses:
            keywords = ["TURNOVER", "SLA", "UPTIME", "ISO", "EMD", "WARRANTY", "MTTR", "EXPERIENCE", "GST", "PAN"]
            for kw in keywords:
                if kw in combined_text:
                    for clause in context.clauses:
                        if kw in (clause.parameter or "").upper() or kw in clause.requirement_text.upper():
                            if clause not in found_clauses:
                                found_clauses.append(clause)

        # 4. Format citations from found clauses
        for c in found_clauses:
            # If cross-document discrepancy, cite the authoritative annexure first
            if c.cross_doc_conflict and c.cross_doc_authoritative_source:
                auth_doc = c.cross_doc_authoritative_source
                auth_page = c.cross_doc_authoritative_page
                key_auth = (auth_doc, auth_page)
                if key_auth not in seen_keys:
                    seen_keys.add(key_auth)
                    citations.append({
                        "source_document": auth_doc,
                        "page_number": auth_page,
                        "has_citation": True
                    })

            # Also cite the primary proposal document containing the original claim
            doc = c.cross_doc_primary_source or c.source_document
            page = c.page_number
            if doc and doc != "N/A":
                key_doc = (doc, page)
                if key_doc not in seen_keys:
                    seen_keys.add(key_doc)
                    citations.append({
                        "source_document": doc,
                        "page_number": page,
                        "has_citation": True
                    })

        # 5. If still no citation, return explicit notice
        if not citations:
            citations.append({
                "source_document": "Source citation unavailable",
                "page_number": None,
                "has_citation": False
            })

        return citations

    def _deterministic_chat_fallback(
        self,
        context: AnalysisContext,
        user_message: str,
    ) -> ChatResult:
        """Rule-based, 100% offline deterministic chat responder.

        Generates accurate, audit-ready responses from stored database records
        without calling any external AI model.
        """
        msg_upper = user_message.upper()

        # 1. Check for specific Requirement ID inquiry (e.g. REQ-001, REQ-1)
        req_match = re.search(r"\b(REQ-?\d+)\b", msg_upper)
        target_clause: Optional[ClauseContext] = None

        if req_match:
            raw_id = req_match.group(1).replace("-", "")
            for c in context.clauses:
                if c.requirement_id.replace("-", "").upper() == raw_id:
                    target_clause = c
                    break

        # If not matched by ID, match by specific key parameters
        if not target_clause:
            param_map = {
                "TURNOVER": ["TURNOVER", "REVENUE", "FINANCIAL"],
                "SLA": ["SLA", "UPTIME", "AVAILABILITY"],
                "ISO": ["ISO", "ISO 27001", "ISO 9001", "CERTIFICATION"],
                "EMD": ["EMD", "EARNEST MONEY", "BID SECURITY"],
                "WARRANTY": ["WARRANTY", "GUARANTEE"],
                "MTTR": ["MTTR", "RESOLUTION TIME", "RESPONSE TIME"],
                "EXPERIENCE": ["EXPERIENCE", "PAST WORK", "CREDENTIAL"],
            }
            for param, terms in param_map.items():
                if any(t in msg_upper for t in terms):
                    for c in context.clauses:
                        if param in (c.parameter or "").upper() or any(t in c.requirement_text.upper() for t in terms):
                            target_clause = c
                            break
                    if target_clause:
                        break

        # Response Case A: Inquiring about a specific requirement
        if target_clause:
            doc_info = f"[{target_clause.source_document}, Page {target_clause.page_number}]" if target_clause.source_document and target_clause.source_document != "N/A" else "[Source citation unavailable]"
            ans_parts = [
                f"According to the stored TenderIQ analysis, requirement {target_clause.requirement_id} ('{target_clause.requirement_text}') was evaluated as {target_clause.official_verdict}."
            ]

            if target_clause.numeric_comparison and isinstance(target_clause.numeric_comparison, dict):
                comp = target_clause.numeric_comparison
                ans_parts.append(f"Numeric comparison: Required {comp.get('operator', '')} {comp.get('required_raw', '')} {comp.get('required_unit', '')}, Offered {comp.get('offered_raw', '')} {comp.get('offered_unit', '')} ({comp.get('status', '')}).")

            if target_clause.reasoning:
                ans_parts.append(f"Reasoning: {target_clause.reasoning}")

            if target_clause.cross_doc_conflict and target_clause.cross_doc_reasoning:
                ans_parts.append(f"Discrepancy: {target_clause.cross_doc_reasoning}")

            if target_clause.official_verdict == "MISSING_EVIDENCE":
                ans_parts.append("Procurement notice: Missing evidence requires clarification or formal submission, not automatic technical disqualification.")

            ans_parts.append(f"Evidence citation: {doc_info}.")
            answer = " ".join(ans_parts)
            citations = self._extract_citations_for_query(context, user_message, answer)
            return ChatResult(
                analysis_id=context.analysis_id,
                answer=answer,
                citations=citations,
                model_used="deterministic_fallback",
                fallback_used=True,
            )

        # Response Case B: Missing Evidence Inquiry
        if any(w in msg_upper for w in ["MISSING", "OMITTED", "ABSENT", "UNLOCATED"]):
            missing = [c for c in context.clauses if c.official_verdict == "MISSING_EVIDENCE"]
            if missing:
                req_names = ", ".join(f"{c.requirement_id} ({c.parameter or c.requirement_text[:30]})" for c in missing)
                answer = (
                    f"According to the stored TenderIQ analysis, {len(missing)} requirement(s) were marked MISSING_EVIDENCE: {req_names}. "
                    f"In accordance with procurement evaluation guidelines, missing evidence findings require formal document submission or procurement clarification rather than automatic technical disqualification."
                )
            else:
                answer = "According to the stored TenderIQ analysis, no requirements were marked MISSING_EVIDENCE. All required documents were located in the submission."
            citations = self._extract_citations_for_query(context, user_message, answer)
            return ChatResult(
                analysis_id=context.analysis_id,
                answer=answer,
                citations=citations,
                model_used="deterministic_fallback",
                fallback_used=True,
            )

        # Response Case C: Discrepancies / Cross-Document Conflicts Inquiry
        if any(w in msg_upper for w in ["CONFLICT", "DISCREPANCY", "CONTRADICTION", "CROSS-DOC"]):
            conflicts = [c for c in context.clauses if c.cross_doc_conflict]
            if conflicts:
                details = "; ".join(f"{c.requirement_id}: {c.cross_doc_reasoning}" for c in conflicts if c.cross_doc_reasoning)
                answer = f"According to the stored TenderIQ analysis, cross-document discrepancies were detected in {len(conflicts)} clause(s): {details}."
            else:
                answer = "According to the stored TenderIQ analysis, zero cross-document discrepancies or conflicts were detected across the proposal and annexures."
            citations = self._extract_citations_for_query(context, user_message, answer)
            return ChatResult(
                analysis_id=context.analysis_id,
                answer=answer,
                citations=citations,
                model_used="deterministic_fallback",
                fallback_used=True,
            )

        # Response Case D: Overall Status / Summary / Breakdown Inquiry
        if any(w in msg_upper for w in ["SUMMARY", "STATUS", "OVERALL", "BREAKDOWN", "OVERVIEW", "RESULT", "HOW MANY"]):
            answer = (
                f"According to the stored TenderIQ analysis for Bidder '{context.bidder_name}' ({context.bidder_filename}) against tender '{context.tender_filename}', "
                f"the official overall status is '{context.overall_status}'. "
                f"Breakdown across {context.total_requirements} evaluated requirements: "
                f"{context.compliant_count} Compliant, {context.non_compliant_count} Non-Compliant, "
                f"{context.uncertain_count} Uncertain, and {context.missing_evidence_count} Missing Evidence "
                f"(Mandatory Shortfalls: {context.mandatory_shortfalls_count})."
            )
            citations = self._extract_citations_for_query(context, user_message, answer)
            return ChatResult(
                analysis_id=context.analysis_id,
                answer=answer,
                citations=citations,
                model_used="deterministic_fallback",
                fallback_used=True,
            )

        # Response Case E: Non-Compliant clauses inquiry
        if any(w in msg_upper for w in ["NON COMPLIANT", "NON-COMPLIANT", "FAIL", "FAILED", "SHORTFALL", "VIOLAT"]):
            non_comp = [c for c in context.clauses if c.official_verdict == "NON_COMPLIANT"]
            if non_comp:
                details = "; ".join(f"{c.requirement_id} ({c.parameter or c.requirement_text[:30]}): {c.reasoning}" for c in non_comp)
                answer = f"According to the stored TenderIQ analysis, {len(non_comp)} requirement(s) were determined to be NON_COMPLIANT: {details}."
            else:
                answer = "According to the stored TenderIQ analysis, zero requirements were evaluated as NON_COMPLIANT."
            citations = self._extract_citations_for_query(context, user_message, answer)
            return ChatResult(
                analysis_id=context.analysis_id,
                answer=answer,
                citations=citations,
                model_used="deterministic_fallback",
                fallback_used=True,
            )

        # Default Fallback: Prompt user to specify a requirement ID or category
        answer = (
            f"According to the stored TenderIQ analysis for Bidder '{context.bidder_name}' ({context.bidder_filename}), "
            f"the official overall compliance status is '{context.overall_status}'. "
            f"The assistant is currently operating in deterministic fallback mode. "
            f"Please specify a requirement ID (e.g., REQ-001) or ask about overall summary, missing evidence, or discrepancies."
        )
        citations = self._extract_citations_for_query(context, user_message, answer)
        return ChatResult(
            analysis_id=context.analysis_id,
            answer=answer,
            citations=citations,
            model_used="deterministic_fallback",
            fallback_used=True,
        )

    def chat(
        self,
        analysis_id: str,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> ChatResult:
        """Processes an evaluator question for a specific analysis.

        Grounded solely in the database analysis context for analysis_id.
        """
        # 1. Input Validation
        self.validate_inputs(user_message=user_message, history=history)

        # 2. Context Retrieval (Isolated to analysis_id)
        context = self.context_builder.build(analysis_id)
        if not context:
            raise AnalysisNotFoundError(f"Analysis '{analysis_id}' was not found in the database.")

        # 3. Attempt Qwen Generation via Ollama
        generated_answer: Optional[str] = None
        try:
            context_str = context.format_for_prompt(max_clauses=8, focus_query=user_message)
            prompt = build_chat_prompt(
                context_str=context_str,
                user_message=user_message,
                history=history,
            )

            raw_ai_text = self.ollama_client.generate(
                prompt=prompt,
                system=SYSTEM_CHAT_PROMPT,
                temperature=0.1,
                max_tokens=400,
            )

            if raw_ai_text and self._validate_ai_chat_response(raw_ai_text, context):
                generated_answer = raw_ai_text.strip()
        except Exception as exc:
            logger.warning("Error generating AI chat completion: %s", exc)
            generated_answer = None

        # 4. Return Qwen Result or Safe Deterministic Fallback
        if generated_answer:
            citations = self._extract_citations_for_query(
                context=context,
                query=user_message,
                answer_text=generated_answer,
            )
            return ChatResult(
                analysis_id=context.analysis_id,
                answer=generated_answer,
                citations=citations,
                model_used=self.ollama_client.model_name,
                fallback_used=False,
            )

        # Fallback path
        return self._deterministic_chat_fallback(context=context, user_message=user_message)
