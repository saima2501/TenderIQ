"""
backend.assistant - Local AI Explanation & Chatbot Service for TenderIQ
Part of Phase 8: Core NLP Reliability Improvements (Step 2G.3 & 2G.4)

Provides localized, grounded explanations and conversational assistance for existing
compliance verification results using Qwen (via Ollama) or deterministic fallback.
"""

from backend.assistant.ollama_client import OllamaClient
from backend.assistant.explanation_service import ExplanationService, ExplanationResult
from backend.assistant.fallback import DeterministicFallbackExplainer
from backend.assistant.context_builder import AnalysisContextBuilder, AnalysisContext, ClauseContext
from backend.assistant.chat_service import ChatService, ChatResult, AnalysisNotFoundError, InvalidChatInputError
from backend.assistant.prompts import (
    SYSTEM_EXPLANATION_PROMPT,
    SYSTEM_CHAT_PROMPT,
    build_clause_prompt,
    build_chat_prompt,
)

__all__ = [
    "OllamaClient",
    "ExplanationService",
    "ExplanationResult",
    "DeterministicFallbackExplainer",
    "AnalysisContextBuilder",
    "AnalysisContext",
    "ClauseContext",
    "ChatService",
    "ChatResult",
    "AnalysisNotFoundError",
    "InvalidChatInputError",
    "SYSTEM_EXPLANATION_PROMPT",
    "SYSTEM_CHAT_PROMPT",
    "build_clause_prompt",
    "build_chat_prompt",
]
