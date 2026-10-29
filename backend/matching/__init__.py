"""
backend.matching package
Evidence retrieval modules (TF-IDF Baseline & Sentence-BERT Semantic)
"""

from .tfidf_retriever import (
    TFIDFRetriever,
    RetrievedEvidenceCandidate,
    RequirementRetrievalResult
)
from .semantic_retriever import SemanticRetriever

__all__ = [
    "TFIDFRetriever",
    "SemanticRetriever",
    "RetrievedEvidenceCandidate",
    "RequirementRetrievalResult"
]

