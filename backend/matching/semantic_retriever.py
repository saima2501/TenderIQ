"""
semantic_retriever.py - Sentence-BERT Semantic Evidence Retrieval Pipeline
Part of Phase 5: Semantic Evidence Retrieval & Matching

Features:
1. Dense semantic embedding retrieval using Sentence-Transformers (all-MiniLM-L6-v2 by default).
2. CPU-friendly, local inference with zero paid API or external LLM dependencies.
3. Vectorized cosine similarity computation over normalized embeddings.
4. Preserves full auditability: evidence sentence, source document, page number, section,
   negation flags, numeric metrics, entities, and retrieval method provenance.
5. Graceful fallback to TF-IDF when dependencies are missing or model fails to load.
6. Does NOT silently pretend semantic model ran when it did not.
"""

import logging
import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from backend.matching.tfidf_retriever import (
    TFIDFRetriever,
    RetrievedEvidenceCandidate,
    RequirementRetrievalResult
)

logger = logging.getLogger(__name__)

# Try importing SentenceTransformer with graceful fallback
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class SemanticRetriever:
    """Semantic evidence retriever powered by Sentence-Transformers with TF-IDF fallback."""

    DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
    DEFAULT_MIN_SIMILARITY = 0.25

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        min_similarity_threshold: float = DEFAULT_MIN_SIMILARITY,
        fallback_to_tfidf: bool = True,
        device: str = "cpu",
        batch_size: int = 32
    ):
        self.model_name = model_name
        self.min_similarity_threshold = min_similarity_threshold
        self.fallback_to_tfidf = fallback_to_tfidf
        self.device = device
        self.batch_size = batch_size

        self.model: Optional[Any] = None
        self.is_available: bool = False
        self.init_error: Optional[str] = None
        self.tfidf_fallback: Optional[TFIDFRetriever] = None

        if self.fallback_to_tfidf:
            self.tfidf_fallback = TFIDFRetriever(min_similarity_threshold=0.08)

        self._initialize_model()

    def _initialize_model(self):
        """Loads the SentenceTransformer model on CPU with transparent error logging."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self.is_available = False
            self.init_error = "sentence-transformers library is not installed in the environment."
            logger.warning("SemanticRetriever: %s Fallback to TF-IDF: %s", self.init_error, self.fallback_to_tfidf)
            return

        try:
            logger.info("SemanticRetriever: Loading '%s' on %s...", self.model_name, self.device)
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.is_available = True
            self.init_error = None
            logger.info("SemanticRetriever: Successfully loaded '%s'.", self.model_name)
        except Exception as e:
            self.is_available = False
            self.init_error = f"Failed to load SentenceTransformer model '{self.model_name}': {str(e)}"
            logger.error("SemanticRetriever initialization failed: %s", self.init_error)

    @property
    def retrieval_method_name(self) -> str:
        """Returns human-readable name of active retrieval method."""
        if self.is_available:
            return f"semantic ({self.model_name})"
        if self.fallback_to_tfidf:
            return f"tfidf_fallback ({self.init_error or 'semantic model unavailable'})"
        return "none (semantic unavailable)"

    def encode_sentences(self, sentences: List[str]) -> np.ndarray:
        """Computes normalized dense embeddings for a list of sentences."""
        if not self.is_available or self.model is None:
            raise RuntimeError(f"SemanticRetriever cannot encode: {self.init_error}")
        if not sentences:
            return np.empty((0, 384), dtype=np.float32)

        embeddings = self.model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings

    def compute_similarity(self, query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
        """Computes cosine similarity between a single query vector and document matrix.

        Since embeddings are L2 normalized, cosine similarity equals the dot product.
        """
        if doc_embs.ndim == 1:
            doc_embs = doc_embs.reshape(1, -1)
        sims = np.dot(doc_embs, query_emb)
        # Clip to [-1.0, 1.0] to safeguard against float rounding artifacts
        return np.clip(sims, -1.0, 1.0)

    def retrieve_top_k(
        self,
        requirements: List[Any],
        evidence_records: List[Any],
        top_k: int = 3
    ) -> List[RequirementRetrievalResult]:
        """Retrieves top-k semantically relevant evidence sentences for each requirement.

        If the semantic model is unavailable and fallback_to_tfidf is enabled,
        gracefully delegates to TFIDFRetriever while explicitly annotating candidates
        with 'tfidf_fallback (...)' so the system never pretends semantic ran.
        """
        if not requirements or not evidence_records:
            return []

        # Graceful fallback path when semantic model is unavailable
        if not self.is_available:
            if not self.fallback_to_tfidf or self.tfidf_fallback is None:
                raise RuntimeError(
                    f"SemanticRetriever is unavailable and TF-IDF fallback is disabled. "
                    f"Cause: {self.init_error}"
                )
            logger.warning(
                "SemanticRetriever: Delegating to TF-IDF fallback due to unavailable model: %s",
                self.init_error
            )
            fallback_results = self.tfidf_fallback.retrieve_top_k(
                requirements=requirements,
                evidence_records=evidence_records,
                top_k=top_k
            )
            fallback_method = self.retrieval_method_name
            # Re-tag candidates and results with transparent fallback method
            for res in fallback_results:
                res.retrieval_method = fallback_method
                for cand in res.candidates:
                    cand.retrieval_method = fallback_method
                    if not cand.review_notes:
                        cand.review_notes = f"Retrieved via TF-IDF fallback: {self.init_error}"
                    else:
                        cand.review_notes += f" | Retrieved via TF-IDF fallback: {self.init_error}"
            return fallback_results

        # Normal Semantic Retrieval Path
        method_tag = self.retrieval_method_name

        # 1. Extract and encode all candidate evidence sentences in bidder proposal
        evd_sentences = [getattr(e, "original_sentence", "") for e in evidence_records]
        evd_embeddings = self.encode_sentences(evd_sentences)

        # 2. Prepare query strings (clause original text + parameter keywords)
        query_strings = [
            f"{getattr(req, 'original_text', '')} {getattr(req, 'parameter', '')}".strip()
            for req in requirements
        ]
        query_embeddings = self.encode_sentences(query_strings)

        # 3. For each requirement, rank candidate evidence statements by cosine similarity
        results: List[RequirementRetrievalResult] = []

        for req_idx, req in enumerate(requirements):
            req_id = getattr(req, "requirement_id", "REQ-UNKNOWN")
            req_text = getattr(req, "original_text", "")
            req_param = getattr(req, "parameter", "")
            q_emb = query_embeddings[req_idx]

            # Compute similarities across all candidate evidence sentences
            sim_scores = self.compute_similarity(q_emb, evd_embeddings)

            # Pair scores with original evidence objects and sort descending
            scored_candidates: List[Tuple[float, Any]] = [
                (float(sim_scores[i]), evidence_records[i])
                for i in range(len(evidence_records))
            ]
            scored_candidates.sort(key=lambda x: x[0], reverse=True)

            # Build top-k candidate structures with full traceability
            top_candidates: List[RetrievedEvidenceCandidate] = []
            for rank_idx, (sim_score, evd_obj) in enumerate(scored_candidates[:top_k], start=1):
                top_candidates.append(RetrievedEvidenceCandidate(
                    rank=rank_idx,
                    evidence_id=getattr(evd_obj, "evidence_id", f"EVD-{rank_idx}"),
                    sentence=getattr(evd_obj, "original_sentence", ""),
                    similarity_score=round(max(0.0, sim_score), 4),
                    source_document=getattr(evd_obj, "source_document", "N/A"),
                    page_number=getattr(evd_obj, "page_number", 1),
                    section_reference=getattr(evd_obj, "section_reference", None),
                    has_negation=getattr(evd_obj, "has_negation", False),
                    numeric_values=getattr(evd_obj, "numeric_values", []),
                    extracted_entities=getattr(evd_obj, "extracted_entities", []),
                    uncertainty_flag=getattr(evd_obj, "uncertainty_flag", False),
                    review_notes=getattr(evd_obj, "review_notes", ""),
                    retrieval_method=method_tag
                ))

            top_sim = top_candidates[0].similarity_score if top_candidates else 0.0
            has_cand = bool(top_candidates and top_sim >= self.min_similarity_threshold)

            results.append(RequirementRetrievalResult(
                requirement_id=req_id,
                requirement_text=req_text,
                parameter=req_param,
                candidates=top_candidates,
                top_similarity=top_sim,
                has_candidate=has_cand,
                retrieval_method=method_tag
            ))

        return results
