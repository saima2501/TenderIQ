"""
tfidf_retriever.py - TF-IDF Evidence Retrieval Pipeline
Part of Phase 4: Baseline Demo Pipeline

Retrieves the top-k most relevant evidence sentences from bidder proposals
for each tender requirement using term frequency-inverse document frequency
and cosine similarity.
"""

import math
import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Tuple, Optional

# Basic English stop words for procurement IR
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our",
    "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "whom", "why", "with", "would", "you", "your", "yours",
    "yourself", "yourselves"
}


@dataclass
class RetrievedEvidenceCandidate:
    """Represents a single retrieved evidence sentence with similarity score and rank."""
    rank: int                             # 1, 2, 3...
    evidence_id: str                      # e.g. "EVD-001"
    sentence: str                         # Sentence text
    similarity_score: float               # Cosine similarity (0.0 to 1.0)
    source_document: str                  # Bidder filename
    page_number: int                      # 1-indexed page number
    section_reference: Optional[str] = None
    has_negation: bool = False
    numeric_values: List[Dict[str, Any]] = field(default_factory=list)
    extracted_entities: List[Dict[str, Any]] = field(default_factory=list)
    uncertainty_flag: bool = False
    review_notes: str = ""
    retrieval_method: str = "tfidf"       # 'tfidf', 'semantic (model-name)', 'tfidf_fallback (...)'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RequirementRetrievalResult:
    """Contains top-k retrieved candidates for a single requirement."""
    requirement_id: str
    requirement_text: str
    parameter: str
    candidates: List[RetrievedEvidenceCandidate] = field(default_factory=list)
    top_similarity: float = 0.0
    has_candidate: bool = False
    retrieval_method: str = "tfidf"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["candidates"] = [c.to_dict() for c in self.candidates]
        return d


class TFIDFRetriever:
    """Lightweight, mathematically exact TF-IDF and Cosine Similarity retriever."""

    def __init__(self, min_similarity_threshold: float = 0.08):
        self.min_similarity_threshold = min_similarity_threshold

    def tokenize(self, text: str) -> List[str]:
        """Lowercases, extracts alphanumeric tokens and filters stop words."""
        if not text:
            return []
        raw_tokens = re.findall(r"\b[A-Za-z0-9\-\.]{2,}\b", text.lower())
        tokens = [t for t in raw_tokens if t not in STOP_WORDS and not t.isdigit()]
        return tokens

    def compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Calculates term frequency for a document."""
        tf = {}
        if not tokens:
            return tf
        for t in tokens:
            tf[t] = tf.get(t, 0.0) + 1.0
        total = float(len(tokens))
        for t in tf:
            tf[t] /= total
        return tf

    def compute_idf(self, corpus_tokens: List[List[str]]) -> Dict[str, float]:
        """Calculates inverse document frequency with smoothed formulation."""
        num_docs = float(len(corpus_tokens))
        df = {}
        for doc in corpus_tokens:
            unique_terms = set(doc)
            for t in unique_terms:
                df[t] = df.get(t, 0) + 1

        idf = {}
        for t, count in df.items():
            # Standard smoothed IDF formula
            idf[t] = math.log((num_docs + 1.0) / (float(count) + 1.0)) + 1.0
        return idf

    def vectorize(self, tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
        """Builds a normalized TF-IDF vector dictionary."""
        tf = self.compute_tf(tokens)
        vec = {}
        for t, freq in tf.items():
            weight = freq * idf.get(t, 1.0)
            if weight > 0:
                vec[t] = weight

        # Compute Euclidean norm
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            for t in vec:
                vec[t] /= norm
        return vec

    def cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Computes dot product between two L2-normalized sparse vector dictionaries."""
        if not vec1 or not vec2:
            return 0.0
        # Iterate over smaller dictionary for efficiency
        if len(vec1) > len(vec2):
            vec1, vec2 = vec2, vec1
        dot = sum(val * vec2[term] for term, val in vec1.items() if term in vec2)
        return max(0.0, min(1.0, float(dot)))

    def retrieve_top_k(
        self,
        requirements: List[Any],
        evidence_records: List[Any],
        top_k: int = 3
    ) -> List[RequirementRetrievalResult]:
        """Retrieves top-k candidate evidence statements for each requirement."""
        if not requirements or not evidence_records:
            return []

        # 1. Prepare corpus tokens from evidence sentences and requirement queries
        all_doc_tokens = []
        evidence_token_list = []
        for evd in evidence_records:
            toks = self.tokenize(getattr(evd, "original_sentence", ""))
            evidence_token_list.append(toks)
            all_doc_tokens.append(toks)

        for req in requirements:
            query_text = f"{getattr(req, 'original_text', '')} {getattr(req, 'parameter', '')}"
            all_doc_tokens.append(self.tokenize(query_text))

        # 2. Compute IDF across entire collection
        idf = self.compute_idf(all_doc_tokens)

        # 3. Vectorize all candidate evidence records
        evidence_vectors = [self.vectorize(toks, idf) for toks in evidence_token_list]

        # 4. For each requirement, rank candidate evidence sentences
        results: List[RequirementRetrievalResult] = []

        for req in requirements:
            req_id = getattr(req, "requirement_id", "REQ-UNKNOWN")
            req_text = getattr(req, "original_text", "")
            req_param = getattr(req, "parameter", "")

            # Query vector combines requirement clause text + parameter keywords
            query_str = f"{req_text} {req_param}"
            query_tokens = self.tokenize(query_str)
            query_vec = self.vectorize(query_tokens, idf)

            scored_candidates: List[Tuple[float, Any]] = []
            for evd, evd_vec in zip(evidence_records, evidence_vectors):
                sim = self.cosine_similarity(query_vec, evd_vec)
                scored_candidates.append((sim, evd))

            # Sort by similarity descending
            scored_candidates.sort(key=lambda x: x[0], reverse=True)

            top_candidates: List[RetrievedEvidenceCandidate] = []
            for rank_idx, (sim_score, evd_obj) in enumerate(scored_candidates[:top_k], start=1):
                top_candidates.append(RetrievedEvidenceCandidate(
                    rank=rank_idx,
                    evidence_id=getattr(evd_obj, "evidence_id", f"EVD-{rank_idx}"),
                    sentence=getattr(evd_obj, "original_sentence", ""),
                    similarity_score=round(sim_score, 4),
                    source_document=getattr(evd_obj, "source_document", "N/A"),
                    page_number=getattr(evd_obj, "page_number", 1),
                    section_reference=getattr(evd_obj, "section_reference", None),
                    has_negation=getattr(evd_obj, "has_negation", False),
                    numeric_values=getattr(evd_obj, "numeric_values", []),
                    extracted_entities=getattr(evd_obj, "extracted_entities", []),
                    uncertainty_flag=getattr(evd_obj, "uncertainty_flag", False),
                    review_notes=getattr(evd_obj, "review_notes", "")
                ))

            top_sim = top_candidates[0].similarity_score if top_candidates else 0.0
            has_cand = bool(top_candidates and top_sim >= self.min_similarity_threshold)

            results.append(RequirementRetrievalResult(
                requirement_id=req_id,
                requirement_text=req_text,
                parameter=req_param,
                candidates=top_candidates,
                top_similarity=top_sim,
                has_candidate=has_cand
            ))

        return results
