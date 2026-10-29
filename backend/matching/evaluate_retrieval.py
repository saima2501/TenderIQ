"""
evaluate_retrieval.py - Benchmark TF-IDF vs Sentence-BERT Semantic Retrieval
Part of Phase 5: Sentence-BERT Semantic Evidence Retrieval

Empirically benchmarks:
1. TF-IDF Lexical Retriever
2. Sentence-BERT Dense Retriever (all-MiniLM-L6-v2)

Metrics:
- Recall@1 (Hits@1)
- Recall@3 (Hits@3)
- Mean Reciprocal Rank (MRR)
- Latency per query (ms)
Breakdown by clause type (direct match, numerical boundary, ambiguous claim, negation conflict).
"""

import json
import os
import sys
import time
from typing import List, Dict, Any, Tuple
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.matching.tfidf_retriever import TFIDFRetriever
from backend.matching.semantic_retriever import SemanticRetriever


class SyntheticEvidenceMock:
    """Mock container mimicking EvidenceRecord for retrieval API compatibility."""
    def __init__(self, sentence: str, evd_id: str, doc: str = "PROPOSAL.pdf", page: int = 1):
        self.evidence_id = evd_id
        self.original_sentence = sentence
        self.source_document = doc
        self.page_number = page
        self.section_reference = "Section 1.0"
        self.has_negation = False
        self.numeric_values = []
        self.extracted_entities = []
        self.uncertainty_flag = False
        self.review_notes = ""


class SyntheticRequirementMock:
    """Mock container mimicking TenderRequirement for retrieval API compatibility."""
    def __init__(self, req_id: str, text: str, param: str = ""):
        self.requirement_id = req_id
        self.original_text = text
        self.parameter = param


def load_test_dataset(test_jsonl_path: str) -> List[Dict[str, Any]]:
    """Loads test split containing requirement-evidence pairs."""
    records = []
    with open(test_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_retrieval_benchmark(
    test_jsonl_path: str = os.path.join(PROJECT_ROOT, "dataset", "splits", "test.jsonl")
) -> Dict[str, Any]:
    """Runs empirical evaluation comparing TF-IDF vs. Sentence-BERT."""
    if not os.path.exists(test_jsonl_path):
        raise FileNotFoundError(f"Test split dataset not found: {test_jsonl_path}")

    all_records = load_test_dataset(test_jsonl_path)

    # Filter to pairs where an actual ground-truth evidence sentence exists
    eval_pairs = [r for r in all_records if r.get("evidence_text") and r.get("evidence_text").strip()]
    missing_pairs = [r for r in all_records if not r.get("evidence_text") or not r.get("evidence_text").strip()]

    print(f"Total test split records: {len(all_records)}", flush=True)
    print(f"Pairs with verifiable ground-truth evidence: {len(eval_pairs)}", flush=True)
    print(f"Pairs with missing evidence (ground-truth omitted): {len(missing_pairs)}", flush=True)

    # Build corpus of candidate evidence sentences (all ground truth sentences + distractors)
    candidate_sentences = list(dict.fromkeys([r["evidence_text"].strip() for r in eval_pairs]))
    candidate_objects = [
        SyntheticEvidenceMock(sentence=s, evd_id=f"EVD-POOL-{idx+1}")
        for idx, s in enumerate(candidate_sentences)
    ]
    print(f"Candidate sentence pool size: {len(candidate_sentences)}", flush=True)

    # Prepare requirement mocks for batch retrieval
    req_mocks = [
        SyntheticRequirementMock(
            req_id=r.get("requirement_id", f"REQ-{idx}"),
            text=r["requirement_text"],
            param=r.get("tender_category", "")
        )
        for idx, r in enumerate(eval_pairs)
    ]

    # Initialize retrievers
    print("\nInitializing TF-IDF Retriever...", flush=True)
    tfidf = TFIDFRetriever(min_similarity_threshold=0.0)

    print("Initializing Semantic Retriever (all-MiniLM-L6-v2 on CPU)...", flush=True)
    semantic = SemanticRetriever(min_similarity_threshold=0.0, device="cpu")

    models = {
        "TF-IDF": tfidf,
        "Sentence-BERT (all-MiniLM-L6-v2)": semantic
    }

    results = {}

    for name, retriever in models.items():
        print(f"\n--- Evaluating {name} (batch vectorized retrieval) ---", flush=True)
        t0 = time.perf_counter()

        # Execute batch retrieval for all requirements against the candidate pool
        batch_results = retriever.retrieve_top_k(
            requirements=req_mocks,
            evidence_records=candidate_objects,
            top_k=min(10, len(candidate_objects))
        )
        duration_sec = time.perf_counter() - t0
        total_eval = len(eval_pairs)
        latency_ms = (duration_sec / total_eval) * 1000

        hits_at_1 = 0
        hits_at_3 = 0
        reciprocal_ranks = []
        by_subtype: Dict[str, Dict[str, Any]] = {}

        for idx, record in enumerate(eval_pairs):
            gt_sentence = record["evidence_text"].strip()
            sub_type = record.get("sub_type", "unknown")

            if sub_type not in by_subtype:
                by_subtype[sub_type] = {"total": 0, "hits_at_1": 0, "hits_at_3": 0, "mrr_sum": 0.0}

            by_subtype[sub_type]["total"] += 1

            ret_res = batch_results[idx]
            top_candidates = ret_res.candidates if ret_res else []
            retrieved_sentences = [c.sentence.strip() for c in top_candidates]

            # Find rank of ground truth sentence
            rank = None
            for cand_rank, cand_s in enumerate(retrieved_sentences, start=1):
                if cand_s == gt_sentence:
                    rank = cand_rank
                    break

            if rank == 1:
                hits_at_1 += 1
                by_subtype[sub_type]["hits_at_1"] += 1

            if rank is not None and rank <= 3:
                hits_at_3 += 1
                by_subtype[sub_type]["hits_at_3"] += 1

            rr = (1.0 / rank) if rank is not None else 0.0
            reciprocal_ranks.append(rr)
            by_subtype[sub_type]["mrr_sum"] += rr

        recall_at_1 = hits_at_1 / total_eval
        recall_at_3 = hits_at_3 / total_eval
        mrr = float(np.mean(reciprocal_ranks))

        # Format subtype table
        subtype_metrics = {}
        for st, data in by_subtype.items():
            st_total = data["total"]
            subtype_metrics[st] = {
                "count": st_total,
                "recall_at_1": round(data["hits_at_1"] / st_total, 4) if st_total else 0.0,
                "recall_at_3": round(data["hits_at_3"] / st_total, 4) if st_total else 0.0,
                "mrr": round(data["mrr_sum"] / st_total, 4) if st_total else 0.0,
            }

        results[name] = {
            "total_queries": total_eval,
            "candidate_pool_size": len(candidate_sentences),
            "recall_at_1": round(recall_at_1, 4),
            "recall_at_3": round(recall_at_3, 4),
            "mrr": round(mrr, 4),
            "latency_ms_per_query": round(latency_ms, 2),
            "duration_seconds": round(duration_sec, 2),
            "by_subtype": subtype_metrics
        }

        print(f"Recall@1: {recall_at_1:.4f} ({hits_at_1}/{total_eval})", flush=True)
        print(f"Recall@3: {recall_at_3:.4f} ({hits_at_3}/{total_eval})", flush=True)
        print(f"MRR:      {mrr:.4f}", flush=True)
        print(f"Latency:  {latency_ms:.2f} ms/query (Total: {duration_sec:.2f}s)", flush=True)

    return results


if __name__ == "__main__":
    benchmark_results = run_retrieval_benchmark()
    print("\n" + "="*60, flush=True)
    print("FINAL COMPARISON SUMMARY", flush=True)
    print("="*60, flush=True)
    print(json.dumps(benchmark_results, indent=2), flush=True)
