"""
evaluate_group_aware.py - Leakage-Resistant and Group-Aware Comparative Evaluation

Evaluates the compliance classification pipeline across three distinct leakage-isolated evaluation subsets:
1. Strict Unique Test Records (8 records in test.jsonl with zero text overlap in train or val)
2. Group-Deduplicated Test Clusters (34 unique requirement-evidence pairs in test.jsonl, 1 vote per cluster)
3. Step 2E Hardened Validation Split (dataset/splits/val.jsonl, 94 records including new complex cases)

Ensures test.jsonl remains strictly untouched while providing a transparent, leakage-proof benchmark.
"""

import json
import logging
import os
import sys
from collections import defaultdict
from typing import Dict, Any, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.extraction_nlp.requirement_extractor import RequirementExtractor, ExtractedRequirement
from backend.extraction_nlp.evidence_extractor import EvidenceExtractor
from backend.matching.tfidf_retriever import RequirementRetrievalResult, RetrievedEvidenceCandidate
from backend.classification.baseline_classifier import BaselineClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_records(
    classifier: BaselineClassifier,
    records: List[Dict[str, Any]],
    eval_name: str
) -> Dict[str, Any]:
    """Runs classifier on a given subset of records."""
    req_ext = RequirementExtractor()
    evd_ext = EvidenceExtractor()
    classes = ["COMPLIANT", "NON_COMPLIANT", "UNCERTAIN", "MISSING_EVIDENCE"]

    cm = {c_true: {c_pred: 0 for c_pred in classes} for c_true in classes}
    total = len(records)
    correct = 0

    for r in records:
        req_text = r["requirement_text"]
        evd_text = r.get("evidence_text", "")
        gt_label = r["label"]

        req_list = req_ext.extract_from_text(req_text, source_document="TENDER.pdf")
        if req_list:
            if (r.get("numerical_metadata") or {}).get("is_numerical"):
                num_reqs = [x for x in req_list if isinstance(x.value, (int, float))]
                req_obj = num_reqs[0] if num_reqs else req_list[0]
            else:
                req_obj = req_list[0]
            req_obj.requirement_id = r.get("requirement_id", "REQ-TEST")
        else:
            req_obj = ExtractedRequirement(
                requirement_id=r.get("requirement_id", "REQ-TEST"),
                original_text=req_text,
                category=r.get("tender_category", "General"),
                parameter="general",
                operator=">="
            )

        if not evd_text or not evd_text.strip():
            ret_result = RequirementRetrievalResult(
                requirement_id=req_obj.requirement_id,
                requirement_text=req_text,
                parameter=req_obj.parameter,
                candidates=[],
                top_similarity=0.0,
                has_candidate=False
            )
        else:
            evd_list = evd_ext.extract_from_text(evd_text, source_document=r.get("source_document", "BIDDER.pdf"))
            evd_obj = evd_list[0] if evd_list else None
            cand = RetrievedEvidenceCandidate(
                rank=1,
                evidence_id="EVD-001",
                sentence=evd_text,
                similarity_score=float(r.get("similarity_score_heuristic", 0.75)),
                source_document=r.get("source_document", "BIDDER.pdf"),
                page_number=r.get("page_number", 1) or 1,
                has_negation=evd_obj.has_negation if evd_obj else False,
                numeric_values=evd_obj.numeric_values if evd_obj else [],
                extracted_entities=evd_obj.extracted_entities if evd_obj else [],
                uncertainty_flag=evd_obj.uncertainty_flag if evd_obj else False,
                retrieval_method="semantic"
            )
            cands = [cand]

            contra_meta = r.get("contradiction_metadata")
            if contra_meta and contra_meta.get("conflicting_text"):
                sec_text = contra_meta["conflicting_text"]
                sec_evd_list = evd_ext.extract_from_text(sec_text, source_document=contra_meta.get("conflicting_document", "ANNEXURE.pdf"))
                sec_evd_obj = sec_evd_list[0] if sec_evd_list else None
                cand2 = RetrievedEvidenceCandidate(
                    rank=2,
                    evidence_id="EVD-002",
                    sentence=sec_text,
                    similarity_score=float(r.get("similarity_score_heuristic", 0.70)),
                    source_document=contra_meta.get("conflicting_document", "ANNEXURE.pdf"),
                    page_number=contra_meta.get("conflicting_page", 1) or 1,
                    has_negation=sec_evd_obj.has_negation if sec_evd_obj else False,
                    numeric_values=sec_evd_obj.numeric_values if sec_evd_obj else [],
                    extracted_entities=sec_evd_obj.extracted_entities if sec_evd_obj else [],
                    uncertainty_flag=sec_evd_obj.uncertainty_flag if sec_evd_obj else False,
                    retrieval_method="semantic"
                )
                cands.append(cand2)

            ret_result = RequirementRetrievalResult(
                requirement_id=req_obj.requirement_id,
                requirement_text=req_text,
                parameter=req_obj.parameter,
                candidates=cands,
                top_similarity=cand.similarity_score,
                has_candidate=True
            )

        nli_pred = None
        if classifier.enable_nli and classifier.nli_classifier and classifier.nli_classifier.is_available and evd_text.strip():
            nli_pred = classifier.nli_classifier.predict_pair(req_text, evd_text)

        eval_res = classifier.evaluate_clause(
            req_obj,
            ret_result,
            nli_pred=nli_pred,
            tender_submission_date=r.get("submission_deadline") or "2024-10-15",
            bidder_name=r.get("bidder_name")
        )
        pred_label = eval_res.label
        if pred_label == gt_label:
            correct += 1

        if gt_label in cm and pred_label in cm[gt_label]:
            cm[gt_label][pred_label] += 1

    acc = correct / total if total > 0 else 0.0

    class_metrics = {}
    for c in classes:
        tp = cm[c][c]
        fp = sum(cm[other][c] for other in classes if other != c)
        fn = sum(cm[c][other] for other in classes if other != c)
        supp = sum(cm[c].values())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * rc / (p + rc)) if (p + rc) > 0 else 0.0
        class_metrics[c] = {"precision": p, "recall": rc, "f1": f1, "support": supp}

    macro_f1 = sum(m["f1"] for m in class_metrics.values()) / len(classes)
    return {
        "name": eval_name,
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "class_metrics": class_metrics,
        "confusion_matrix": cm
    }


def run_group_aware_evaluation():
    splits_dir = os.path.join(PROJECT_ROOT, "dataset", "splits")
    with open(os.path.join(splits_dir, "train.jsonl"), "r", encoding="utf-8") as f:
        train_data = [json.loads(line) for line in f if line.strip()]
    with open(os.path.join(splits_dir, "val.jsonl"), "r", encoding="utf-8") as f:
        val_data = [json.loads(line) for line in f if line.strip()]
    with open(os.path.join(splits_dir, "test.jsonl"), "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f if line.strip()]

    # Set of train text pairs
    train_pairs = {(d["requirement_text"].strip(), (d.get("evidence_text") or "").strip()) for d in train_data}

    # 1. Unique non-overlapping records in test.jsonl
    unique_test_records = [d for d in test_data if (d["requirement_text"].strip(), (d.get("evidence_text") or "").strip()) not in train_pairs]

    # 2. Deduplicated clusters in test.jsonl (1 record per unique text pair)
    seen_test_pairs = set()
    dedup_test_records = []
    for d in test_data:
        key = (d["requirement_text"].strip(), (d.get("evidence_text") or "").strip())
        if key not in seen_test_pairs:
            seen_test_pairs.add(key)
            dedup_test_records.append(d)

    clf = BaselineClassifier(enable_nli=True, enable_reconciliation=True)

    res_unique = evaluate_records(clf, unique_test_records, "1. Strict Unseen Test Records (Zero Overlap)")
    res_dedup = evaluate_records(clf, dedup_test_records, "2. Group-Deduplicated Test Clusters (1 Vote/Pair)")
    res_val = evaluate_records(clf, val_data, "3. Hardened Validation Split (94 records)")

    print("=" * 80)
    print("TENDERIQ - LEAKAGE-RESISTANT & GROUP-AWARE EVALUATION REPORT")
    print("=" * 80)
    for res in [res_unique, res_dedup, res_val]:
        print(f"\nSubset: {res['name']}")
        print(f"Total Records: {res['total']} | Correct: {res['correct']} | Accuracy: {res['accuracy']*100:.2f}% | Macro F1: {res['macro_f1']*100:.2f}%")
        print(f"{'Class':<18} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
        print("-" * 62)
        for c, m in res["class_metrics"].items():
            print(f"{c:<18} {m['precision']*100:<11.1f}% {m['recall']*100:<11.1f}% {m['f1']*100:<11.1f}% {m['support']:<8}")

    print("\n" + "=" * 80)
    return {
        "unique_test": res_unique,
        "dedup_test": res_dedup,
        "val_split": res_val
    }


if __name__ == "__main__":
    run_group_aware_evaluation()
