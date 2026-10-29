"""
evaluate_classifier.py - Comparative Evaluation Benchmark for Procurement Compliance Classifiers
Phases 4, 6 & 7

Compares:
1. Phase 4 Baseline Classifier (Deterministic Rule-Based)
2. Phase 6 Augmented Classifier (Sentence-BERT / TF-IDF + Cross-Encoder NLI + Deterministic Guards)
3. Phase 7 Augmented Classifier (Phase 6 + Cross-Document Evidence Reconciliation)

Evaluates on the labelled synthetic test dataset (dataset/splits/test.jsonl).
Reports:
- Precision, Recall, F1 per class (COMPLIANT, NON_COMPLIANT, UNCERTAIN, MISSING_EVIDENCE)
- Macro-averaged and Weighted-averaged metrics
- 4x4 Confusion Matrices for all three pipelines
- Subtype-level accuracy breakdown (numerical_boundary, unit_conversion, high_similarity_negation, etc.)
- Phase 7 cross-document reconciliation accuracy on cross_document_contradiction subtype
- Transparent reporting without synthetic data leakage or fabricated scores.
"""

import json
import logging
import os
import sys
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional

# Ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.extraction_nlp.requirement_extractor import RequirementExtractor, ExtractedRequirement
from backend.extraction_nlp.evidence_extractor import EvidenceExtractor, ExtractedEvidence
from backend.matching.tfidf_retriever import RequirementRetrievalResult, RetrievedEvidenceCandidate
from backend.classification.baseline_classifier import BaselineClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TEST_SPLIT = os.path.join(PROJECT_ROOT, "dataset", "splits", "test.jsonl")


def evaluate_pipeline_on_split(
    classifier: BaselineClassifier,
    records: List[Dict[str, Any]],
    pipeline_name: str
) -> Dict[str, Any]:
    """Runs a compliance classifier against labelled ground-truth test pairs.

    Args:
        classifier: BaselineClassifier configured with or without NLI.
        records: List of test pair dictionaries from test.jsonl.
        pipeline_name: Descriptive name for reporting.

    Returns:
        Structured evaluation metrics dictionary.
    """
    req_ext = RequirementExtractor()
    evd_ext = EvidenceExtractor()

    classes = ["COMPLIANT", "NON_COMPLIANT", "UNCERTAIN", "MISSING_EVIDENCE"]
    y_true: List[str] = []
    y_pred: List[str] = []
    by_subtype: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    mismatches: List[Dict[str, Any]] = []

    for r in records:
        req_text = r["requirement_text"]
        evd_text = r.get("evidence_text", "")
        gt_label = r["label"]
        subtype = r.get("sub_type", "unspecified")

        # 1. Extract requirement structure
        req_list = req_ext.extract_from_text(req_text, source_document="TENDER.pdf")
        if req_list:
            req_obj = req_list[0]
            req_obj.requirement_id = r.get("requirement_id", "REQ-TEST")
        else:
            req_obj = ExtractedRequirement(
                requirement_id=r.get("requirement_id", "REQ-TEST"),
                original_text=req_text,
                category=r.get("tender_category", "General Technical"),
                parameter="general",
                operator=">="
            )

        # 2. Extract evidence candidate structure
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
            evd_list = evd_ext.extract_from_text(
                evd_text,
                source_document=r.get("source_document", "BIDDER.pdf")
            )
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

            # In multi-document scenarios, include evidence retrieved from secondary/annexure documents
            contra_meta = r.get("contradiction_metadata")
            if contra_meta and contra_meta.get("conflicting_text"):
                sec_text = contra_meta["conflicting_text"]
                sec_doc = contra_meta.get("conflicting_document", "ANNEXURE.pdf")
                sec_page = contra_meta.get("conflicting_page", 1) or 1
                sec_evd_list = evd_ext.extract_from_text(sec_text, source_document=sec_doc)
                sec_evd_obj = sec_evd_list[0] if sec_evd_list else None
                cand2 = RetrievedEvidenceCandidate(
                    rank=2,
                    evidence_id="EVD-002",
                    sentence=sec_text,
                    similarity_score=float(r.get("similarity_score_heuristic", 0.70)),
                    source_document=sec_doc,
                    page_number=sec_page,
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

        # 3. Optional NLI inference
        nli_pred = None
        if classifier.enable_nli and classifier.nli_classifier and classifier.nli_classifier.is_available and evd_text.strip():
            nli_pred = classifier.nli_classifier.predict_pair(req_text, evd_text)

        # 4. Evaluate clause compliance
        eval_res = classifier.evaluate_clause(req_obj, ret_result, nli_pred=nli_pred)
        pred_label = eval_res.label

        y_true.append(gt_label)
        y_pred.append(pred_label)

        by_subtype[subtype]["total"] += 1
        if pred_label == gt_label:
            by_subtype[subtype]["correct"] += 1
        else:
            mismatches.append({
                "pair_id": r.get("pair_id", "N/A"),
                "subtype": subtype,
                "gt": gt_label,
                "pred": pred_label,
                "requirement": req_text,
                "evidence": evd_text,
                "reasoning": eval_res.reasoning,
                "nli_relation": getattr(nli_pred, "relation", None)
            })

    # 5. Compute confusion matrix: cm[actual][predicted]
    cm = {c_true: {c_pred: 0 for c_pred in classes} for c_true in classes}
    for t, p in zip(y_true, y_pred):
        if t in cm and p in cm[t]:
            cm[t][p] += 1

    # 6. Compute per-class precision, recall, and F1
    total_samples = len(y_true)
    total_correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    class_metrics = {}
    weighted_p = 0.0
    weighted_r = 0.0
    weighted_f1 = 0.0

    for c in classes:
        tp = cm[c][c]
        fp = sum(cm[other][c] for other in classes if other != c)
        fn = sum(cm[c][other] for other in classes if other != c)
        support = sum(cm[c].values())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        class_metrics[c] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support
        }

        weighted_p += prec * support
        weighted_r += rec * support
        weighted_f1 += f1 * support

    macro_precision = sum(m["precision"] for m in class_metrics.values()) / len(classes)
    macro_recall = sum(m["recall"] for m in class_metrics.values()) / len(classes)
    macro_f1 = sum(m["f1"] for m in class_metrics.values()) / len(classes)

    weighted_precision = weighted_p / total_samples if total_samples > 0 else 0.0
    weighted_recall = weighted_r / total_samples if total_samples > 0 else 0.0
    weighted_f1 = weighted_f1 / total_samples if total_samples > 0 else 0.0

    return {
        "pipeline_name": pipeline_name,
        "total_samples": total_samples,
        "accuracy": overall_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "class_metrics": class_metrics,
        "confusion_matrix": cm,
        "by_subtype": dict(by_subtype),
        "mismatches": mismatches
    }


def evaluate_cross_document_reconciliation(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluates Phase 7 cross-document reconciler on test records with contradiction_metadata.

    For each record with contradiction_metadata, runs reconcile_from_metadata() using the
    primary evidence text and the conflicting document metadata provided in the dataset.
    Compares predicted label against ground-truth label.
    """
    from backend.reconciliation.cross_document_reconciler import CrossDocumentReconciler
    from backend.extraction_nlp.requirement_extractor import RequirementExtractor

    req_ext = RequirementExtractor()
    reconciler = CrossDocumentReconciler()

    cross_doc_records = [
        r for r in records
        if r.get("sub_type") == "cross_document_contradiction"
        and r.get("contradiction_metadata") is not None
    ]

    if not cross_doc_records:
        return {"total": 0, "correct": 0, "accuracy": None, "details": []}

    results = []
    correct = 0

    for r in cross_doc_records:
        req_text = r["requirement_text"]
        gt_label = r["label"]
        meta = r["contradiction_metadata"]

        # Extract numeric requirement value if available
        req_list = req_ext.extract_from_text(req_text, source_document="TENDER.pdf")
        req_obj = req_list[0] if req_list else None
        req_val = getattr(req_obj, "value", None)
        req_unit = getattr(req_obj, "unit", None)
        req_val_float = float(req_val) if isinstance(req_val, (int, float)) else None

        # Run reconciler with both evidence sources
        rec_result = reconciler.reconcile_from_metadata(
            requirement_id=r.get("requirement_id", "REQ-TEST"),
            requirement_text=req_text,
            requirement_value=req_val_float,
            requirement_unit=req_unit,
            primary_evidence_text=r.get("evidence_text", ""),
            primary_source_document=r.get("source_document", "BIDDER-PROPOSAL.pdf"),
            primary_page=r.get("page_number", 1),
            conflicting_document=meta.get("conflicting_document", "ANNEXURE.pdf"),
            conflicting_page=meta.get("conflicting_page", 1),
            conflicting_text=meta.get("conflicting_text", ""),
        )

        pred_label = rec_result.reconciled_label or "UNCERTAIN"
        is_correct = (pred_label == gt_label)
        if is_correct:
            correct += 1

        results.append({
            "pair_id": r.get("pair_id", "N/A"),
            "requirement": req_text[:100],
            "gt_label": gt_label,
            "pred_label": pred_label,
            "has_conflict": rec_result.has_conflict,
            "conflict_severity": rec_result.conflict_severity,
            "conflict_type": rec_result.conflicts[0].conflict_type if rec_result.conflicts else "none",
            "reconciled_reasoning": rec_result.reconciled_reasoning[:200],
            "correct": is_correct,
        })

    accuracy = correct / len(cross_doc_records) if cross_doc_records else 0.0
    return {
        "total": len(cross_doc_records),
        "correct": correct,
        "accuracy": accuracy,
        "details": results
    }


def run_benchmark(test_split_path: str = DEFAULT_TEST_SPLIT) -> Dict[str, Any]:
    """Runs a side-by-side comparative benchmark between Phase 4, Phase 6, and Phase 7."""
    if not os.path.exists(test_split_path):
        raise FileNotFoundError(f"Test split dataset not found: {test_split_path}")

    with open(test_split_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    logger.info("Loaded %d labelled records from %s", len(records), test_split_path)

    # Initialize Phase 4 baseline classifier (NLI disabled, reconciliation disabled)
    clf_phase4 = BaselineClassifier(enable_nli=False, enable_reconciliation=False)
    report_p4 = evaluate_pipeline_on_split(clf_phase4, records, "Phase 4 Deterministic Baseline")

    # Initialize Phase 6 classifier (NLI enabled, reconciliation disabled)
    clf_phase6 = BaselineClassifier(enable_nli=True, enable_reconciliation=False)
    report_p6 = evaluate_pipeline_on_split(clf_phase6, records, "Phase 6 NLI-Integrated Pipeline")

    # Initialize Phase 7 classifier (NLI enabled, reconciliation enabled)
    clf_phase7 = BaselineClassifier(enable_nli=True, enable_reconciliation=True)
    report_p7 = evaluate_pipeline_on_split(clf_phase7, records, "Phase 7 Cross-Doc Reconciliation Pipeline")

    # Run dedicated cross-document reconciliation evaluation
    cross_doc_results = evaluate_cross_document_reconciliation(records)

    return {
        "dataset_path": test_split_path,
        "total_records": len(records),
        "phase_4_baseline": report_p4,
        "phase_6_nli": report_p6,
        "phase_7_reconciliation": report_p7,
        "cross_doc_evaluation": cross_doc_results,
    }


def print_formatted_report(benchmark_results: Dict[str, Any]):
    """Prints a clean ASCII report suitable for audit documentation."""
    p4 = benchmark_results["phase_4_baseline"]
    p6 = benchmark_results["phase_6_nli"]
    p7 = benchmark_results.get("phase_7_reconciliation", p6)
    cross_doc = benchmark_results.get("cross_doc_evaluation", {})
    total = benchmark_results["total_records"]

    print("=" * 80)
    print("AI-POWERED BID COMPLIANCE PLATFORM (GeM PROCURE-NLP)")
    print("PHASE 7 CROSS-DOCUMENT RECONCILIATION - COMPARATIVE EVALUATION AUDIT REPORT")
    print("=" * 80)
    print(f"Dataset Split:       {benchmark_results['dataset_path']}")
    print(f"Total Test Records:  {total}")
    print(f"Ground Truth Labels: COMPLIANT, NON_COMPLIANT, UNCERTAIN, MISSING_EVIDENCE")
    print("-" * 80)

    print("\n1. OVERALL METRIC COMPARISON (Phase 4 vs Phase 6 vs Phase 7)")
    print(f"{'Metric':<25} {'Phase 4 Baseline':<20} {'Phase 6 (NLI)':<20} {'Phase 7 (Reconcil.)':<22} {'P7 vs P4 Delta':<15}")
    print("-" * 102)
    metrics = [
        ("Accuracy", "accuracy"),
        ("Macro Precision", "macro_precision"),
        ("Macro Recall", "macro_recall"),
        ("Macro F1-Score", "macro_f1"),
        ("Weighted F1-Score", "weighted_f1"),
    ]
    for label, key in metrics:
        v4 = p4[key] * 100
        v6 = p6[key] * 100
        v7 = p7[key] * 100
        print(f"{label:<25} {v4:.2f}%{'':<13} {v6:.2f}%{'':<13} {v7:.2f}%{'':<15} {v7 - v4:+.2f}%")

    print("\n2. PER-CLASS PERFORMANCE (PHASE 7)")
    print(f"{'Class':<20} {'Precision':<14} {'Recall':<14} {'F1-Score':<14} {'Support':<8}")
    print("-" * 70)
    for c, m in p7["class_metrics"].items():
        print(f"{c:<20} {m['precision']*100:.1f}%{'':<8} {m['recall']*100:.1f}%{'':<8} {m['f1']*100:.1f}%{'':<8} {m['support']}")

    print("\n3. 4x4 CONFUSION MATRIX (PHASE 7)")
    print(r"Actual \ Predicted" + f"{'':<4} {'COMPLIANT':<14} {'NON_COMPLIANT':<16} {'UNCERTAIN':<14} {'MISSING_EVIDENCE':<18}")
    print("-" * 82)
    classes = ["COMPLIANT", "NON_COMPLIANT", "UNCERTAIN", "MISSING_EVIDENCE"]
    for r_c in classes:
        row_str = f"{r_c:<20} "
        for p_c in classes:
            w = 16 if p_c == "NON_COMPLIANT" else 14
            row_str += f"{p7['confusion_matrix'][r_c][p_c]:<{w}} "
        print(row_str)

    print("\n4. SUBTYPE BREAKDOWN ACCURACY (PHASE 4 vs PHASE 6 vs PHASE 7)")
    print(f"{'Subtype':<32} {'Count':<8} {'Phase 4':<12} {'Phase 6':<12} {'Phase 7':<12} {'P7 Delta':<10}")
    print("-" * 86)
    for st in sorted(p7["by_subtype"].keys()):
        tot = p7["by_subtype"][st]["total"]
        c4 = p4["by_subtype"].get(st, {}).get("correct", 0)
        c6 = p6["by_subtype"].get(st, {}).get("correct", 0)
        c7 = p7["by_subtype"][st]["correct"]
        acc4 = (c4 / tot) * 100 if tot > 0 else 0.0
        acc6 = (c6 / tot) * 100 if tot > 0 else 0.0
        acc7 = (c7 / tot) * 100 if tot > 0 else 0.0
        print(f"{st:<32} {tot:<8} {acc4:.1f}%{'':<6} {acc6:.1f}%{'':<6} {acc7:.1f}%{'':<6} {acc7 - acc4:+.1f}%")

    print("\n5. PHASE 7 CROSS-DOCUMENT RECONCILIATION EVALUATION")
    print("-" * 80)
    if cross_doc.get("total", 0) > 0:
        print(f"cross_document_contradiction subset: {cross_doc['total']} records")
        print(f"Phase 6 (no reconciler):  {p6['by_subtype'].get('cross_document_contradiction', {}).get('correct', 0)}/{cross_doc['total']} correct"
              f" = {p6['by_subtype'].get('cross_document_contradiction', {}).get('correct', 0) / cross_doc['total'] * 100:.1f}%")
        print(f"Phase 7 (reconciler):     {cross_doc['correct']}/{cross_doc['total']} correct"
              f" = {cross_doc['accuracy'] * 100:.1f}%")
        print()
        for det in cross_doc["details"]:
            status = "[PASS]" if det["correct"] else "[FAIL]"
            print(f"  {status} {det['pair_id']}: GT={det['gt_label']}, Pred={det['pred_label']}, conflict={det['conflict_type']}")
            print(f"    Req: {det['requirement'][:90]}")
            print(f"    Reasoning: {det['reconciled_reasoning'][:120]}")
    else:
        print("  No cross_document_contradiction records in test split.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    results = run_benchmark()
    print_formatted_report(results)
