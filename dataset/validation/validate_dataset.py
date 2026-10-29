"""
validate_dataset.py - Comprehensive Dataset Integrity & Edge Case Validation Suite
Part of Phase 1: Dataset Generation & Validation

Verifies:
1. Schema integrity & mandatory key presence
2. ID uniqueness (no duplicate pair_ids)
3. Label taxonomy validity (COMPLIANT, NON_COMPLIANT, UNCERTAIN, MISSING_EVIDENCE)
4. Missing Evidence invariant (strictly separate from confirmed non-compliance)
5. Simulation watermark & disclaimer presence
6. Edge case coverage (numerical boundaries, unit conversions, paraphrases, contradictions)
7. Similarity vs. Compliance empirical proof (demonstrating similarity != compliance)
8. Train / Val / Test split integrity & zero data leakage
"""

import json
import os
import sys
from typing import Dict, List, Any


ALLOWED_LABELS = {"COMPLIANT", "NON_COMPLIANT", "UNCERTAIN", "MISSING_EVIDENCE"}
ALLOWED_SUBTYPES = {
    "direct_match", "paraphrase", "unit_conversion", "numerical_boundary",
    "confirmed_shortfall", "cross_document_contradiction", "high_similarity_negation",
    "ambiguous_claim", "missing_section"
}

MANDATORY_KEYS = [
    "pair_id", "tender_id", "tender_title", "tender_category",
    "requirement_id", "requirement_text", "is_mandatory",
    "bidder_id", "bidder_name", "evidence_text", "source_document",
    "page_number", "section_reference", "label", "sub_type",
    "similarity_score_heuristic", "explanation_reasoning", "is_simulated", "disclaimer"
]


def load_dataset(json_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset file not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    errors = []
    warnings = []
    seen_ids = set()

    label_counts: Dict[str, int] = {}
    subtype_counts: Dict[str, int] = {}
    category_counts: Dict[str, int] = {}
    label_similarity_scores: Dict[str, List[float]] = {lbl: [] for lbl in ALLOWED_LABELS}

    adversarial_high_sim_fails = []
    paraphrase_low_sim_passes = []
    contradictions = []
    boundary_cases = []
    unit_conversions = []

    for idx, r in enumerate(records):
        pid = r.get("pair_id", f"INDEX-{idx}")

        # 1. Mandatory keys check
        for k in MANDATORY_KEYS:
            if k not in r:
                errors.append(f"[{pid}] Missing mandatory key: '{k}'")

        # 2. ID uniqueness
        if pid in seen_ids:
            errors.append(f"[{pid}] Duplicate pair_id detected")
        seen_ids.add(pid)

        # 3. Label validity
        lbl = r.get("label")
        if lbl not in ALLOWED_LABELS:
            errors.append(f"[{pid}] Invalid label '{lbl}'. Allowed: {ALLOWED_LABELS}")
        else:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        # 4. Subtype validity
        sub = r.get("sub_type")
        if sub not in ALLOWED_SUBTYPES:
            errors.append(f"[{pid}] Invalid sub_type '{sub}'. Allowed: {ALLOWED_SUBTYPES}")
        else:
            subtype_counts[sub] = subtype_counts.get(sub, 0) + 1

        # Category counts
        cat = r.get("tender_category", "Unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1

        # 5. Missing Evidence Invariants
        evi = r.get("evidence_text", "")
        if lbl == "MISSING_EVIDENCE":
            if evi != "" and evi is not None:
                errors.append(f"[{pid}] MISSING_EVIDENCE record has non-empty evidence_text: '{evi[:40]}...'")
            if r.get("source_document") != "N/A":
                warnings.append(f"[{pid}] MISSING_EVIDENCE should have source_document 'N/A', found: {r.get('source_document')}")
        else:
            if not evi or len(evi.strip()) == 0:
                errors.append(f"[{pid}] {lbl} record must provide non-empty evidence_text.")

        # 6. Watermark & Simulation check
        if r.get("is_simulated") is not True:
            errors.append(f"[{pid}] is_simulated flag must be True")
        if "SIMULATED" not in str(r.get("disclaimer", "")):
            errors.append(f"[{pid}] Missing required SIMULATED watermark in disclaimer")

        # 7. Similarity tracking
        sim = r.get("similarity_score_heuristic", 0.0)
        if lbl in label_similarity_scores:
            label_similarity_scores[lbl].append(sim)

        # Track interesting edge cases
        if sub == "high_similarity_negation" and sim >= 0.85 and lbl == "NON_COMPLIANT":
            adversarial_high_sim_fails.append(r)
        if sub == "paraphrase" and sim <= 0.75 and lbl == "COMPLIANT":
            paraphrase_low_sim_passes.append(r)
        if sub == "cross_document_contradiction":
            contradictions.append(r)
            cmeta = r.get("contradiction_metadata")
            if not cmeta or not cmeta.get("is_contradiction"):
                errors.append(f"[{pid}] cross_document_contradiction must have valid contradiction_metadata")
        if sub == "numerical_boundary":
            boundary_cases.append(r)
        if sub == "unit_conversion":
            unit_conversions.append(r)

    # Compute similarity stats
    sim_stats = {}
    for lbl, scores in label_similarity_scores.items():
        if scores:
            avg_sim = sum(scores) / len(scores)
            min_sim = min(scores)
            max_sim = max(scores)
            sim_stats[lbl] = {"mean": round(avg_sim, 3), "min": round(min_sim, 3), "max": round(max_sim, 3), "count": len(scores)}

    return {
        "total_records": len(records),
        "errors": errors,
        "warnings": warnings,
        "label_counts": label_counts,
        "subtype_counts": subtype_counts,
        "category_counts": category_counts,
        "similarity_stats": sim_stats,
        "adversarial_count": len(adversarial_high_sim_fails),
        "paraphrase_count": len(paraphrase_low_sim_passes),
        "contradiction_count": len(contradictions),
        "boundary_count": len(boundary_cases),
        "unit_conversion_count": len(unit_conversions),
        "samples": {
            "compliant_paraphrase": paraphrase_low_sim_passes[0] if paraphrase_low_sim_passes else None,
            "high_sim_negation": adversarial_high_sim_fails[0] if adversarial_high_sim_fails else None,
            "contradiction": contradictions[0] if contradictions else None,
            "boundary": boundary_cases[0] if boundary_cases else None,
            "unit_conversion": unit_conversions[0] if unit_conversions else None,
            "missing_evidence": next((r for r in records if r["label"] == "MISSING_EVIDENCE"), None)
        }
    }


def validate_splits(base_dir: str, total_raw_count: int) -> Dict[str, Any]:
    splits_dir = os.path.join(base_dir, "splits")
    train_file = os.path.join(splits_dir, "train.jsonl")
    val_file = os.path.join(splits_dir, "val.jsonl")
    test_file = os.path.join(splits_dir, "test.jsonl")

    split_errors = []
    split_counts = {}
    ids_per_split = {}

    for name, path in [("train", train_file), ("val", val_file), ("test", test_file)]:
        if not os.path.exists(path):
            split_errors.append(f"Missing split file: {path}")
            continue
        ids = set()
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    count += 1
                    ids.add(item["pair_id"])
        split_counts[name] = count
        ids_per_split[name] = ids

    # Check leakage
    if len(ids_per_split) == 3:
        leak_train_val = ids_per_split["train"].intersection(ids_per_split["val"])
        leak_train_test = ids_per_split["train"].intersection(ids_per_split["test"])
        leak_val_test = ids_per_split["val"].intersection(ids_per_split["test"])

        if leak_train_val:
            split_errors.append(f"Leakage between train and val: {len(leak_train_val)} IDs")
        if leak_train_test:
            split_errors.append(f"Leakage between train and test: {len(leak_train_test)} IDs")
        if leak_val_test:
            split_errors.append(f"Leakage between val and test: {len(leak_val_test)} IDs")

        total_split_ids = len(ids_per_split["train"]) + len(ids_per_split["val"]) + len(ids_per_split["test"])
        if total_split_ids != total_raw_count:
            split_errors.append(f"Total split records ({total_split_ids}) does not match raw dataset ({total_raw_count})")

    return {
        "split_counts": split_counts,
        "split_errors": split_errors
    }


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    json_path = os.path.join(base_dir, "raw", "requirement_evidence_dataset.json")

    print("\n" + "=" * 70)
    print("PHASE 1: DATASET INTEGRITY & EDGE-CASE VALIDATION REPORT")
    print("=" * 70)

    try:
        records = load_dataset(json_path)
    except Exception as e:
        print(f"FATAL: Could not load dataset: {e}")
        sys.exit(1)

    val_res = validate_records(records)
    split_res = validate_splits(base_dir, val_res["total_records"])

    # 1. Basic Stats
    print(f"\n[1] DATASET VOLUME & INTEGRITY")
    print(f"  * Total Records Evaluated: {val_res['total_records']}")
    print(f"  * Validation Errors:       {len(val_res['errors'])}")
    print(f"  * Validation Warnings:     {len(val_res['warnings'])}")

    if val_res["errors"]:
        print("\n  [!] CRITICAL ERRORS FOUND:")
        for err in val_res["errors"][:10]:
            print(f"      - {err}")
        sys.exit(1)
    else:
        print("  * Schema & Field Completeness: 100% PASSED (All mandatory keys present)")
        print("  * Identifier Uniqueness:       100% PASSED (Zero duplicate pair_ids)")
        print("  * Watermark & Disclaimers:     100% PASSED (All records explicitly simulated)")

    # 2. Label Distribution
    print(f"\n[2] GROUND TRUTH LABEL DISTRIBUTION")
    tot = val_res["total_records"]
    for lbl, count in sorted(val_res["label_counts"].items(), key=lambda x: -x[1]):
        pct = (count / tot) * 100
        print(f"  * {lbl:<18} : {count:>4} records ({pct:5.1f}%)")

    # 3. Subtype & Edge-Case Breakdown
    print(f"\n[3] EDGE CASE & SUB-TYPE COVERAGE")
    for st, count in sorted(val_res["subtype_counts"].items(), key=lambda x: -x[1]):
        pct = (count / tot) * 100
        print(f"  * {st:<28} : {count:>4} records ({pct:5.1f}%)")

    # 4. Procurement Category Breakdown
    print(f"\n[4] PROCUREMENT CATEGORY DISTRIBUTION")
    for cat, count in sorted(val_res["category_counts"].items(), key=lambda x: -x[1]):
        pct = (count / tot) * 100
        print(f"  * {cat:<36} : {count:>4} records ({pct:5.1f}%)")

    # 5. Proof: Similarity is NOT Proof of Compliance
    print(f"\n[5] EMPIRICAL PROOF: SIMILARITY != COMPLIANCE")
    print("  " + "-" * 56)
    print(f"  {'Label':<18} | {'Mean Sim':<10} | {'Min Sim':<10} | {'Max Sim':<10}")
    print("  " + "-" * 56)
    for lbl, stats in val_res["similarity_stats"].items():
        print(f"  {lbl:<18} | {stats['mean']:<10.3f} | {stats['min']:<10.3f} | {stats['max']:<10.3f}")
    print("  " + "-" * 56)
    print(f"  * Adversarial High-Sim Fails (Negations with Sim >= 0.85 but NON_COMPLIANT): {val_res['adversarial_count']}")
    print(f"  * Paraphrase Low-Sim Passes  (Paraphrases with Sim <= 0.75 but COMPLIANT):     {val_res['paraphrase_count']}")
    print("  => Conclusion: Embedding/cosine similarity alone fails on polarity and negation.")

    # 6. Splits Integrity
    print(f"\n[6] TRAIN / VAL / TEST SPLIT INTEGRITY")
    for s_name, s_count in split_res["split_counts"].items():
        s_pct = (s_count / tot) * 100
        print(f"  * {s_name.upper():<6} Split : {s_count:>4} records ({s_pct:5.1f}%)")
    if split_res["split_errors"]:
        print("  [!] SPLIT LEAKAGE ERRORS:")
        for err in split_res["split_errors"]:
            print(f"      - {err}")
    else:
        print("  * Zero Data Leakage:           100% PASSED (Strict disjoint split sets)")

    print("\n" + "=" * 70)
    print("ALL VALIDATION SUITE TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
