"""
inspect_samples.py - Utility to extract and display representative examples
Part of Phase 1: Dataset Generation & Validation
"""

import json
import os

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    json_path = os.path.join(base_dir, "raw", "requirement_evidence_dataset.json")

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    targets = [
        ("COMPLIANT", "direct_match", "Direct Match (Fulfillment)"),
        ("COMPLIANT", "paraphrase", "Paraphrase (Semantic Entailment)"),
        ("COMPLIANT", "unit_conversion", "Unit Conversion (Months -> Years / Gbps -> Mbps)"),
        ("COMPLIANT", "numerical_boundary", "Numerical Boundary Pass (Value == Threshold)"),
        ("NON_COMPLIANT", "numerical_boundary", "Numerical Boundary Fail (Value < Threshold)"),
        ("NON_COMPLIANT", "confirmed_shortfall", "Confirmed Shortfall (Direct Quantitative/Qualitative Fail)"),
        ("NON_COMPLIANT", "high_similarity_negation", "High Similarity Negation (Adversarial: Sim >= 0.88, Label = NON_COMPLIANT)"),
        ("NON_COMPLIANT", "cross_document_contradiction", "Cross-Document Contradiction (Proposal vs Annexure)"),
        ("UNCERTAIN", "ambiguous_claim", "Uncertain / Ambiguous Claim (Vague / Subjective assertion)"),
        ("MISSING_EVIDENCE", "missing_section", "Missing Evidence (Omitted Mandatory Clause)")
    ]

    print("\n" + "=" * 80)
    print("REPRESENTATIVE DATASET EXAMPLES ACROSS ALL EDGE CASES")
    print("=" * 80)

    for target_label, target_sub, title in targets:
        match = None
        for r in records:
            if r["label"] == target_label and r["sub_type"] == target_sub:
                match = r
                break

        print(f"\n>>> [{title.upper()}]")
        if match:
            print(f"  * Pair ID:           {match['pair_id']}")
            print(f"  * Tender:            {match['tender_title']} ({match['tender_id']})")
            print(f"  * Category:          {match['tender_category']}")
            print(f"  * Requirement:       \"{match['requirement_text']}\"")
            print(f"  * Ground Truth:      {match['label']} (sub_type: {match['sub_type']})")
            print(f"  * Evidence Extracted:\"{match['evidence_text']}\"")
            print(f"  * Source Reference:  {match['source_document']}, Page {match['page_number']}, {match['section_reference']}")
            print(f"  * Similarity Score:  {match['similarity_score_heuristic']} (Simulated Cosine Similarity)")
            if match.get("contradiction_metadata"):
                print(f"  * Contradiction:     {match['contradiction_metadata']}")
            if match.get("numerical_metadata"):
                print(f"  * Numerical Meta:    {match['numerical_metadata']}")
            print(f"  * Explanation:       {match['explanation_reasoning']}")
        else:
            print("  [!] No matching record found.")
        print("-" * 80)

if __name__ == "__main__":
    main()
