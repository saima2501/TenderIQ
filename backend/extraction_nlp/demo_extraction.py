"""
demo_extraction.py - Command-Line Demo for Phase 3 Requirement & Evidence Extraction
Part of Phase 3: Requirement & Evidence Extraction

Runs the extraction pipeline on representative tender and bidder PDFs and prints formatted,
structured records illustrating:
- Tender requirements extraction (modality, operators, numeric thresholds, parameters)
- Bidder evidence extraction (named entities, numeric values, polarity/negation, uncertainty)
"""

import os
import sys
import json
from pprint import pprint

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.extraction_nlp.requirement_extractor import extract_requirements_from_pdf
from backend.extraction_nlp.evidence_extractor import extract_evidence_from_pdf


def run_demo():
    print("=" * 80)
    print("PHASE 3 NLP EXTRACTION PIPELINE DEMO: TENDER REQUIREMENTS & BIDDER EVIDENCE")
    print("=" * 80)

    # 1. Tender Requirements Extraction Demo
    tender_pdf = os.path.join(PROJECT_ROOT, "dataset", "synthetic_pdfs", "tenders", "TENDER-2024-IT-001.pdf")
    print(f"\n[1] EXTRACTING REQUIREMENTS FROM: {os.path.basename(tender_pdf)}")
    req_report = extract_requirements_from_pdf(tender_pdf)

    print(f"Total Pages Analyzed: {req_report.total_pages}")
    print(f"Total Requirements Extracted: {req_report.total_requirements}")
    print(f"Mandatory Requirements: {req_report.mandatory_count}")
    print(f"Optional / Preferred: {req_report.optional_count}")
    print(f"Marked for Human Review (Uncertain/Incomplete): {req_report.uncertain_count}")
    print(f"NLP Engine Used: {req_report.nlp_engine}\n")

    print("Representative Structured Requirements:")
    print("-" * 80)
    for idx, req in enumerate(req_report.requirements[:5], 1):
        print(f"Requirement #{idx}:")
        print(f"  * ID:               {req.requirement_id}")
        print(f"  * Text:             {req.original_text[:110]}...")
        print(f"  * Category:         {req.category}")
        print(f"  * Parameter:        {req.parameter}")
        print(f"  * Operator:         {req.operator}")
        print(f"  * Value & Unit:     {req.value} {req.unit or ''}")
        print(f"  * Mandatory:        {req.is_mandatory}")
        print(f"  * Source & Page:    {req.source_document} (Page {req.page_number})")
        print(f"  * Section:          {req.section_reference}")
        print(f"  * Uncertainty Flag: {req.uncertainty_flag}")
        print(f"  * Review Notes:     {req.review_notes}")
        print()

    # 2. Bidder Evidence Extraction Demo
    bidder_pdf = os.path.join(PROJECT_ROOT, "dataset", "synthetic_pdfs", "bidders", "BIDDER-ALPHA-TECH-PROPOSAL.pdf")
    print("=" * 80)
    print(f"[2] EXTRACTING EVIDENCE CLAIMS FROM: {os.path.basename(bidder_pdf)}")
    evd_report = extract_evidence_from_pdf(bidder_pdf)

    print(f"Bidder Identified: {evd_report.bidder_name}")
    print(f"Total Pages Analyzed: {evd_report.total_pages}")
    print(f"Total Evidence Records: {evd_report.total_evidence_records}")
    print(f"Records with Extracted Metrics: {evd_report.records_with_numeric_values}")
    print(f"Records with Detected Entities: {evd_report.records_with_entities}")
    print(f"Records with Negation Markers: {evd_report.records_with_negation}")
    print(f"Records Flagged for Review: {evd_report.uncertain_records_count}")
    print(f"NLP Engine Used: {evd_report.nlp_engine}\n")

    print("Representative Structured Evidence Records:")
    print("-" * 80)
    for idx, evd in enumerate(evd_report.evidence_records[:5], 1):
        print(f"Evidence #{idx}:")
        print(f"  * ID:               {evd.evidence_id}")
        print(f"  * Sentence:         {evd.original_sentence[:110]}...")
        print(f"  * Entities:         {evd.extracted_entities}")
        print(f"  * Numeric Metrics:  {evd.numeric_values}")
        print(f"  * Negation Found:   {evd.has_negation}")
        print(f"  * Source & Page:    {evd.source_document} (Page {evd.page_number})")
        print(f"  * Section:          {evd.section_reference}")
        print(f"  * Uncertainty Flag: {evd.uncertainty_flag}")
        print(f"  * Review Notes:     {evd.review_notes}")
        print()

    # 3. Demonstration of Negation Detection (Epsilon Proposal)
    epsilon_pdf = os.path.join(PROJECT_ROOT, "dataset", "synthetic_pdfs", "bidders", "BIDDER-EPSILON-INFRA-PROPOSAL.pdf")
    print("=" * 80)
    print(f"[3] NEGATION & CONTRADICTION EXTRACTION DEMO: {os.path.basename(epsilon_pdf)}")
    eps_report = extract_evidence_from_pdf(epsilon_pdf)
    neg_records = [e for e in eps_report.evidence_records if e.has_negation]
    print(f"Found {len(neg_records)} statement(s) with explicit negation:\n")
    for nr in neg_records:
        print(f"  * Sentence:      {nr.original_sentence}")
        print(f"  * Negation:      {nr.has_negation}")
        print(f"  * Numeric:       {nr.numeric_values}")
        print(f"  * Page:          Page {nr.page_number}")
        print(f"  * Review Notes:  {nr.review_notes}")
        print()

    print("=" * 80)
    print("PHASE 3 DEMO COMPLETE.")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
