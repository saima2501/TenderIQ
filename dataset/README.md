# Dataset Architecture & Data Dictionary
> **Project:** NLP Tender & Bidder Compliance Analysis System  
> **Status:** Phase 1 – Dataset Generation & Validation  
> **Data Type:** Fully Simulated Synthetic Dataset for Testing & Evaluation  
> **Disclaimer:** *All tender clauses, bidder excerpts, and financial figures are synthetic and created strictly for testing NLP compliance verification pipelines. They do not represent official procurement actions, nor do they claim real-time verification against government portals.*

---

## 1. Executive Summary & Purpose

Procurement tenders (RFPs, IFBs) contain hundreds of technical, legal, financial, and operational clauses. Evaluating bidder proposals against these requirements requires understanding **semantic entailment, numerical thresholds, unit conversions, and cross-document coherence**.

A common pitfall in naive NLP pipelines is equating **semantic similarity** with **compliance**:
- A proposal stating *"We have NO pending litigation"* has high semantic similarity to *"Bidder has 4 pending litigations"*, but their compliance polarities are opposite.
- A bidder offering *"2 years warranty"* satisfies a tender demanding *"24 months warranty"*, even though lexical overlap is low.
- A missing declaration is **NOT** the same as an explicit non-compliance; it represents **MISSING_EVIDENCE**, which requires an audit request or request for clarification (RFC), not premature disqualification.

This dataset provides **over 500 rigorously labeled requirement-evidence pairs** designed to train and benchmark NLP models with complete coverage of edge cases.

---

## 2. Dataset Schema (Data Dictionary)

Each record in the dataset contains the following structured attributes:

| Field | Type | Description | Allowed Values / Example |
| :--- | :--- | :--- | :--- |
| `pair_id` | String | Unique record identifier | `"PAIR-0001"` ... `"PAIR-0520"` |
| `tender_id` | String | Identifier for simulated tender | e.g., `"TND-2024-IT-001"` |
| `tender_title` | String | Title of the tender document | e.g., `"Enterprise Cloud & Cyber Infrastructure"` |
| `tender_category` | String | High-level procurement category | `Technical`, `Financial`, `Experience`, `Statutory`, `Staffing`, `SLA & Warranty` |
| `requirement_id` | String | Specific clause identifier in the tender | e.g., `"REQ-IT-014"` |
| `requirement_text`| String | Complete clause text from tender | `"The cloud hosting infrastructure must guarantee an uptime SLA of at least 99.95% per month."` |
| `is_mandatory` | Boolean | Whether the clause is mandatory (`true`) or evaluative (`false`) | `true`, `false` |
| `bidder_id` | String | Simulated bidder organization ID | e.g., `"BIDDER-ALPHA-TECH"` |
| `bidder_name` | String | Simulated bidder organization name | e.g., `"Alpha Cloud & Cyber Systems Ltd."` |
| `evidence_text` | String/Null | Exact quote extracted from bidder submission | `"Our cloud SLA contractually guarantees 99.98% monthly uptime."` (or `""` if missing) |
| `source_document` | String | Filename of the bidder PDF containing evidence | e.g., `"BIDDER-ALPHA-TECH-PROPOSAL.pdf"` or `"N/A"` |
| `page_number` | Integer/Null | Specific page number in source PDF | `1` to `20`, or `null` if missing |
| `section_reference`| String | Section heading within the document | e.g., `"Section 4.2 - SLA Guarantees"` |
| `label` | String | Ground-truth compliance assessment | `COMPLIANT`, `NON_COMPLIANT`, `UNCERTAIN`, `MISSING_EVIDENCE` |
| `sub_type` | String | Edge case or semantic category tag | `direct_match`, `paraphrase`, `unit_conversion`, `numerical_boundary`, `confirmed_shortfall`, `cross_document_contradiction`, `high_similarity_negation`, `ambiguous_claim`, `missing_section` |
| `similarity_score_heuristic` | Float | Baseline lexical/embedding cosine score (0.0 to 1.0) | Demonstrates disparity between similarity and logical compliance |
| `contradiction_metadata` | Object/Null | Conflicting cross-document details | Contains `is_contradiction`, `conflicting_document`, `conflicting_page`, `conflicting_text` |
| `numerical_metadata` | Object/Null | Numerical evaluation details | Contains `is_numerical`, `required_val`, `required_unit`, `offered_val`, `offered_unit`, `operator` |
| `explanation_reasoning` | String | Human-readable explanation of why this label was assigned | Beginner-friendly rationale |
| `is_simulated` | Boolean | Simulation watermark flag | `true` |
| `disclaimer` | String | Standard academic simulation disclaimer | `"SIMULATED DATA FOR TESTING..."` |

---

## 3. Label Taxonomies & Definitions

### 1. `COMPLIANT`
- **Definition:** The evidence explicitly and unambiguously satisfies or exceeds all criteria stated in the tender clause.
- **Examples:**
  - *Direct match:* Requirement asks for ISO 27001; proposal provides valid ISO 27001 certificate reference.
  - *Paraphrase:* Requirement: *"System must operate without continuous internet connectivity"*; Evidence: *"The software supports full offline operation with local database caching."*
  - *Unit conversion:* Requirement: *"Minimum 24 months warranty"*; Evidence: *"Comprehensive 2-year warranty covering parts and labor."*
  - *Boundary pass:* Requirement: *"Annual turnover >= $10.0M"*; Evidence: *"Audited annual turnover of $10.0M (FY2023)."*

### 2. `NON_COMPLIANT`
- **Definition:** The evidence explicitly fails, falls short of, or violates a tender requirement.
- **Examples:**
  - *Confirmed shortfall:* Requirement: *"Minimum 5 years experience"*; Evidence: *"Our firm was incorporated 3 years ago in 2021."*
  - *Numerical boundary fail:* Requirement: *"Minimum uptime 99.95%"*; Evidence: *"We commit to an uptime SLA of 99.90%."*
  - *High similarity negation:* Requirement: *"Must include 24/7 on-site incident response"*; Evidence: *"Support does NOT include 24/7 on-site incident response; remote assistance is provided."*
  - *Cross-document contradiction:* Technical proposal claims 12 certified senior engineers, but Annexure B (Staffing Bio Table) only details 7 certified engineers.

### 3. `UNCERTAIN`
- **Definition:** Evidence is ambiguous, vague, promotional, or lacks required verification details.
- **Examples:**
  - Requirement: *"Must demonstrate experience with databases exceeding 10 TB in size."*
  - Evidence: *"Our team has handled massive enterprise databases with millions of records."* *(Lacks specific size or verifiable figure).*
  - Requirement: *"Third-party audited SOC 2 Type II report required."*
  - Evidence: *"We undergo regular internal security reviews aligned with SOC 2 principles."* *(Internal review is not a third-party audit).*

### 4. `MISSING_EVIDENCE`
- **Definition:** The bidder's submission contains no statement, section, or annexure addressing the requirement.
- **Key Architectural Rule:** *Missing evidence is distinct from confirmed non-compliance.* A missing document may trigger a Request for Clarification (RFC) or remediation window, whereas an explicit shortfall is a verified discrepancy.

---

## 4. Edge Cases Represented

1. **Numerical Boundary Cases:** Values tested right at the threshold ($10.0M vs $10.0M, $9.99M vs $10.0M, $10.1M vs $10.0M).
2. **Unit Conversions:** Months to years, Gbps to Mbps, metric tons to kilograms, hours to business days.
3. **Paraphrasing (Semantic Entailment):** Completely distinct vocabulary that logically implies compliance.
4. **Cross-Document Contradictions:** Inconsistencies between Technical Proposals, Financial Proposals, and Annexures.
5. **Adversarial / High-Similarity Negations:** Lexical overlap > 0.85 where negative words (`not`, `excluding`, `exempt`) invert compliance.

---

## 5. Storage Formats

The generated dataset is serialized into:
- `dataset/raw/requirement_evidence_dataset.json` (Structured JSON array)
- `dataset/raw/requirement_evidence_dataset.jsonl` (Line-delimited JSON for streaming / model loaders)
- `dataset/raw/requirement_evidence_dataset.csv` (Tabular format for spreadsheet analysis and inspection)
- `dataset/splits/train.jsonl`, `val.jsonl`, `test.jsonl` (Stratified 70% / 15% / 15% splits)

---

## 6. Integration with Pipeline Phases

- **Phase 2 (PDF Extraction):** Implemented in `backend/extraction/` (`pdf_extractor.py`, `text_cleaner.py`).
- **Phase 3 (NLP Extraction):** Implemented in `backend/extraction_nlp/` (`requirement_extractor.py`, `evidence_extractor.py`).
- Refer to the root [`README.md`](../README.md) for full architecture and execution instructions.

