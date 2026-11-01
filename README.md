# TenderIQ: AI-Powered Bid Compliance Verification Platform for GeM Procurement

> **Academic & Research Project Notice:**
> TenderIQ is an academic NLP and Deep Learning course project inspired by the Smart India Hackathon problem statement **SIH26100** (*Automated Bid Compliance Verification for GeM Procurement*).
> All evaluations, benchmark metrics, and document collections within this repository are based on a **controlled synthetic dataset** engineered to simulate procurement scenarios. **TenderIQ has not been validated on real, confidential Government e-Marketplace (GeM) submissions, and its benchmark results do not represent accuracy on live government portals.**

---

## 1. Project Title & Overview

**TenderIQ** is an explainable, auditable Natural Language Processing (NLP) pipeline designed to evaluate complex procurement proposals against tender Request for Proposals (RFPs) and Instructions to Bidders (IFBs).

Operating entirely on local CPU without paid third-party APIs or black-box generative models, TenderIQ combines dense semantic retrieval, Natural Language Inference (NLI), deterministic numeric and unit evaluation, and multi-document cross-referencing. Every clause assessment provides transparent provenance, citing exact document filenames, 1-indexed page citations, and verbatim evidence extracts.

---

## 2. Problem Statement

Public procurement authorities (such as India's Government e-Marketplace — GeM) process high-stakes tenders containing hundreds of technical, financial, statutory, experience, and Service Level Agreement (SLA) clauses. Evaluating multiple bidder submissions against these requirements involves reviewing hundreds of pages of proposals, audited financial schedules, and technical datasheets.

Manual evaluation is labor-intensive, error-prone, and prone to oversight. Furthermore, conventional automated document search approaches fail in procurement environments because:
- **Semantic similarity does not equal compliance:** A proposal stating *"We do NOT provide 24/7 on-site support"* yields high semantic overlap with *"Must provide 24/7 on-site support"*, but has opposite compliance polarity.
- **Unit differences and numerical thresholds:** Comparing *"1 Gbps"* against *"1,000 Mbps"* or *"36 months"* against *"3 years"* requires mathematical normalization, not keyword matching.
- **Internal multi-document discrepancies:** Bidders may present self-declared narrative claims in an executive summary (e.g. claiming $14.5M annual revenue) that contradict certified balance sheets in an annexed schedule (e.g. reporting $6.2M audited revenue).

---

## 3. Why TenderIQ Is Needed

Generic Large Language Models (LLMs) and standard Retrieval-Augmented Generation (RAG) tools pose severe risks when applied to statutory procurement:
1. **Hallucination Risk:** Generative models can invent unmentioned certifications, interpolate missing metrics, or produce non-reproducible compliance verdicts.
2. **Opacity:** Black-box embeddings cannot explain *why* an evaluation passed or failed to a statutory audit committee.
3. **Data Sovereignty & Cost:** Transmitting sensitive public procurement bids to external paid cloud APIs violates confidentiality and incurs recurring operational costs.

TenderIQ addresses these challenges by enforcing a strict **zero-hallucination, deterministic verification architecture** that runs offline on standard hardware.

---

## 4. Project Objectives

- **Automate Clause-by-Clause Evaluation:** Extract requirements from tender RFPs and match them with corresponding bidder claims across technical bids and certified annexures.
- **Eliminate Lexical Traps:** Detect subtle negations, waivers, and disclaimers using local Natural Language Inference (NLI) rather than surface similarity.
- **Perform Deterministic Numeric Normalization:** Verify numerical constraints (`>=`, `<=`, `==`) with automatic canonical unit conversions (Gbps/Mbps, years/months, hours/minutes).
- **Reconcile Multi-Document Contradictions:** Establish a strict source-tier hierarchy where certified annexures supersede narrative proposal statements on quantitative metrics.
- **Provide Audit-Ready Explanations:** Output structured evaluations with 1-indexed page citations, raw evidence sentences, and human-readable audit reasoning.

---

## 5. Main Features Currently Implemented (Phases 1–7)

- [x] **Phase 1: Synthetic Dataset & Ground Truth Engineering:** 556 structured requirement-evidence pairs across 4 simulated tenders and 5 bidders covering 9 challenge subtypes (`dataset/splits/`).
- [x] **Phase 2: PDF Extraction Pipeline with Fallback:** Layout-aware text extraction via `pdfplumber` with automatic fallback to `PyMuPDF` (`fitz`), preserving 1-indexed page numbering and flagging scanned bitmap pages (`is_scanned=True`, `SCANNED_NEEDS_OCR`).
- [x] **Phase 3: Structured Information Extraction:** Rule-based NLP extractors (`RequirementExtractor`, `EvidenceExtractor`) that parse parameters, comparison operators, threshold values, units, statutory entities (GSTIN, PAN, ISO), and negation markers.
- [x] **Phase 4: Deterministic Baseline Engine:** Unigram/bigram TF-IDF retrieval, deterministic `NumericComparator`, rule-based `BaselineClassifier`, and SQLite persistence (`compliance_demo.db`).
- [x] **Phase 5: Dense Semantic Retrieval:** Vectorized embedding search using Sentence-BERT (`sentence-transformers/all-MiniLM-L6-v2`) with transparent fallback to TF-IDF.
- [x] **Phase 6: Natural Language Inference Verification:** Local CPU Cross-Encoder inference (`cross-encoder/nli-distilroberta-base`) producing calibrated softmax probabilities across entailment, contradiction, and neutral relationships.
- [x] **Phase 7: Multi-Document Evidence Reconciliation:** `CrossDocumentReconciler` implementing a source-tier hierarchy (Tier 1 Audited Annexures > Tier 3 Proposal Narrative) with a mathematically verified one-way escalation guarantee.
- [x] **Phase 7 Hardening & Interactive Web Demo:** Multi-document package ingestion, contextual financial figure parsing, cross-bidder document isolation (HTTP 400 rejection), side-by-side discrepancy cards, and FastAPI web dashboard.

---

## 6. System Architecture & Technology Stack

```text
[Tender RFP PDF] & [Bidder Proposal PDF + Annexures]
                     │
                     ▼
       Phase 2: PDF Document Extraction
       (pdfplumber + PyMuPDF fallback | TextCleaner)
                     │
                     ▼
       Phase 3: Information Extraction (NLP & NER)
       (RequirementExtractor & EvidenceExtractor)
                     │
                     ▼
       Phase 4/5: Evidence Retrieval & Ranking
       (Sentence-BERT: all-MiniLM-L6-v2 | TF-IDF Fallback)
                     │
                     ▼
       Phase 6: Natural Language Inference (NLI)
       (Cross-Encoder: nli-distilroberta-base)
                     │
                     ▼
       Phase 7: Cross-Document Reconciliation
       (CrossDocumentReconciler | Source-Tier Precedence)
                     │
                     ▼
       Phase 4: Deterministic Rule-Based Classification
       (NumericComparator: Unit & Boundary Checks | BaselineClassifier)
                     │
                     ▼
       Output & Audit Interface
       (SQLite: compliance_demo.db | FastAPI REST API | HTML5 Dashboard)
```

### Technology Stack

| Component | Library / Framework | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Backend Framework** | `fastapi` / `uvicorn` | `0.157.0` / `0.54.0` | Asynchronous REST API & demo server |
| **Data Validation** | `pydantic` | `2.14.1` | Request and schema modeling |
| **PDF Extraction** | `pdfplumber` / `pymupdf` | `0.11.10` / `2.0` | Layout-aware parsing with high-speed fallback |
| **Array Mathematics** | `numpy` | `2.5.4` | Vector operations & similarity dot products |
| **Dense Embeddings** | `sentence-transformers` | `6.2.0` | Sentence-BERT semantic evidence retrieval |
| **NLI Inference** | `transformers` / `torch` | `5.20.2` / `2.14.1` | Local CPU Cross-Encoder inference |
| **Testing** | `pytest` | `9.1.1` | Automated regression test execution |
| **Database** | SQLite (`sqlite3`) | Standard Library | Local audit persistence (`compliance_demo.db`) |
| **Frontend UI** | HTML5 / CSS3 / Vanilla JS | Native Browser | Interactive compliance dashboard |

---

## 7. Explanation of the NLP Pipeline

1. **PDF Document Text Extraction (`backend/extraction/pdf_extractor.py`):**
   Extracts page-by-page text layers while preserving exact 1-indexed page citations. If font tables are damaged or layout extraction fails, it automatically delegates to PyMuPDF. If a document consists of scanned images without a text layer, it sets `is_scanned=True` and status `SCANNED_NEEDS_OCR` without fabricating text.
2. **Requirement & Evidence Extraction (`backend/extraction_nlp/`):**
   - `RequirementExtractor`: Parses RFP text into structured requirements with functional categories (`Technical`, `Financial`, `Experience`, `Statutory`, `SLA`), criteria parameters, comparison operators (`>=`, `<=`, `==`, `MANDATORY_SUBMISSION`), threshold values, and mandatory flags (`must/shall` vs `should/preferred`).
   - `EvidenceExtractor`: Parses proposal sentences, identifies domain entities (GSTIN, PAN, ISO standards), extracts numeric metrics and units, and flags explicit negation markers (`has_negation=True`).
3. **TF-IDF Retrieval (`backend/matching/tfidf_retriever.py`):**
   Builds term frequency-inverse document frequency vectors across bidder candidate sentences and computes cosine similarity against tender queries, returning the top-3 candidate evidence sentences.
4. **Sentence-BERT Semantic Retrieval (`backend/matching/semantic_retriever.py`):**
   Computes 384-dimensional dense semantic embeddings using `all-MiniLM-L6-v2`. Ranks candidates via vectorized cosine similarity to capture conceptual matches that use differing vocabulary. If model weights are unavailable, it falls back to TF-IDF with transparent provenance tagging.
5. **NLI Verification (`backend/classification/nli_classifier.py`):**
   Passes candidate premise-hypothesis pairs through `cross-encoder/nli-distilroberta-base` on local CPU, generating calibrated probabilities for `entailment`, `contradiction`, and `neutral`. Contradictions $\ge 0.65$ override superficial lexical similarity.
6. **Deterministic Numeric Comparison (`backend/scoring/numeric_comparator.py`):**
   Performs strict mathematical threshold checks with canonical unit normalization:
   - Bandwidth: $1\text{ Gbps} = 1,000\text{ Mbps}$.
   - Duration: $1\text{ year} = 12\text{ months}$; $1\text{ hour} = 60\text{ minutes}$.
   - Boundary checks: e.g. offered $9.95\text{M}$ vs required $\ge 10.0\text{M} \to \text{VIOLATED}$.
7. **Cross-Document Evidence Reconciliation (`backend/reconciliation/cross_document_reconciler.py`):**
   Cross-references evidence sentences across multiple submitted files (proposal narrative vs audited annexures). Enforces source precedence: **Tier 1 (Audited balance sheets, staffing schedules, client completion certificates) strictly supersedes Tier 3 (Main proposal self-declarations)**.

---

## 8. The Four Compliance Classification Outcomes

TenderIQ categorizes every tender requirement into one of four distinct, non-overlapping compliance states:

| Status Badge | Semantic Definition | Illustrative Procurement Example | Policy Action |
| :--- | :--- | :--- | :--- |
| **`COMPLIANT`** | Evidence explicitly and unambiguously satisfies or exceeds all criteria stated in the tender clause without negation. | Demand: $\ge 36\text{ months}$ warranty.<br>Offer: $3\text{ years}$ on-site warranty. | Verified pass on criterion. |
| **`NON_COMPLIANT`** | Evidence explicitly fails, falls short of a required threshold, or disclaims mandatory requirements. | Demand: Uptime $\ge 99.95\%$.<br>Offer: Uptime contractually capped at $99.90\%$. | Flagged shortfall for evaluation committee review. |
| **`UNCERTAIN`** | Statement is promotional, vague, unquantified, unit-incompatible, or internally conflicting across documents. | Demand: Experience with databases $> 10\text{ TB}$.<br>Offer: *"Our engineers handle massive enterprise databases."* | Marked for human auditor clarification (RFC). |
| **`MISSING_EVIDENCE`** | Proposal submission contains no statement, section, or annexure addressing the clause (retrieval similarity below threshold). | Demand: Third-party SOC 2 Type II audit report.<br>Offer: No mention or file in submission. | Triggers formal Request for Clarification / Missing Document notice. |

> **Architectural Invariant:** *Missing evidence is strictly distinguished from confirmed non-compliance.* An unmentioned document permits a clarification window under procurement rules, whereas an explicit shortfall is a verified discrepancy.

---

## 9. Dataset Information

- **Type:** Fully simulated synthetic procurement corpus modeled after GeM tender formats.
- **Data Volume:** 556 total requirement-evidence pairs across 4 simulated tenders and 5 bidding entities.
- **Split Strategy:** Stratified multi-class split into:
  - **Train:** 387 records (69.6%)
  - **Validation:** 82 records (14.7%)
  - **Test (Held-Out Benchmark Split):** 87 records (15.6%)
- **Challenge Subtypes Represented:**
  - `numerical_boundary`: Boundary checks right at the threshold ($10.0\text{M}$ vs $10.0\text{M}$, $9.99\text{M}$ vs $10.0\text{M}$).
  - `unit_conversion`: Cross-unit equivalencies (months to years, Gbps to Mbps).
  - `high_similarity_negation`: Lexical overlap $> 85\%$ containing negative polarity words (*"does not include"*).
  - `cross_document_contradiction`: Discrepancies between proposal narrative claims and certified financial annexures.
  - `direct_match`, `paraphrase`, `confirmed_shortfall`, `ambiguous_claim`, and `missing_section`.

---

## 10. Existing Benchmark Results

> **Important Disclosure:** The following metrics were measured on TenderIQ's held-out synthetic test dataset (`dataset/splits/test.jsonl`, $N = 87$ records). **These numbers demonstrate algorithmic progression across development phases on controlled test data and do not constitute accuracy metrics on live GeM procurement bids.**

### Progression Across Implementation Phases

| Evaluation Metric | Phase 4 Deterministic Baseline | Phase 6 NLI Integrated | Phase 7 Cross-Doc Reconciled | Total Delta (P7 vs P4) |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 72.41% (63/87) | 74.71% (65/87) | **79.31% (69/87)** | **+6.90%** |
| **Macro Precision** | 78.91% | 80.00% | **82.44%** | **+3.54%** |
| **Macro Recall** | 75.37% | 77.29% | **81.14%** | **+5.77%** |
| **Macro F1-Score** | 74.28% | 76.51% | **80.68%** | **+6.40%** |
| **Weighted F1-Score** | 72.66% | 75.38% | **80.30%** | **+7.64%** |

### Per-Class Performance (Phase 7 Reconciled)

| Compliance Class | Precision | Recall | F1-Score | Support ($N=87$) |
| :--- | :---: | :---: | :---: | :---: |
| **`COMPLIANT`** | 77.8% (21/27) | 75.0% (21/28) | 76.4% | 28 |
| **`NON_COMPLIANT`** | 100.0% (19/19) | 73.1% (19/26) | 84.4% | 26 |
| **`UNCERTAIN`** | 52.0% (13/25) | 76.5% (13/17) | 61.9% | 17 |
| **`MISSING_EVIDENCE`** | 100.0% (16/16) | 100.0% (16/16) | 100.0% | 16 |

### Confusion Matrix (Phase 7)

```text
Actual \ Predicted     COMPLIANT   NON_COMPLIANT   UNCERTAIN   MISSING_EVIDENCE    Total
COMPLIANT                 21             0             7              0             28
NON_COMPLIANT              2            19             5              0             26
UNCERTAIN                  4             0            13              0             17
MISSING_EVIDENCE           0             0             0             16             16
Total Predicted           27            19            25             16             87
```

- **Reconciliation Impact:** Accuracy on the `cross_document_contradiction` subtype climbed from **0.0% (0/4) in Phase 4/6 to 100.0% (4/4) in Phase 7**.
- **Adversarial Negation Impact:** Accuracy on `high_similarity_negation` increased from **16.7% in Phase 4 to 50.0% in Phase 6/7** via NLI contradiction detection.

---

## 11. Installation Instructions (Windows 64-bit)

### Prerequisites
- Python 3.10, 3.11, or 3.12 (Tested on Python 3.12 64-bit on Windows).
- Git for Windows.
- PowerShell or Windows Command Prompt.

### Step 1: Clone Repository
```powershell
git clone https://github.com/your-username/TenderIQ.git
cd TenderIQ
```

### Step 2: Create and Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### Step 3: Install Pinned Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 12. Instructions to Run the Application Locally

### Step 1: Launch FastAPI Server
From the project root with the virtual environment activated, run:
```powershell
# Set offline cache variables for CPU execution
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"

# Start Uvicorn server on port 8000
python -m uvicorn backend.api.app:app --host 127.0.0.1 --port 8000 --reload
```

### Step 2: Access the Interactive Dashboard
Open your web browser and navigate to:
```text
http://127.0.0.1:8000
```

### Step 3: Run an End-to-End Analysis
1. Select a Tender RFP from the dropdown (e.g. `TENDER-2024-FIN-004.pdf`).
2. Select a Bidder Proposal (e.g. `BIDDER-ALPHA-TECH-PROPOSAL.pdf`).
3. Notice that `ANNEXURE-FINANCIAL-STATEMENTS.pdf` is automatically linked in the **Package Annexures** container.
4. Select retrieval engine: `Sentence-BERT (all-MiniLM-L6-v2)` with `NLI Verification` enabled.
5. Click **Run Compliance Analysis**.
6. Inspect the summary stat cards, compliance badges, verbatim evidence quotes, and the **Cross-Document Discrepancy Card** showing proposal claims vs audited annexure figures.

---

## 13. Instructions to Run Tests

TenderIQ includes 127 automated tests across 8 test suites. Execute the complete test suite with:

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
pytest -o pythonpath=. backend/ -v
```

### Expected Output
```text
======================= 127 passed in ~160s =======================
```

To run individual phase test suites:
```powershell
# PDF Extraction tests (8 tests)
pytest -o pythonpath=. backend/extraction/test_extractor.py -v

# Information Extraction NLP tests (9 tests)
pytest -o pythonpath=. backend/extraction_nlp/test_nlp_extractors.py -v

# Phase 4 Bug Fix Regression tests (31 tests)
pytest -o pythonpath=. backend/test_phase4_bug_fixes.py -v

# Phase 5 Sentence-BERT Semantic tests (11 tests)
pytest -o pythonpath=. backend/test_phase5_semantic.py -v

# Phase 6 NLI Cross-Encoder tests (17 tests)
pytest -o pythonpath=. backend/test_phase6_nli.py -v

# Phase 7 Multi-Document Reconciliation tests (29 tests)
pytest -o pythonpath=. backend/test_phase7_reconciliation.py -v

# Phase 7 Demo Blocker Verification tests (16 tests)
pytest -o pythonpath=. backend/test_demo_blockers.py -v
```

To reproduce the benchmark evaluation on `test.jsonl`:
```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
python backend/evaluation/evaluate_classifier.py
```

---

## 14. Model Information & First-Run Downloads

TenderIQ utilizes two lightweight, open-weight transformer models that run locally on CPU:

1. **Sentence-BERT (`sentence-transformers/all-MiniLM-L6-v2`):**
   - Size: ~90 MB (22.7M parameters).
   - Embedding Dimension: 384.
   - Purpose: Dense semantic retrieval of top-3 candidate evidence sentences per clause.
2. **NLI Cross-Encoder (`cross-encoder/nli-distilroberta-base`):**
   - Size: ~330 MB (82M parameters).
   - Purpose: Direct sequence-pair classification assessing entailment, contradiction, and neutrality.

### First-Run Internet Requirement
On the initial run, Hugging Face downloads and caches model weights into `~/.cache/huggingface/hub/`.
Once cached, the entire platform runs completely offline using:
```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
```

---

## 15. Current Limitations

1. **OCR Scope Boundary:** Text extraction currently parses native digital PDF text layers. Non-text scanned bitmap images are detected and flagged as `SCANNED_NEEDS_OCR` without synthetic text fabrication. OCR integration (e.g. Tesseract) is planned for future work.
2. **Synthetic Dataset Scope:** All evaluations are conducted on controlled synthetic documents modeled after GeM specifications; the platform has not been evaluated on live portal tenders.
3. **Number-Word Parsing:** Spelled-out numbers (e.g. *"one gigabit per second"*, *"over a decade"*) currently fail numeric regexes and default to `UNCERTAIN`.
4. **Currency Scope:** Financial comparison currently converts USD currency formats (Million, M, K). Indian currency units (₹, INR, Lakhs, Crores) are planned.
5. **Cold-Start Latency:** Loading PyTorch and both transformer models on CPU takes ~30 seconds on cold server launch. Subsequent queries execute in under 2 seconds.

---

## 16. Future Improvement Roadmap (Planned Phase 8+)

The following improvements are planned for future development milestones:
- **Number-Word Normalization:** Convert written numerical phrases (*"one gigabit"*, *"five years"*, *"a decade"*) into canonical floats to improve paraphrase detection.
- **Accreditation Expiry Verification:** Extract certificate issue and expiration dates and cross-reference them against the tender submission deadline.
- **Entity Scope Verification:** Disambiguate prime bidder capabilities from parent consortium, subsidiary, or subcontractor claims.
- **Indian Financial Unit Support:** Integrate Indian Rupees (₹, INR, Lakhs, Crores) into `NumericComparator` with cross-currency isolation.
- **Dataset Re-Partitioning:** Re-split synthetic datasets using Stratified Group Splitting on template clusters to eliminate cross-split template overlap.
- **Local OCR Engine:** Integrate local Tesseract OCR for scanned image PDFs.

---

## 17. Human-in-the-Loop Decision-Making & Responsible AI Safeguards

TenderIQ is engineered as an **auditable decision-support assistant**, not an autonomous judicial agent:

1. **No Automated Disqualification:** The system outputs clause-level evaluations, evidence quotations, and audit reasoning. Final qualification or disqualification decisions remain exclusively with human procurement officers.
2. **Similarity Is Not Compliance:** Vector similarity is solely used to retrieve candidates. High similarity never awards compliance without deterministic verification.
3. **Conservative Entailment Policy:** Qualitative NLI entailment alone never grants `COMPLIANT`. Without empirical figures or verified third-party certificates, claims default to `UNCERTAIN` for procurement officer verification.
4. **One-Way Override Guarantee:** Cross-document reconciliation can only escalate scrutiny (`COMPLIANT` $\to$ `UNCERTAIN` or `NON_COMPLIANT`). It is mathematically impossible for reconciliation to upgrade `NON_COMPLIANT` to `COMPLIANT`.
5. **Clarification Window Protection:** Missing declarations are classified as `MISSING_EVIDENCE` to trigger a formal Request for Clarification (RFC) rather than premature disqualification.
