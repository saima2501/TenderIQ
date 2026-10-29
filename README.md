# NLP Procurement Tender & Bidder Compliance Analysis System

An end-to-end Natural Language Processing (NLP) pipeline designed to evaluate procurement tender documents (RFPs, IFBs) against bidder proposals.

> **Project Status:**  
> - **Phase 1:** Synthetic Dataset Generation & Validation ✅ *(Complete)*  
> - **Phase 2:** PDF Document Text Extraction Pipeline with Fallback & Scanned Detection ✅ *(Complete)*  
> - **Phase 3:** NLP Information Extraction (Structured Tender Requirements & Bidder Evidence) ✅ *(Complete)*  
> - **Phase 4:** End-to-End Baseline Demo (TF-IDF Retrieval, Deterministic Scoring, SQLite & FastAPI) ✅ *(Complete)*  
> - **Phase 5:** Fine-Tuned Transformer / NLI Semantic Verification ⏳ *(Pending Approval)*

---

## 1. System Architecture

The pipeline processes documents through modular stages without black-box claims or external paid APIs:

```text
[Tender PDF] & [Bidder PDF]
         │
         ▼
[Phase 2: PDFExtractor & TextCleaner] (pdfplumber + PyMuPDF fallback)
         │
         ▼
[Phase 3: RequirementExtractor & EvidenceExtractor] (Rule-based NLP & Entity extraction)
         │
         ▼
[Phase 4: TFIDFRetriever] (Top-3 candidate evidence sentences retrieved per clause)
         │
         ▼
[Phase 4: NumericComparator] (Deterministic boundary & unit checks: Gbps/Mbps, years/months)
         │
         ▼
[Phase 4: BaselineClassifier] (COMPLIANT, NON_COMPLIANT, UNCERTAIN, MISSING_EVIDENCE)
         │
         ▼
[Phase 4: DatabaseManager & FastAPI] (SQLite persistence & interactive web UI at http://127.0.0.1:8000)
```

---

## 2. Phase 4 Modules Overview

1. **TF-IDF Evidence Retrieval (`backend/matching/tfidf_retriever.py`):**
   - Ranks all candidate evidence sentences in a bidder document against each tender requirement.
   - Computes cosine similarity between query vectors and evidence vectors.
   - Returns the top 3 candidate evidence statements with their similarity scores.
2. **Deterministic Numeric Comparison (`backend/scoring/numeric_comparator.py`):**
   - Evaluates numeric thresholds (`>=`, `<=`, `==`).
   - Automatically converts compatible units:
     - Bandwidth: `1 Gbps` = `1,000 Mbps`
     - Duration: `1 year` = `12 months`, `1 hour` = `60 minutes`
     - Threshold boundaries (e.g. `USD 9.95M` vs `>= USD 10.0M` &rarr; `VIOLATED`).
   - Never equates similarity with compliance.
3. **Transparent Baseline Classifier (`backend/classification/baseline_classifier.py`):**
   - Assigns four explicit compliance states:
     - `COMPLIANT`: Threshold or certification verified without negation.
     - `NON_COMPLIANT`: Numeric shortfall or high-similarity negative claim (e.g. *"does not guarantee"*).
     - `UNCERTAIN`: Vague promotional statements (e.g. *"collective domain knowledge"*) or incompatible units.
     - `MISSING_EVIDENCE`: No matching statement in the bidder proposal (top similarity < 0.08).
4. **SQLite Database Storage (`backend/database/db_manager.py`):**
   - Automatically creates `compliance_demo.db` in `backend/database/`.
   - Stores analysis metadata, summary metrics, and itemized clause evaluations with full audit trails.
5. **FastAPI Backend & Interactive Dashboard (`backend/api/app.py`):**
   - REST API endpoints for document listing, analysis triggering, and history inspection.
   - Built-in HTML5/CSS/JavaScript web dashboard served at `http://127.0.0.1:8000/`.

---

## 3. Quickstart: How to Run the Application

Execute all commands from the project root in **PowerShell** or **Command Prompt**:

### Step 1: Activate Virtual Environment
```powershell
.\venv\Scripts\activate
```

### Step 2: Run Automated Tests
```powershell
# Run Phase 4 tests (Pipeline, API, SQLite, TF-IDF, Numeric, Classifier)
.\venv\Scripts\python -m unittest backend.test_phase4_baseline -v

# Run all project tests across all phases (23 tests)
.\venv\Scripts\python -m unittest discover backend -v
```

### Step 3: Start the Local Application Server
```powershell
.\venv\Scripts\python -m uvicorn backend.api.app:app --reload --port 8000
```

### Step 4: Access the Interactive Web Demo
Open your browser and navigate to:
```text
http://127.0.0.1:8000
```
- Select any tender PDF from the dropdown (e.g. `TENDER-2024-IT-001.pdf`).
- Select any bidder proposal PDF (e.g. `BIDDER-ALPHA-TECH-PROPOSAL.pdf` or `BIDDER-EPSILON-INFRA-PROPOSAL.pdf`).
- Click **Run Compliance Analysis** to view live extracted clauses, top retrieved evidence, compliance badges, and audit explanations.

---

## 4. REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Service status, database check, available document counts |
| `GET` | `/api/documents/tenders` | Lists all synthetic tender RFP files in workspace |
| `GET` | `/api/documents/bidders` | Lists all synthetic bidder proposal files in workspace |
| `POST` | `/api/analyze` | Executes end-to-end analysis; returns structured evaluations & saves to SQLite |
| `GET` | `/api/analyses` | Lists historical analysis summaries |
| `GET` | `/api/analyses/{id}` | Retrieves full itemized evaluations for a specific run |
| `GET` | `/` | Interactive local demo web dashboard |

---

## 5. Directory Structure

```text
NLP_PROJECT/
├── backend/
│   ├── api/                             # FastAPI application & web demo
│   │   ├── __init__.py
│   │   └── app.py                       # REST API & HTML5 dashboard
│   ├── classification/                  # Baseline compliance classifier
│   │   ├── __init__.py
│   │   └── baseline_classifier.py
│   ├── database/                        # SQLite storage
│   │   ├── __init__.py
│   │   ├── compliance_demo.db           # SQLite database
│   │   └── db_manager.py
│   ├── extraction/                      # Phase 2: PDF extraction
│   │   ├── pdf_extractor.py
│   │   ├── text_cleaner.py
│   │   └── test_extractor.py
│   ├── extraction_nlp/                  # Phase 3: Requirement & evidence extraction
│   │   ├── requirement_extractor.py
│   │   ├── evidence_extractor.py
│   │   ├── test_nlp_extractors.py
│   │   └── demo_extraction.py
│   ├── matching/                        # Phase 4: TF-IDF retrieval
│   │   ├── __init__.py
│   │   └── tfidf_retriever.py
│   ├── scoring/                         # Phase 4: Deterministic numeric comparator
│   │   ├── __init__.py
│   │   └── numeric_comparator.py
│   ├── pipeline.py                      # End-to-end analysis orchestrator
│   └── test_phase4_baseline.py          # Phase 4 test suite (6 tests)
├── dataset/
│   ├── synthetic_pdfs/
│   │   ├── tenders/                     # 4 synthetic tender RFP documents
│   │   └── bidders/                     # 6 synthetic bidder proposals & annexures
│   └── README.md
└── README.md
```

---

## 6. Engineering Principles & Limitations

1. **Zero Hallucination:** No missing metrics or values are fabricated. Incomplete clauses and unverified claims are marked for human review.
2. **Similarity Is Not Compliance:** High lexical similarity with negation (e.g. *"does not include 24/7 on-site support"*) is strictly classified as `NON_COMPLIANT`.
3. **No Disqualification Decisions:** The system generates clause-level compliance assessments and audit explanations. Final procurement qualification decisions are reserved for human officers.
4. **Local CPU Execution:** Runs entirely on local CPU without external or paid LLM dependencies.
