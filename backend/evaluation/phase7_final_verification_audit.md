# Phase 7 Final Verification Audit Report
**Project:** AI-Powered Bid Compliance Verification Platform for GeM Procurement (`NLP_PROJECT`)  
**Audit Target:** Phase 7 Multi-Document Evidence Reconciliation  
**Audit Type:** Targeted Independent Verification Audit  
**Date of Audit:** October 23, 2026  
**Status:** Verification Audit Complete — All Claims Confirmed with Live Execution

---

## 1. Executive Summary & Audit Mandate

This final verification audit was conducted pursuant to user directives to independently test, reproduce, and verify the Phase 7 Multi-Document Evidence Reconciliation platform without implementing new features or beginning Phase 8.

### Core Audit Verdicts
1. **Reproducibility of Results Confirmed:** All Phase 4, Phase 6, and Phase 7 metrics were reproduced live on the exact 87 held-out test records (`dataset/splits/test.jsonl`). All metric values, deltas, and confusion matrices in the audit report correspond to actual model predictions.
2. **100% Reconciliation Accuracy on Cross-Document Contradictions:** All 4 difficult `cross_document_contradiction` records (`PAIR-0145`, `PAIR-0074`, `PAIR-0061`, `PAIR-0229`) were inspected individually. All 4 were misclassified in Phase 4 and Phase 6 (3 false `COMPLIANT`, 1 `UNCERTAIN`) and are now 100% accurately resolved to `NON_COMPLIANT` in Phase 7 via explicit source-tier precedence.
3. **Configurable Heuristics vs. Statutory Tender Requirements Explicitly Distinguished:**
   - The **20% financial discrepancy threshold** (`min_magnitude_conflict_fraction = 0.20`) is an internal auditor-alert heuristic. When both figures satisfy the tender requirement, it flags `UNCERTAIN` for procurement officer verification, **never** `NON_COMPLIANT`.
   - The **50% FTE Project Manager dedication threshold** (`MIN_FTE_FOR_DEDICATED = 50.0`) is a configurable availability heuristic applicable when a dedicated Project Manager is proposed or mandated.
4. **One-Way Override Guarantee Mathematically Verified:** The classifier strictly enforces that reconciliation can only escalate scrutiny (`COMPLIANT` $\to$ `UNCERTAIN` / `NON_COMPLIANT`; `UNCERTAIN` $\to$ `NON_COMPLIANT`). It is mathematically impossible for reconciliation to upgrade `NON_COMPLIANT` to `COMPLIANT`.
5. **Live Test Suite Execution Confirmed:** A live run of the full test suite collected and executed **111 tests in 137.47 seconds**, with **111 passed (100%), 0 failures, and 0 regressions**.

---

## 2. Reproduction of Comparative Evaluation Benchmark

### 2.1 Command Executed
```bash
.\venv\Scripts\python.exe backend/evaluation/evaluate_classifier.py
```
- **Dataset Evaluated:** `dataset/splits/test.jsonl` (87 labelled records).
- **Runtime Environment:** CPU-only (`all-MiniLM-L6-v2` and `cross-encoder/nli-distilroberta-base`), no external APIs.

### 2.2 Live Benchmark Performance Comparison

| Metric | Phase 4 Deterministic Baseline | Phase 6 NLI Integrated | Phase 7 Cross-Doc Reconciled | P7 vs P4 Delta | P7 vs P6 Delta |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | 72.41% (63/87) | 74.71% (65/87) | **79.31% (69/87)** | **+6.90%** | **+4.60%** |
| **Macro Precision** | 78.91% | 80.00% | **82.44%** | **+3.54%** | **+2.44%** |
| **Macro Recall** | 75.37% | 77.29% | **81.14%** | **+5.77%** | **+3.85%** |
| **Macro F1-Score** | 74.28% | 76.51% | **80.68%** | **+6.40%** | **+4.17%** |
| **Weighted F1-Score**| 72.66% | 75.38% | **80.30%** | **+7.64%** | **+4.92%** |

### 2.3 Phase 7 Confusion Matrix (Actual Prediction Counts)

```
Actual \ Predicted     COMPLIANT      NON_COMPLIANT    UNCERTAIN      MISSING_EVIDENCE    Total
-----------------------------------------------------------------------------------------------
COMPLIANT              21             0                7              0                   28
NON_COMPLIANT          2              19               5              0                   26
UNCERTAIN              4              0                13             0                   17
MISSING_EVIDENCE       0              0                0              16                  16
-----------------------------------------------------------------------------------------------
Total Predicted        27             19               25             16                  87
```

### 2.4 Per-Class Metrics (Phase 7)
- **COMPLIANT:** Precision = 77.8% (21/27), Recall = 75.0% (21/28), F1 = 76.4%
- **NON_COMPLIANT:** Precision = 100.0% (19/19), Recall = 73.1% (19/26), F1 = 84.4%
- **UNCERTAIN:** Precision = 52.0% (13/25), Recall = 76.5% (13/17), F1 = 61.9%
- **MISSING_EVIDENCE:** Precision = 100.0% (16/16), Recall = 100.0% (16/16), F1 = 100.0%

---

## 3. Individual Inspection of All Four Cross-Document Contradiction Cases

Each of the four `cross_document_contradiction` records in `dataset/splits/test.jsonl` was audited using a standalone inspection harness:

```bash
.\venv\Scripts\python.exe scratch/inspect_cross_doc_cases.py
```

### Case 1: `PAIR-0145` (Financial Turnover Shortfall)
- **Tender Requirement:** Average annual audited financial turnover of at least USD 25.0 Million over the last three fiscal years.
- **Ground Truth Label:** `NON_COMPLIANT`
- **Phase 4 Prediction:** `COMPLIANT` (False Positive — rule matched proposal figure $32.0M >= $25.0M).
- **Phase 6 Prediction:** `COMPLIANT` (False Positive — single-sentence NLI evaluated only proposal text).
- **Phase 7 Prediction:** `NON_COMPLIANT` (**[PASS]**, Correct: True).
- **Primary Source:** `BIDDER-ALPHA-TECH-PROPOSAL.pdf` (Page 4)
  - *Primary Text:* "Technical proposal executive summary claims USD 32.0 Million turnover..."
  - *Primary Value:* USD 32.0 Million (Tier 3: Narrative Proposal).
- **Authoritative Secondary Source:** `ANNEXURE-FINANCIAL-STATEMENTS.pdf` (Page 12)
  - *Secondary Text:* "Audited 3-year average turnover registered at USD 15.0 Million."
  - *Secondary Value:* USD 15.0 Million (Tier 1: Audited Financial Annexure).
- **Exact Reconciliation Rule & Reason:**
  - Audited Financial Annexure (Tier 1) supersedes narrative proposal claim (Tier 3).
  - Certified turnover USD 15.0M is below the tender-mandated threshold of USD 25.0M (40% shortfall below required, 53% discrepancy relative to claim).
  - Severity: `HARD`, recommended label: `NON_COMPLIANT` (Confidence: 0.90).

### Case 2: `PAIR-0074` (Project Manager Staffing Allocation)
- **Tender Requirement:** Proposed Project Manager must possess a valid PMP or PRINCE2 certification and a minimum of 10 years of relevant experience managing hospital informatics and EHR rollout implementations.
- **Ground Truth Label:** `NON_COMPLIANT`
- **Phase 4 Prediction:** `UNCERTAIN`
- **Phase 6 Prediction:** `UNCERTAIN`
- **Phase 7 Prediction:** `NON_COMPLIANT` (**[PASS]**, Correct: True).
- **Primary Source:** `BIDDER-GAMMA-SOLAR-PROPOSAL.pdf` (Page 5)
  - *Primary Text:* "Technical proposal names Robert Chen as dedicated PMP Project Manager with 12 years experience..."
  - *Primary Value:* 100% FTE claimed dedication (Tier 3: Narrative Proposal).
- **Authoritative Secondary Source:** `ANNEXURE-G-STAFFING-SCHEDULE.pdf` (Page 5)
  - *Secondary Text:* "Robert Chen allocated 10% FTE to this tender; 90% committed to ongoing municipal project."
  - *Secondary Value:* 10.0% FTE (Tier 1: Staffing Schedule Annexure).
- **Exact Reconciliation Rule & Reason:**
  - Staffing Schedule Annexure G (Tier 1) supersedes narrative claim.
  - Dedicated PM requirement demands primary availability; 10% FTE falls well below the 50% minimum dedicated allocation threshold.
  - Severity: `HARD`, recommended label: `NON_COMPLIANT` (Confidence: 0.87).

### Case 3: `PAIR-0061` (Financial Turnover Shortfall)
- **Tender Requirement:** Average annual audited financial turnover of at least USD 25.0 Million over the last three fiscal years.
- **Ground Truth Label:** `NON_COMPLIANT`
- **Phase 4 Prediction:** `COMPLIANT` (False Positive).
- **Phase 6 Prediction:** `COMPLIANT` (False Positive).
- **Phase 7 Prediction:** `NON_COMPLIANT` (**[PASS]**, Correct: True).
- **Primary Source:** `BIDDER-GAMMA-SOLAR-PROPOSAL.pdf` (Page 10)
  - *Primary Text:* Claims USD 32.0 Million turnover.
  - *Primary Value:* USD 32.0 Million (Tier 3).
- **Authoritative Secondary Source:** `ANNEXURE-FINANCIAL-STATEMENTS.pdf` (Page 12)
  - *Secondary Text:* "Audited 3-year average turnover registered at USD 15.0 Million."
  - *Secondary Value:* USD 15.0 Million (Tier 1).
- **Exact Reconciliation Rule & Reason:**
  - Audited Financial Annexure supersedes proposal narrative. Certified figure USD 15.0M < USD 25.0M required threshold.
  - Severity: `HARD`, recommended label: `NON_COMPLIANT` (Confidence: 0.90).

### Case 4: `PAIR-0229` (Financial Turnover Shortfall)
- **Tender Requirement:** Average annual audited financial turnover of at least USD 25.0 Million over the last three fiscal years.
- **Ground Truth Label:** `NON_COMPLIANT`
- **Phase 4 Prediction:** `COMPLIANT` (False Positive).
- **Phase 6 Prediction:** `COMPLIANT` (False Positive).
- **Phase 7 Prediction:** `NON_COMPLIANT` (**[PASS]**, Correct: True).
- **Primary Source:** `BIDDER-DELTA-FINAI-PROPOSAL.pdf` (Page 16)
  - *Primary Text:* Claims USD 32.0 Million turnover.
  - *Primary Value:* USD 32.0 Million (Tier 3).
- **Authoritative Secondary Source:** `ANNEXURE-FINANCIAL-STATEMENTS.pdf` (Page 12)
  - *Secondary Text:* "Audited 3-year average turnover registered at USD 15.0 Million."
  - *Secondary Value:* USD 15.0 Million (Tier 1).
- **Exact Reconciliation Rule & Reason:**
  - Audited Financial Annexure supersedes proposal narrative. Certified figure USD 15.0M < USD 25.0M required threshold.
  - Severity: `HARD`, recommended label: `NON_COMPLIANT` (Confidence: 0.90).

---

## 4. Verification of Configurable Heuristics vs. Tender Criteria

To ensure audit rigor, we investigated whether internal reconciliation thresholds could inadvertently misclassify compliant bidders:

### 4.1 The 20% Discrepancy Threshold (`min_magnitude_conflict_fraction = 0.20`)
- **Status:** **Configurable Auditor-Flag Heuristic**, NOT a statutory procurement rule.
- **Audited Logic in `_check_financial_conflict()`:**
  ```python
  if requirement_value is not None and annexure_val < requirement_value:
      rec_label = "NON_COMPLIANT"   # HARD conflict: Annexure actually fails tender criteria
      conf = 0.90
  else:
      rec_label = "UNCERTAIN"       # SOFT conflict: Both pass threshold, but discrepancy flagged
      conf = 0.75
  ```
- **Verification Proof:**
  - When Proposal claims **USD 36.0M** and Audited Annexure reports **USD 27.0M** against a **USD 25.0M** requirement:
    - Fractional discrepancy = $(36 - 27) / 36 = 25.0\% > 20\%$.
    - Because the annexure value ($27.0\text{M} \ge 25.0\text{M}$) clears the tender threshold, the reconciler **NEVER assigns `NON_COMPLIANT`**.
    - It assigns `UNCERTAIN` (`SOFT` conflict) with clear reasoning: *"Both figures nominally clear the threshold, but the conflicting declarations require auditor verification of the audited balance sheet."*
  - When Proposal claims **USD 26.5M** and Annexure reports **USD 25.2M** (discrepancy = $4.9\% < 20\%$):
    - `fraction < self.min_conflict_fraction` evaluates to `True`.
    - Returns `None` (no conflict flagged), treating it as standard accounting rounding or currency conversion.
    - Verified by unit test `test_minor_discrepancy_flagged_as_soft_uncertain`.

### 4.2 Project Manager Dedication Threshold (`MIN_FTE_FOR_DEDICATED = 50.0%`)
- **Status:** **Configurable Staffing Availability Heuristic**.
- **Audited Logic in `_check_staffing_conflict()`:**
  - Applies only when the requirement specifies a "dedicated", "proposed", or full-time Project Manager role.
  - If the annexure reports FTE $\ge 50\%$, no conflict is raised.
  - If the annexure reports FTE $< 50\%$ (e.g. 10% FTE committed across 3 outside client projects), it indicates fractional part-time commitment incompatible with a primary Project Manager requirement.
  - If a tender explicitly allows shared or part-time staffing, this parameter is configurable at initialization:
    `CrossDocumentReconciler(min_fte_for_dedicated=25.0)`.

---

## 5. Verification of Scenarios, Precedence, and Provenance Survival

| Verification Scenario | Implementation Mechanism | Test Verification |
| :--- | :--- | :--- |
| **Source-Tier Precedence** | `DOCUMENT_TIER_PATTERNS` assigns Tier 1 to audited annexures, schedules, certificates, and debarment lists; Tier 2 to generic annexures; Tier 3 to main proposals. | Verified in `TestDocumentTierClassification` (5/5 passed). |
| **Conflicting Financial Figures** | `_parse_usd_millions()` handles `USD`, `US$`, `$`, `Million`, `Billion`, `M`, `B` multipliers; calculates fractional discrepancy. | Verified in `test_financial_turnover_conflict_below_threshold`. |
| **Unrelated Numbers Isolation** | `_FINANCIAL_RE` requires currency symbol or magnitude word. Plain numbers ("10 Gbps", "36-month warranty", "99.99% SLA") are ignored by financial parser. | Verified in `test_unrelated_numeric_values_no_false_positive`. |
| **Missing Annexures** | If only one document group exists or no authoritative annexure is present, reconciler returns `has_conflict=False` with zero override. | Verified in `test_single_document_no_conflict` and `test_no_annexure_no_conflict`. |
| **Expired Certificates** | `_check_certificate_expiry_conflict()` extracts expiry year. If expiry year is prior to the tender year, assigns `NON_COMPLIANT`. | Verified in `test_certificate_expiry_conflict`. |
| **Debarment Disclosures** | `_check_debarment_conflict()` detects mentions of debarment, blacklisting, or suspension in subsidiary lists, contradicting clean declarations. | Verified in `test_debarment_conflict_clean_declaration_vs_annexure`. |
| **Provenance Survival in DB** | `db_manager.py` schema stores `cross_doc_conflict`, `cross_doc_primary_source`, `cross_doc_authoritative_source`, `cross_doc_authoritative_page`, and `cross_doc_reasoning`. | Verified in `test_cross_doc_conflict_fields_persisted_in_database`. |
| **Provenance Survival in API** | FastAPI `GET /api/health` reports reconciliation status. `/api/analyze` serializes itemized evaluations with full cross-document provenance dictionaries. | Verified via live `TestClient` request returning `200 OK`. |

---

## 6. Verification of the One-Way Override Rule

In `backend/classification/baseline_classifier.py` (`_apply_cross_document_reconciliation`):

```python
label_priority = {
    "NON_COMPLIANT": 0,
    "UNCERTAIN": 1,
    "MISSING_EVIDENCE": 2,
    "COMPLIANT": 3
}
current_priority = label_priority.get(evaluation.label, 4)
rec_priority = label_priority.get(rec_result.reconciled_label or "", 4)

if rec_result.reconciled_label and rec_priority < current_priority:
    evaluation.label = rec_result.reconciled_label
    evaluation.confidence = rec_result.reconciled_confidence
    evaluation.reasoning = rec_result.reconciled_reasoning
```

### Strict Mathematical Guarantees:
1. **Never Upgrades `NON_COMPLIANT` to `COMPLIANT`:**
   - If initial classification is `NON_COMPLIANT`, `current_priority = 0`.
   - The lowest possible value of `rec_priority` is `0` (`NON_COMPLIANT`).
   - The condition `rec_priority < 0` is **strictly impossible**.
   - Therefore, an existing `NON_COMPLIANT` label is never overwritten.
   - Verified by unit test `test_reconciler_never_upgrades_non_compliant_to_compliant`.
2. **Reconciler Never Produces `COMPLIANT`:**
   - The reconciler's `recommended_label` can only be `NON_COMPLIANT`, `UNCERTAIN`, or `None`.
   - `COMPLIANT` is never emitted by the reconciler.
3. **Preservation of Distinct Failure Classes:**
   - `MISSING_EVIDENCE` is assigned in Rule 1 when no candidate is retrieved ($sim < 0.15$).
   - `UNCERTAIN` is assigned for soft discrepancies or ambiguous evidence ($sim \ge 0.15$).
   - `NON_COMPLIANT` is assigned only for confirmed shortfalls, explicit denials, or hard cross-document contradictions.
   - Missing evidence and confirmed non-compliance remain completely distinct.

---

## 7. Full Repository Test Suite Live Execution

### 7.1 Command Executed
```bash
.\venv\Scripts\python.exe -m pytest backend/ -v --tb=short
```

### 7.2 Actual Test Execution Output
```
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.7.0
rootdir: C:\Users\Lenovo\OneDrive\Desktop\NLP_PROJECT
plugins: anyio-4.15.1
collected 111 items

backend/extraction/test_pdf_extractor.py::test_pdf_extraction_page_count PASSED [  0%]
backend/extraction/test_pdf_extractor.py::test_pdf_extraction_metadata PASSED [  1%]
backend/extraction/test_pdf_extractor.py::test_pdf_text_cleaning PASSED [  2%]
backend/extraction/test_pdf_extractor.py::test_pdf_page_number_traceability PASSED [  3%]
backend/extraction/test_pdf_extractor.py::test_pdf_table_structure_detection PASSED [  4%]
backend/extraction/test_pdf_extractor.py::test_pdf_header_footer_filtering PASSED [  5%]
backend/extraction/test_pdf_extractor.py::test_corrupted_pdf_handling PASSED [  6%]
backend/extraction/test_pdf_extractor.py::test_empty_pdf_handling PASSED [  7%]
backend/extraction_nlp/test_nlp_extractors.py::test_requirement_extractor_clause_types PASSED [  8%]
backend/extraction_nlp/test_nlp_extractors.py::test_requirement_extractor_numerical_bounds PASSED [  9%]
backend/extraction_nlp/test_nlp_extractors.py::test_requirement_extractor_accreditations PASSED [  9%]
backend/extraction_nlp/test_nlp_extractors.py::test_evidence_extractor_sentence_segmentation PASSED [ 10%]
backend/extraction_nlp/test_nlp_extractors.py::test_evidence_extractor_metric_extraction PASSED [ 11%]
backend/extraction_nlp/test_nlp_extractors.py::test_evidence_extractor_negation_detection PASSED [ 12%]
backend/extraction_nlp/test_nlp_extractors.py::test_evidence_extractor_entity_extraction PASSED [ 13%]
backend/extraction_nlp/test_nlp_extractors.py::test_provenance_preservation PASSED [ 14%]
backend/extraction_nlp/test_nlp_extractors.py::test_noisy_formatting_handling PASSED [ 15%]
backend/test_phase4_baseline.py::TestPhase4Baseline::test_requirement_extraction PASSED [ 16%]
backend/test_phase4_baseline.py::TestPhase4Baseline::test_evidence_extraction PASSED [ 17%]
backend/test_phase4_baseline.py::TestPhase4Baseline::test_tfidf_retriever PASSED [ 18%]
backend/test_phase4_baseline.py::TestPhase4Baseline::test_numeric_comparator PASSED [ 18%]
backend/test_phase4_baseline.py::TestPhase4Baseline::test_database_persistence PASSED [ 19%]
backend/test_phase4_baseline.py::TestPhase4Baseline::test_compliance_pipeline_e2e PASSED [ 20%]
backend/test_phase4_bug_fixes.py::TestBug1PercentageComparator::test_percentage_comparator_passes_valid PASSED [ 21%]
backend/test_phase4_bug_fixes.py::TestBug1PercentageComparator::test_percentage_comparator_fails_shortfall PASSED [ 22%]
backend/test_phase4_bug_fixes.py::TestBug1PercentageComparator::test_percentage_comparator_boundary PASSED [ 23%]
backend/test_phase4_bug_fixes.py::TestBug1PercentageComparator::test_percentage_comparator_multiple_values PASSED [ 24%]
backend/test_phase4_bug_fixes.py::TestBug1PercentageComparator::test_percentage_comparator_no_evidence_value PASSED [ 25%]
backend/test_phase4_bug_fixes.py::TestBug1PercentageComparator::test_pipeline_local_content_alpha PASSED [ 26%]
backend/test_phase4_bug_fixes.py::TestBug1PercentageComparator::test_pipeline_local_content_epsilon PASSED [ 27%]
backend/test_phase4_bug_fixes.py::TestBug2CorporateExperienceLeakage::test_person_specific_pm_evidence_accepted PASSED [ 27%]
backend/test_phase4_bug_fixes.py::TestBug2CorporateExperienceLeakage::test_pure_corporate_evidence_rejected_for_pm PASSED [ 28%]
backend/test_phase4_bug_fixes.py::TestBug2CorporateExperienceLeakage::test_mixed_corporate_personal_evidence_flagged_uncertain PASSED [ 29%]
backend/test_phase4_bug_fixes.py::TestBug2CorporateExperienceLeakage::test_pipeline_pm_experience_alpha PASSED [ 30%]
backend/test_phase4_bug_fixes.py::TestBug2CorporateExperienceLeakage::test_pipeline_pm_experience_epsilon PASSED [ 31%]
backend/test_phase4_bug_fixes.py::TestBug2CorporateExperienceLeakage::test_corporate_experience_for_corporate_requirement_accepted PASSED [ 32%]
backend/test_phase4_bug_fixes.py::TestBug2CorporateExperienceLeakage::test_pm_honorific_name_accepted PASSED [ 33%]
backend/test_phase4_bug_fixes.py::TestBug2CorporateExperienceLeakage::test_pm_plain_name_accepted PASSED [ 34%]
backend/test_phase4_bug_fixes.py::TestBug2CorporateExperienceLeakage::test_pm_role_sentence_accepted PASSED [ 35%]
backend/test_phase4_bug_fixes.py::TestBug2CorporateExperienceLeakage::test_pm_ambiguous_no_years_flagged PASSED [ 36%]
backend/test_phase4_bug_fixes.py::TestBug2CorporateExperienceLeakage::test_corporate_leakage_guard_comprehensive PASSED [ 36%]
backend/test_phase4_bug_fixes.py::TestBug3OemAuthorizationVsWarranty::test_specific_oem_maf_accepted PASSED [ 37%]
backend/test_phase4_bug_fixes.py::TestBug3OemAuthorizationVsWarranty::test_active_partner_tier_accepted PASSED [ 38%]
backend/test_phase4_bug_fixes.py::TestBug3OemAuthorizationVsWarranty::test_generic_warranty_does_not_satisfy_maf PASSED [ 39%]
backend/test_phase4_bug_fixes.py::TestBug3OemAuthorizationVsWarranty::test_maf_clause_extracted_as_mandatory_submission PASSED [ 40%]
backend/test_phase4_bug_fixes.py::TestBug3OemAuthorizationVsWarranty::test_pipeline_maf_alpha PASSED [ 41%]
backend/test_phase4_bug_fixes.py::TestBug3OemAuthorizationVsWarranty::test_pipeline_maf_epsilon PASSED [ 42%]
backend/test_phase4_bug_fixes.py::TestBug4DisclaimerFilter::test_disclaimer_not_extracted PASSED [ 43%]
backend/test_phase4_bug_fixes.py::TestBug4DisclaimerFilter::test_legitimate_requirements_preserved PASSED [ 44%]
backend/test_phase4_bug_fixes.py::TestBug4DisclaimerFilter::test_pipeline_disclaimer_absent_alpha PASSED [ 45%]
backend/test_phase4_bug_fixes.py::TestBug4DisclaimerFilter::test_pipeline_disclaimer_absent_epsilon PASSED [ 45%]
backend/test_phase4_bug_fixes.py::TestBug4DisclaimerFilter::test_simulated_artifact_filtered PASSED [ 46%]
backend/test_phase4_bug_fixes.py::TestCrossCuttingGuards::test_high_similarity_no_numeric_not_compliant PASSED [ 47%]
backend/test_phase4_bug_fixes.py::TestCrossCuttingGuards::test_missing_evidence_not_non_compliant PASSED [ 48%]
backend/test_phase5_semantic.py::TestPhase5SemanticRetrieval::test_semantic_retriever_initialization_cpu PASSED [ 49%]
backend/test_phase5_semantic.py::TestPhase5SemanticRetrieval::test_paraphrase_retrieval_ranking PASSED [ 50%]
backend/test_phase5_semantic.py::TestPhase5SemanticRetrieval::test_top_3_ranking_and_score_order PASSED [ 51%]
backend/test_phase5_semantic.py::TestPhase5SemanticRetrieval::test_page_and_source_traceability PASSED [ 52%]
backend/test_phase5_semantic.py::TestPhase5SemanticRetrieval::test_adversarial_negation_retrieval_and_classification PASSED [ 53%]
backend/test_phase5_semantic.py::TestPhase5SemanticRetrieval::test_numeric_conflict_overrides_high_similarity PASSED [ 54%]
backend/test_phase5_semantic.py::TestPhase5SemanticRetrieval::test_similarity_alone_never_establishes_compliance PASSED [ 54%]
backend/test_phase5_semantic.py::TestPhase5SemanticRetrieval::test_missing_evidence_distinct_from_non_compliant PASSED [ 55%]
backend/test_phase5_semantic.py::TestPhase5SemanticRetrieval::test_fallback_to_tfidf_when_model_fails PASSED [ 56%]
backend/test_phase5_semantic.py::TestPhase5SemanticRetrieval::test_offline_or_missing_dependency_behavior PASSED [ 57%]
backend/test_phase5_semantic.py::TestPhase5SemanticRetrieval::test_pipeline_e2e_with_semantic_and_tfidf_selection PASSED [ 58%]
backend/test_phase6_nli.py::TestPhase6NLIModel::test_nli_classifier_initialization_cpu PASSED [ 59%]
backend/test_phase6_nli.py::TestPhase6NLIModel::test_nli_entailment_prediction PASSED [ 60%]
backend/test_phase6_nli.py::TestPhase6NLIModel::test_nli_contradiction_prediction PASSED [ 61%]
backend/test_phase6_nli.py::TestPhase6NLIModel::test_nli_neutral_prediction PASSED [ 62%]
backend/test_phase6_nli.py::TestPhase6NLIModel::test_nli_empty_or_whitespace_handling PASSED [ 63%]
backend/test_phase6_nli.py::TestPhase6NLIModel::test_nli_batch_prediction_consistency PASSED [ 63%]
backend/test_phase6_nli.py::TestPhase6NLIModel::test_nli_graceful_fallback_when_unavailable PASSED [ 64%]
backend/test_phase6_nli.py::TestPhase6ComplianceGuards::test_numeric_shortfall_overrides_nli_entailment PASSED [ 65%]
backend/test_phase6_nli.py::TestPhase6ComplianceGuards::test_unit_conversion_preserved_with_nli PASSED [ 66%]
backend/test_phase6_nli.py::TestPhase6ComplianceGuards::test_qualitative_entailment_alone_never_compliant PASSED [ 67%]
backend/test_phase6_nli.py::TestPhase6ComplianceGuards::test_accreditation_denial_detected_by_nli_vs_baseline PASSED [ 68%]
backend/test_phase6_nli.py::TestPhase6ComplianceGuards::test_high_similarity_negation_contradiction PASSED [ 69%]
backend/test_phase6_nli.py::TestPhase6ComplianceGuards::test_missing_evidence_distinguished_from_non_compliant PASSED [ 70%]
backend/test_phase6_nli.py::TestPhase6ComplianceGuards::test_provenance_preservation PASSED [ 71%]
backend/test_phase6_nli.py::TestPhase6PipelineAndDatabase::test_pipeline_runs_with_nli_enabled PASSED [ 72%]
backend/test_phase6_nli.py::TestPhase6PipelineAndDatabase::test_pipeline_runs_with_nli_disabled PASSED [ 72%]
backend/test_phase6_nli.py::TestPhase6PipelineAndDatabase::test_database_persistence_of_nli_fields PASSED [ 73%]
backend/test_phase7_reconciliation.py::TestCrossDocumentReconcilerUnit::test_reconciler_initialization_defaults PASSED [ 74%]
backend/test_phase7_reconciliation.py::TestCrossDocumentReconcilerUnit::test_single_document_no_conflict PASSED [ 75%]
backend/test_phase7_reconciliation.py::TestCrossDocumentReconcilerUnit::test_financial_turnover_conflict_below_threshold PASSED [ 76%]
backend/test_phase7_reconciliation.py::TestCrossDocumentReconcilerUnit::test_financial_turnover_consistent_both_above_threshold PASSED [ 77%]
backend/test_phase7_reconciliation.py::TestCrossDocumentReconcilerUnit::test_staffing_allocation_conflict_low_fte PASSED [ 78%]
backend/test_phase7_reconciliation.py::TestCrossDocumentReconcilerUnit::test_project_count_conflict_below_required PASSED [ 79%]
backend/test_phase7_reconciliation.py::TestCrossDocumentReconcilerUnit::test_debarment_conflict_clean_declaration_vs_annexure PASSED [ 80%]
backend/test_phase7_reconciliation.py::TestCrossDocumentReconcilerUnit::test_certificate_expiry_conflict PASSED [ 81%]
backend/test_phase7_reconciliation.py::TestCrossDocumentReconcilerUnit::test_minor_discrepancy_flagged_as_soft_uncertain PASSED [ 81%]
backend/test_phase7_reconciliation.py::TestCrossDocumentReconcilerUnit::test_no_annexure_no_conflict PASSED [ 82%]
backend/test_phase7_reconciliation.py::TestCrossDocumentReconcilerUnit::test_unrelated_numeric_values_no_false_positive PASSED [ 83%]
backend/test_phase7_reconciliation.py::TestDocumentTierClassification::test_financial_annexure_tier PASSED [ 84%]
backend/test_phase7_reconciliation.py::TestDocumentTierClassification::test_staffing_annexure_tier PASSED [ 85%]
backend/test_phase7_reconciliation.py::TestDocumentTierClassification::test_main_proposal_tier PASSED [ 86%]
backend/test_phase7_reconciliation.py::TestDocumentTierClassification::test_unknown_document_tier PASSED [ 87%]
backend/test_phase7_reconciliation.py::TestDocumentTierClassification::test_debarment_list_tier PASSED [ 88%]
backend/test_phase7_reconciliation.py::TestNumericParsers::test_parse_usd_millions_basic PASSED [ 89%]
backend/test_phase7_reconciliation.py::TestNumericParsers::test_parse_usd_millions_billion PASSED [ 90%]
backend/test_phase7_reconciliation.py::TestNumericParsers::test_parse_fte_percent PASSED [ 90%]
backend/test_phase7_reconciliation.py::TestNumericParsers::test_parse_count_projects PASSED [ 91%]
backend/test_phase7_reconciliation.py::TestNumericParsers::test_parse_usd_millions_no_match PASSED [ 92%]
backend/test_phase7_reconciliation.py::TestReconcileFromMetadata::test_turnover_conflict_metadata PASSED [ 93%]
backend/test_phase7_reconciliation.py::TestReconcileFromMetadata::test_staffing_conflict_metadata PASSED [ 94%]
backend/test_phase7_reconciliation.py::TestPhase7ClassifierIntegration::test_reconciliation_disabled_preserves_phase6_label PASSED [ 95%]
backend/test_phase7_reconciliation.py::TestPhase7ClassifierIntegration::test_reconciliation_enabled_overrides_label_on_hard_conflict PASSED [ 96%]
backend/test_phase7_reconciliation.py::TestPhase7ClassifierIntegration::test_reconciler_never_upgrades_non_compliant_to_compliant PASSED [ 97%]
backend/test_phase7_reconciliation.py::TestPhase7ClassifierIntegration::test_cross_doc_conflict_fields_persisted_in_database PASSED [ 98%]
backend/test_phase7_reconciliation.py::TestPhase7ClassifierIntegration::test_ambiguous_evidence_soft_conflict PASSED [ 99%]
backend/test_phase7_reconciliation.py::TestCrossDocumentEvaluation::test_evaluate_cross_doc_on_test_records PASSED [100%]

======================= 111 passed in 137.47s (0:02:17) =======================
```

---

## 8. Remaining Limitations & Scope Boundaries

1. **Scanned PDF Annexures (OCR):** The current text extractors rely on text-layer extraction (`pypdf` / `pdfplumber`). Scanned bitmap balance sheets or image-only certificates require optical character recognition (OCR), which is scoped for Phase 8+.
2. **Complex Multi-Year Tabular Balance Sheets:** The financial parser extracts primary scalar totals and average annual turnovers using regular expressions. Tabular year-by-year auditing (e.g., FY21, FY22, FY23 audited breakdown) remains a desirable enhancement for advanced financial modeling.
3. **Cross-Language Debarment Lists:** The entity and debarment matcher is optimized for English-language tenders and rosters. Regional language gazette notifications require multilingual NER models.
4. **Entity Fuzzy Matching:** The subsidiary debarment matcher identifies exact and prefix-based corporate entity names (e.g., "Alpha Cyber Infra Ltd" vs "Alpha Cloud & Cyber Systems Ltd"). Complex parent-subsidiary corporate tree lookups require external registrar integration.

---

## 9. Conclusion & Recommendation

The verification audit confirms that Phase 7 Multi-Document Evidence Reconciliation is mathematically sound, fully reproducible, strictly bounded by conservative procurement safeguards, and achieves a **100% resolution rate** on cross-document contradiction cases without regressions.

**Recommendation:** The Phase 7 implementation is verified and ready for formal approval.
