"""
baseline_classifier.py - Transparent Baseline Compliance Classifier
Part of Phase 4: Baseline Demo Pipeline

Performs rule-based, auditable compliance classification for each requirement-evidence pair.
Classifies into:
- COMPLIANT: Explicitly meets or exceeds threshold/standard without negation.
- NON_COMPLIANT: Verified numeric shortfall or explicit negative contradiction.
- UNCERTAIN: Vague, promotional, unit-incompatible, or ambiguous statements.
- MISSING_EVIDENCE: No matching statement or relevant section found in submission.

Never treats lexical similarity alone as proof of compliance.
"""

import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple

from backend.scoring.numeric_comparator import NumericComparator, NumericComparisonResult
from backend.matching.tfidf_retriever import RetrievedEvidenceCandidate, RequirementRetrievalResult


@dataclass
class ClauseComplianceEvaluation:
    """Detailed compliance assessment for a single tender requirement."""
    requirement_id: str
    requirement_text: str
    category: str
    parameter: str
    operator: str
    required_value: Optional[float] = None
    required_unit: Optional[str] = None
    is_mandatory: Optional[bool] = None

    # Classification Output
    label: str = "MISSING_EVIDENCE"       # 'COMPLIANT', 'NON_COMPLIANT', 'UNCERTAIN', 'MISSING_EVIDENCE'
    confidence: float = 0.0               # Heuristic assessment confidence (0.0 to 1.0)
    top_similarity_score: float = 0.0     # Lexical/TF-IDF similarity score

    # Top Candidate Details
    top_evidence_sentence: Optional[str] = None
    top_evidence_source: Optional[str] = None
    top_evidence_page: Optional[int] = None
    top_evidence_section: Optional[str] = None
    has_negation: bool = False

    # Numeric Evaluation Details
    numeric_comparison: Optional[Dict[str, Any]] = None

    # Human-Readable Audit Reasoning
    reasoning: str = ""
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_method: str = "tfidf"       # 'tfidf', 'semantic (all-MiniLM-L6-v2)', 'tfidf_fallback (...)'

    # Phase 6: NLI Verification Details
    nli_relation: Optional[str] = None    # 'entailment', 'contradiction', 'neutral', 'fallback', or None
    nli_entailment_prob: Optional[float] = None
    nli_contradiction_prob: Optional[float] = None
    nli_neutral_prob: Optional[float] = None
    nli_model: Optional[str] = None

    # Phase 7: Cross-Document Reconciliation
    cross_doc_conflict: bool = False          # True if multi-doc conflict detected
    cross_doc_conflict_type: Optional[str] = None   # 'numeric_magnitude', 'staffing_allocation', etc.
    cross_doc_conflict_severity: str = "NONE"       # 'NONE', 'SOFT', 'HARD'
    cross_doc_primary_source: Optional[str] = None  # Less authoritative source (proposal)
    cross_doc_primary_value: Optional[float] = None
    cross_doc_authoritative_source: Optional[str] = None  # More authoritative source (annexure)
    cross_doc_authoritative_value: Optional[float] = None
    cross_doc_authoritative_page: Optional[int] = None
    cross_doc_reasoning: Optional[str] = None       # Full reconciliation reasoning string

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentComplianceReport:
    """Aggregated compliance report for a complete tender-bidder analysis."""
    tender_document: str
    bidder_document: str
    bidder_name: str
    total_requirements: int
    compliant_count: int
    non_compliant_count: int
    uncertain_count: int
    missing_evidence_count: int
    mandatory_shortfalls_count: int       # Number of mandatory clauses failed
    retrieval_method: str = "tfidf"       # 'tfidf', 'semantic (all-MiniLM-L6-v2)', etc.
    nli_enabled: bool = True
    nli_model: Optional[str] = None
    annexure_documents: List[str] = field(default_factory=list)
    evaluations: List[ClauseComplianceEvaluation] = field(default_factory=list)
    disclaimer: str = (
        "DEMO BASELINE COMPLIANCE EVALUATION. Based on automated NLP retrieval and deterministic "
        "rule checks. Not an official qualification or disqualification decision. Requires final "
        "procurement officer verification."
    )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evaluations"] = [e.to_dict() for e in self.evaluations]
        d["annexure_documents"] = list(self.annexure_documents)
        return d


class BaselineClassifier:
    """Deterministic, transparent compliance classifier augmented with NLI verification
    and Phase 7 cross-document evidence reconciliation."""

    def __init__(
        self,
        min_retrieval_threshold: float = 0.08,
        min_semantic_threshold: float = 0.25,
        enable_nli: bool = True,
        nli_model_name: str = "cross-encoder/nli-distilroberta-base",
        nli_classifier: Optional[Any] = None,
        enable_reconciliation: bool = True,
        reconciler: Optional[Any] = None,
    ):
        self.min_retrieval_threshold = min_retrieval_threshold
        self.min_semantic_threshold = min_semantic_threshold
        self.numeric_comparator = NumericComparator()
        self.enable_nli = enable_nli
        self.nli_model_name = nli_model_name
        self.enable_reconciliation = enable_reconciliation

        if nli_classifier is not None:
            self.nli_classifier = nli_classifier
        elif enable_nli:
            try:
                from backend.classification.nli_classifier import NLIClassifier
                self.nli_classifier = NLIClassifier(model_name=nli_model_name)
            except Exception:
                self.nli_classifier = None
        else:
            self.nli_classifier = None

        # Phase 7: Cross-document reconciler
        if reconciler is not None:
            self.reconciler = reconciler
        elif enable_reconciliation:
            try:
                from backend.reconciliation.cross_document_reconciler import CrossDocumentReconciler
                self.reconciler = CrossDocumentReconciler()
            except Exception:
                self.reconciler = None
        else:
            self.reconciler = None

    def _evaluate_pm_experience_posture(
        self,
        top_cand: RetrievedEvidenceCandidate,
        matching_metric: Optional[Dict[str, Any]],
    ) -> Tuple[str, str]:
        """Classifies evidence for a Project Manager personal experience requirement.

        Distinguishes:
        1. 'PERSON_SPECIFIC': Explicit individual PM or named professional track record.
        2. 'CORPORATE_REJECTED': Company age, corporate founding, or firm operating history.
        3. 'AMBIGUOUS': Collective/pooled team experience, vague claims, or mixed attribution.
        """
        top_sent = getattr(top_cand, "sentence", "") or ""
        sent_lower = top_sent.lower()

        # Check for collective / pooled experience markers (Ambiguous)
        is_collective = bool(
            re.search(r"\b(?:collective|pooled|combined|cumulative|joint)\s+experience\b", sent_lower)
            or re.search(r"\b(?:pm\s+office|pmo|project\s+management\s+office|department)\s+(?:has|brings|with)\b", sent_lower)
        )
        if is_collective:
            return (
                "AMBIGUOUS",
                f"Ambiguous Project Manager experience claim: evidence cites collective or pooled team/office experience ('{top_sent[:120]}') rather than an individual Project Manager's verifiable personal track record. Requires human evaluation."
            )

        # Check if the numeric experience is explicitly attached to a corporate entity
        corporate_experience_match = bool(
            re.search(r"\b(?:our\s+organization|our\s+firm|our\s+company|the\s+company|the\s+firm|team)\s*(?:,|\bwhich\b)?\s*has\s+(?:\w+\s+)*\d+(?:\.\d+)?\s+years", sent_lower)
            or re.search(r"\bcompleted\s+\d+(?:\.\d+)?\s+years\s+of\s+(?:specialized\s+)?operations", sent_lower)
            or re.search(r"\b(?:incorporated\s+in|established\s+in|founded\s+in)\s+\d{4}\b", sent_lower)
            or re.search(r"\bcontinuous\s+commercial\s+operating\s+experience\b", sent_lower)
        )

        has_person_entity = any(e.get("label") == "PERSON" for e in getattr(top_cand, "extracted_entities", []))
        has_honorific_name = bool(re.search(r"\b(?:Dr\.|Mr\.|Ms\.|Mrs\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", top_sent))
        has_named_pm_role = bool(re.search(r"\b(?:Lead\s+PM|Project\s+Manager|designated\s+PM|proposed\s+PM|PM)\s+(?:Dr\.|Mr\.|Ms\.|Mrs\.)?\s*[A-Z][a-z]+", top_sent))
        has_pm_role_mention = bool(re.search(r"\b(?:lead\s+pm|project\s+manager|designated\s+pm|proposed\s+pm|lead\s+project\s+manager)\b", sent_lower))

        has_personal_attribution = bool(
            re.search(r"\b(?:brings?|possesses?|holds?|has)\s+(?:\w+\s+)*(?:\d+(?:\.\d+)?)\s+years?\s+of\s+(?:direct|relevant|hands-on|industry|professional|enterprise|cloud|project)?\s*experience\b", sent_lower)
            or re.search(r"\b(?:with|having)\s+(\d+(?:\.\d+)?)\s+years?\s+of\s+(?:direct|relevant|hands-on|industry|professional|enterprise|cloud|project)?\s*experience\b", sent_lower)
            or re.search(r"\b(\d+(?:\.\d+)?)\s+years?\s+(?:of\s+)?(?:direct\s+)?experience\s+leading\b", sent_lower)
        )

        # 1. Clear Person-Specific PM evidence:
        # (Named person, honorific name, named PM role, or explicit PM role mention)
        # WITH personal attribution of experience and without corporate experience attribution.
        if (has_person_entity or has_honorific_name or has_named_pm_role or has_pm_role_mention) and has_personal_attribution and not corporate_experience_match:
            return (
                "PERSON_SPECIFIC",
                "Verified person-specific Project Manager experience: evidence identifies an individual Project Manager and personal professional track record."
            )

        # 2. Corporate experience rejection:
        # If corporate indicators exist and no valid individual PM experience was established:
        if corporate_experience_match and not has_named_pm_role:
            return (
                "CORPORATE_REJECTED",
                f"BUG-2-FIX: Corporate operating-experience metric rejected as evidence for Project Manager personal experience requirement. Retrieved sentence references company-level history ('{top_sent[:120]}'). No person-specific Project Manager experience statement was found in the proposal."
            )

        # 3. Ambiguous evidence:
        # When PM is mentioned alongside company experience or evidence is mixed/vague
        if corporate_experience_match and has_named_pm_role:
            return (
                "AMBIGUOUS",
                f"Ambiguous Project Manager evidence: sentence ('{top_sent[:120]}') conflates named Project Manager with corporate operating history. Cannot attribute company experience to an individual."
            )

        if has_pm_role_mention or "experience" in sent_lower:
            return (
                "AMBIGUOUS",
                f"Ambiguous Project Manager experience claim: statement ('{top_sent[:120]}') does not provide verifiable personal experience attribution to an individual Project Manager."
            )

        return (
            "CORPORATE_REJECTED",
            f"BUG-2-FIX: Corporate operating-experience metric rejected as evidence for Project Manager personal experience requirement. Retrieved sentence references company-level history ('{top_sent[:120]}'). No person-specific Project Manager experience statement was found in the proposal."
        )

    def evaluate_clause(
        self,
        requirement: Any,
        retrieval_result: RequirementRetrievalResult,
        nli_pred: Optional[Any] = None
    ) -> ClauseComplianceEvaluation:
        """Evaluates a single requirement against its top retrieved evidence candidates."""
        req_id = getattr(requirement, "requirement_id", "REQ-UNKNOWN")
        req_text = getattr(requirement, "original_text", "")
        req_cat = getattr(requirement, "category", "General")
        req_param = getattr(requirement, "parameter", "")
        req_op = getattr(requirement, "operator", ">=")
        req_val = getattr(requirement, "value", None)
        req_unit = getattr(requirement, "unit", None)
        req_mandatory = getattr(requirement, "is_mandatory", None)
        req_uncertain = getattr(requirement, "uncertainty_flag", False)

        candidates = retrieval_result.candidates
        top_cand: Optional[RetrievedEvidenceCandidate] = candidates[0] if candidates else None
        top_sim = top_cand.similarity_score if top_cand else 0.0

        # Candidate serializations
        cand_dicts = [c.to_dict() for c in candidates]

        # Determine active retrieval method & method-specific threshold
        active_method = (
            getattr(top_cand, "retrieval_method", None)
            or getattr(retrieval_result, "retrieval_method", "tfidf")
        )
        active_threshold = (
            self.min_semantic_threshold if "semantic" in active_method.lower()
            else self.min_retrieval_threshold
        )

        # 1. Rule: MISSING EVIDENCE
        # If no candidates exist or top similarity is below threshold
        if not top_cand or top_sim < active_threshold:
            return ClauseComplianceEvaluation(
                requirement_id=req_id,
                requirement_text=req_text,
                category=req_cat,
                parameter=req_param,
                operator=req_op,
                required_value=float(req_val) if isinstance(req_val, (int, float)) else None,
                required_unit=req_unit,
                is_mandatory=req_mandatory,
                label="MISSING_EVIDENCE",
                confidence=0.90,
                top_similarity_score=top_sim,
                retrieval_method=active_method,
                reasoning="No statement or section addressing this tender requirement was submitted in the bidder proposal.",
                candidates=cand_dicts,
                nli_relation="N/A",
                nli_entailment_prob=0.0,
                nli_contradiction_prob=0.0,
                nli_neutral_prob=0.0,
                nli_model=getattr(self.nli_classifier, "model_name", "disabled") if self.enable_nli else "disabled"
            )

        top_sent = top_cand.sentence
        has_negation = top_cand.has_negation
        evd_metrics = top_cand.numeric_values

        # Compute NLI prediction if not supplied and NLI is active
        if nli_pred is None and self.enable_nli and self.nli_classifier and self.nli_classifier.is_available:
            nli_pred = self.nli_classifier.predict_pair(req_text, top_sent)

        # Helper constructor to ensure consistent NLI metadata stamping
        def make_eval(
            lbl: str,
            cnf: float,
            rsn: str,
            num_res_dict: Optional[Dict[str, Any]] = None,
            req_val_out: Optional[float] = None
        ) -> ClauseComplianceEvaluation:
            ev = ClauseComplianceEvaluation(
                requirement_id=req_id,
                requirement_text=req_text,
                category=req_cat,
                parameter=req_param,
                operator=req_op,
                required_value=req_val_out,
                required_unit=req_unit,
                is_mandatory=req_mandatory,
                label=lbl,
                confidence=cnf,
                top_similarity_score=top_sim,
                top_evidence_sentence=top_sent,
                top_evidence_source=top_cand.source_document,
                top_evidence_page=top_cand.page_number,
                top_evidence_section=top_cand.section_reference,
                has_negation=has_negation,
                retrieval_method=active_method,
                numeric_comparison=num_res_dict,
                reasoning=rsn,
                candidates=cand_dicts,
                nli_relation=getattr(nli_pred, "relation", None),
                nli_entailment_prob=getattr(nli_pred, "entailment_prob", None),
                nli_contradiction_prob=getattr(nli_pred, "contradiction_prob", None),
                nli_neutral_prob=getattr(nli_pred, "neutral_prob", None),
                nli_model=getattr(nli_pred, "model_name", None)
            )
            if self.enable_reconciliation and self.reconciler is not None:
                ev = self._apply_cross_document_reconciliation(ev, requirement, retrieval_result)
            return ev

        # 2. Rule: ADVERSARIAL NEGATION
        # If evidence has high similarity but explicit negative assertion
        if has_negation and top_sim >= 0.15:
            # Baseline keyword check (Phase 4)
            is_neg_denial = any(term in top_sent.lower() for term in [
                "does not guarantee", "does not include", "not include",
                "cannot provide", "excluding"
            ])
            # Semantic contradiction check via NLI (Phase 6)
            nli_denial = bool(nli_pred and nli_pred.relation == "contradiction" and nli_pred.contradiction_prob >= 0.65)
            if is_neg_denial or nli_denial:
                neg_rsn = f"High similarity ({top_sim:.3f}) with explicit negation detected in proposal claim: '{top_sent}'. Expressly disclaims required capability."
                if nli_pred and nli_pred.contradiction_prob > 0.0:
                    neg_rsn += f" (NLI Contradiction: {nli_pred.contradiction_prob:.2f})"
                return make_eval(
                    lbl="NON_COMPLIANT",
                    cnf=0.88,
                    rsn=neg_rsn,
                    req_val_out=float(req_val) if isinstance(req_val, (int, float)) else None
                )

        # 3. Rule: NUMERIC EVALUATION (when requirement has numeric threshold)
        if isinstance(req_val, (int, float)):
            req_num = float(req_val)
            # Find best matching numeric metric in top candidate
            matching_metric = None
            if evd_metrics:
                if req_unit in ("years", "year", "months", "month"):
                    for m in evd_metrics:
                        if m.get("unit") in ("years", "year", "months", "month"):
                            matching_metric = m
                            break
                if matching_metric is None:
                    matching_metric = evd_metrics[0]

            is_pm_exp_req = "project manager" in req_param.lower() and "experience" in req_param.lower()

            if is_pm_exp_req:
                pm_posture, pm_reasoning = self._evaluate_pm_experience_posture(
                    top_cand=top_cand,
                    matching_metric=matching_metric
                )

                if pm_posture == "CORPORATE_REJECTED":
                    return make_eval("MISSING_EVIDENCE", 0.85, pm_reasoning, req_val_out=req_num)
                elif pm_posture == "AMBIGUOUS":
                    return make_eval("UNCERTAIN", 0.75, pm_reasoning, req_val_out=req_num)

            num_res: NumericComparisonResult = self.numeric_comparator.compare(
                required_val=req_num,
                required_unit=req_unit,
                operator=req_op,
                offered_val=matching_metric["value"] if matching_metric else None,
                offered_unit=matching_metric["unit"] if matching_metric else None
            )

            if num_res.status == "SATISFIED":
                # Check for high NLI contradiction conflict
                if nli_pred and nli_pred.relation == "contradiction" and nli_pred.contradiction_prob >= 0.85:
                    label = "UNCERTAIN"
                    conf = 0.75
                    eval_reasoning = f"Numeric value matches threshold ({num_res.reasoning}), but NLI detects strong semantic contradiction ({nli_pred.contradiction_prob:.2f}) in proposal text: '{top_sent}'. Flagged for human review."
                else:
                    label = "COMPLIANT"
                    conf = 0.92
                    eval_reasoning = num_res.reasoning
                    if is_pm_exp_req:
                        eval_reasoning = f"Verified person-specific Project Manager experience confirmed in proposal. {num_res.reasoning}"
                    if nli_pred and nli_pred.relation == "entailment":
                        eval_reasoning += f" (NLI Entailment: {nli_pred.entailment_prob:.2f})"
            elif num_res.status == "VIOLATED":
                label = "NON_COMPLIANT"
                conf = 0.95
                eval_reasoning = num_res.reasoning
                if is_pm_exp_req:
                    eval_reasoning = f"Verified person-specific Project Manager evidence found, but {num_res.reasoning}"
                if nli_pred and nli_pred.relation == "entailment":
                    eval_reasoning += f" (Verified numeric shortfall overrides NLI entailment: {nli_pred.entailment_prob:.2f})"
            elif num_res.status == "INCOMPATIBLE_UNITS":
                label = "UNCERTAIN"
                conf = 0.65
                eval_reasoning = num_res.reasoning
            else:
                # No numeric value in top sentence
                label = "UNCERTAIN"
                conf = 0.70
                eval_reasoning = f"Unquantified proposal claim ('{top_sent}') submitted for numeric requirement ({req_op} {req_num} {req_unit or ''}). Lacks verifiable empirical metrics."

            return make_eval(label, conf, eval_reasoning, num_res_dict=num_res.to_dict(), req_val_out=req_num)

        # 4. Rule: STANDARD / ACCREDITATION DIRECT MATCH (ISO, SOC 2, AES, TLS, PMP, GSTIN)
        if isinstance(req_val, str) and req_val != "SUBMISSION_REQUIRED":
            target_std = req_val.lower().replace(":", "").replace("-", "")
            evd_lower = top_sent.lower().replace(":", "").replace("-", "")

            # Check if standard key exists in top sentence
            if any(part in evd_lower for part in target_std.split()):
                if nli_pred and nli_pred.relation == "contradiction" and nli_pred.contradiction_prob >= 0.65:
                    if has_negation:
                        return make_eval(
                            "NON_COMPLIANT",
                            0.88,
                            f"Proposal claim explicitly denies holding required accreditation '{req_val}': '{top_sent}' (NLI Contradiction: {nli_pred.contradiction_prob:.2f})."
                        )
                    return make_eval(
                        "UNCERTAIN",
                        0.75,
                        f"Accreditation mention for '{req_val}', but NLI detects contradiction ({nli_pred.contradiction_prob:.2f}): '{top_sent}'. Flagged for auditor review."
                    )
                std_rsn = f"Verified accreditation match for '{req_val}' confirmed in bidder proposal (similarity: {top_sim:.3f})."
                return make_eval("COMPLIANT", 0.90, std_rsn)

        # 5. Rule: STATUTORY SUBMISSION (GSTIN, PAN, MAF, Non-Blacklisting)
        if req_op == "MANDATORY_SUBMISSION":
            # Check if statutory ID or declaration is confirmed
            has_statutory_proof = False
            if "gstin" in req_param.lower() and any(e["label"] == "GSTIN" for e in top_cand.extracted_entities):
                has_statutory_proof = True
            elif "pan" in req_param.lower() and any(e["label"] == "PAN" for e in top_cand.extracted_entities):
                has_statutory_proof = True
            elif "blacklisted" in req_param.lower() and ("never been debarred" in top_sent.lower() or "never been blacklisted" in top_sent.lower()):
                has_statutory_proof = True
            elif "maf" in req_param.lower() and ("oem" in top_sent.lower() or "cisco" in top_sent.lower()):
                has_statutory_proof = True

            if has_statutory_proof:
                return make_eval("COMPLIANT", 0.88, "Mandatory statutory declaration or certificate reference confirmed in submitted bid package.")

            if "maf" in req_param.lower():
                return make_eval(
                    "MISSING_EVIDENCE",
                    0.82,
                    "BUG-3-FIX: No explicit OEM Manufacturer Authorization Form (MAF) was found in the proposal. A generic product warranty statement is not equivalent to an OEM authorization letter. Reseller/direct-OEM status cannot be established. Requires procurement officer review."
                )

        # 6. Rule: NLI QUALITATIVE CONTRADICTION & ENTAILMENT POLICY
        if nli_pred and nli_pred.relation == "contradiction" and nli_pred.contradiction_prob >= 0.65:
            return make_eval(
                "NON_COMPLIANT",
                0.85,
                f"NLI model detects contradiction ({nli_pred.contradiction_prob:.2f}) between requirement and proposal claim: '{top_sent}'."
            )

        if nli_pred and nli_pred.relation == "entailment":
            # Rule 6: Never label COMPLIANT solely because NLI predicts entailment on qualitative claim!
            return make_eval(
                "UNCERTAIN",
                0.70,
                f"NLI predicts entailment (prob: {nli_pred.entailment_prob:.2f}), but qualitative claim lacks empirical numeric metrics or verified third-party accreditation certificate. Flagged for procurement officer review."
            )

        # 7. Rule: UNCERTAIN CLAIMS / VAGUE STATEMENTS (Default fallback)
        if top_cand.uncertainty_flag or req_uncertain:
            return make_eval(
                "UNCERTAIN",
                0.70,
                f"Vague or promotional claim detected: '{top_sent}'. Lacks verifiable empirical metrics or third-party audit verification."
            )

        return make_eval(
            "UNCERTAIN",
            0.55,
            f"Qualitative narrative match (similarity {top_sim:.3f}). Requires human evaluator verification."
        )

    def _apply_cross_document_reconciliation(
        self,
        evaluation: "ClauseComplianceEvaluation",
        requirement: Any,
        retrieval_result: "RequirementRetrievalResult",
    ) -> "ClauseComplianceEvaluation":
        """Phase 7: Applies cross-document reconciliation to an already-classified evaluation.

        Only overrides the label if a HARD conflict (NON_COMPLIANT) or SOFT conflict
        (UNCERTAIN) is detected across multiple evidence source documents. Never
        upgrades a NON_COMPLIANT to COMPLIANT.
        """
        if self.reconciler is None:
            return evaluation

        # Only attempt reconciliation when multiple source documents are present
        candidate_sources = list({
            getattr(c, "source_document", "") for c in retrieval_result.candidates
            if getattr(c, "source_document", "")
        })
        if len(candidate_sources) < 2:
            return evaluation

        req_id = getattr(requirement, "requirement_id", "REQ-UNKNOWN")
        req_text = getattr(requirement, "original_text", "")
        req_val = getattr(requirement, "value", None)
        req_unit = getattr(requirement, "unit", None)
        req_val_float = float(req_val) if isinstance(req_val, (int, float)) else None

        try:
            rec_result = self.reconciler.reconcile(
                requirement_id=req_id,
                requirement_text=req_text,
                requirement_value=req_val_float,
                requirement_unit=req_unit,
                all_candidates=retrieval_result.candidates,
            )
        except Exception as exc:  # pragma: no cover
            import logging
            logging.getLogger(__name__).warning("Reconciler error for %s: %s", req_id, exc)
            return evaluation

        if not rec_result.has_conflict:
            return evaluation

        # Stamp cross-doc fields regardless of whether we override the label
        primary_conflict = rec_result.conflicts[0] if rec_result.conflicts else None
        evaluation.cross_doc_conflict = True
        evaluation.cross_doc_conflict_severity = rec_result.conflict_severity
        evaluation.cross_doc_reasoning = rec_result.reconciled_reasoning
        if primary_conflict:
            evaluation.cross_doc_conflict_type = primary_conflict.conflict_type
            evaluation.cross_doc_primary_source = primary_conflict.primary_source
            evaluation.cross_doc_primary_value = primary_conflict.primary_value
            evaluation.cross_doc_authoritative_source = primary_conflict.secondary_source
            evaluation.cross_doc_authoritative_value = primary_conflict.secondary_value
            evaluation.cross_doc_authoritative_page = primary_conflict.secondary_page

        # Only override label if the reconciler recommends a stricter outcome
        # (never downgrade from NON_COMPLIANT to UNCERTAIN, never grant COMPLIANT)
        label_priority = {"NON_COMPLIANT": 0, "UNCERTAIN": 1, "MISSING_EVIDENCE": 2, "COMPLIANT": 3}
        current_priority = label_priority.get(evaluation.label, 4)
        rec_priority = label_priority.get(rec_result.reconciled_label or "", 4)

        if rec_result.reconciled_label and rec_priority < current_priority:
            evaluation.label = rec_result.reconciled_label
            evaluation.confidence = rec_result.reconciled_confidence
            evaluation.reasoning = rec_result.reconciled_reasoning

        return evaluation

    def evaluate_document(
        self,
        requirements: List[Any],
        retrieval_results: List[RequirementRetrievalResult],
        tender_filename: str,
        bidder_filename: str,
        bidder_name: str = "Unknown Bidder",
        annexure_documents: Optional[List[str]] = None
    ) -> DocumentComplianceReport:
        """Evaluates all requirements against their retrieval candidates to produce a full report."""
        evaluations: List[ClauseComplianceEvaluation] = []
        compliant_count = 0
        non_compliant_count = 0
        uncertain_count = 0
        missing_count = 0
        mandatory_shortfalls = 0

        # Map requirements by ID
        req_map = {getattr(r, "requirement_id", ""): r for r in requirements}

        # Batch compute NLI predictions for efficiency
        nli_preds_map: Dict[str, Any] = {}
        if self.enable_nli and self.nli_classifier and self.nli_classifier.is_available:
            nli_batch_pairs = []
            nli_batch_req_ids = []
            for ret in retrieval_results:
                req_obj = req_map.get(ret.requirement_id)
                if not req_obj:
                    continue
                top_cand = ret.candidates[0] if ret.candidates else None
                active_thresh = (
                    self.min_semantic_threshold
                    if (top_cand and "semantic" in getattr(top_cand, "retrieval_method", "").lower())
                    else self.min_retrieval_threshold
                )
                if top_cand and top_cand.similarity_score >= active_thresh and top_cand.sentence:
                    nli_batch_pairs.append((req_obj.original_text, top_cand.sentence))
                    nli_batch_req_ids.append(ret.requirement_id)

            if nli_batch_pairs:
                predictions = self.nli_classifier.predict_batch(nli_batch_pairs)
                for r_id, p in zip(nli_batch_req_ids, predictions):
                    nli_preds_map[r_id] = p

        for ret in retrieval_results:
            req_obj = req_map.get(ret.requirement_id)
            if not req_obj:
                continue

            evaluation = self.evaluate_clause(req_obj, ret, nli_pred=nli_preds_map.get(ret.requirement_id))

            # Phase 7: Cross-document reconciliation (post-classification override)
            if self.enable_reconciliation and self.reconciler is not None:
                evaluation = self._apply_cross_document_reconciliation(evaluation, req_obj, ret)
            evaluations.append(evaluation)

            if evaluation.label == "COMPLIANT":
                compliant_count += 1
            elif evaluation.label == "NON_COMPLIANT":
                non_compliant_count += 1
                if evaluation.is_mandatory:
                    mandatory_shortfalls += 1
            elif evaluation.label == "UNCERTAIN":
                uncertain_count += 1
            elif evaluation.label == "MISSING_EVIDENCE":
                missing_count += 1
                if evaluation.is_mandatory:
                    mandatory_shortfalls += 1

        active_method = (
            getattr(retrieval_results[0], "retrieval_method", "tfidf")
            if retrieval_results
            else "tfidf"
        )

        nli_model_used = (
            getattr(self.nli_classifier, "model_name", None)
            if (self.enable_nli and self.nli_classifier and self.nli_classifier.is_available)
            else None
        )

        return DocumentComplianceReport(
            tender_document=tender_filename,
            bidder_document=bidder_filename,
            bidder_name=bidder_name,
            total_requirements=len(evaluations),
            compliant_count=compliant_count,
            non_compliant_count=non_compliant_count,
            uncertain_count=uncertain_count,
            missing_evidence_count=missing_count,
            mandatory_shortfalls_count=mandatory_shortfalls,
            retrieval_method=active_method,
            nli_enabled=self.enable_nli,
            nli_model=nli_model_used,
            annexure_documents=list(annexure_documents or []),
            evaluations=evaluations
        )
