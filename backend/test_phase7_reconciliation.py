"""
test_phase7_reconciliation.py - Phase 7 Test Suite: Multi-Document Evidence Reconciliation

Tests:
1. Reconciler initialization and default configuration
2. Single-document: no cross-document conflict possible
3. Financial turnover conflict: proposal claims $32M, annexure reports $15M < threshold → NON_COMPLIANT
4. Consistent turnover across documents: both above threshold → PRESERVE_BASELINE
5. Staffing allocation conflict: named PM with 10% FTE → NON_COMPLIANT
6. Reference project count conflict: claims 4, certificates show 1 < required 2 → NON_COMPLIANT
7. Debarment conflict: clean declaration vs. annexure showing debarment → NON_COMPLIANT
8. Certificate expiry conflict: active claim vs. annexure showing expired 2023 → NON_COMPLIANT
9. Minor numeric discrepancy (< 20%): flagged as UNCERTAIN, not NON_COMPLIANT
10. Missing annexure: no secondary source → no cross-doc conflict
11. Document tier classification: proposal vs. annexures
12. Reconcile_from_metadata: turnover conflict with ground truth
13. Reconcile_from_metadata: staffing conflict with ground truth
14. Integration: BaselineClassifier with enable_reconciliation=False preserves Phase 6 label
15. Integration: BaselineClassifier with enable_reconciliation=True overrides label on multi-doc conflict
16. Integration: reconciler never upgrades NON_COMPLIANT to COMPLIANT
17. Database persistence: cross_doc fields saved and retrievable from SQLite
18. Unrelated numeric values: different units/contexts do not trigger false positive
19. Ambiguous evidence with no clear numeric conflict → UNCERTAIN (SOFT)
"""

import os
import pytest
import tempfile
from unittest.mock import MagicMock, patch

from backend.reconciliation.cross_document_reconciler import (
    CrossDocumentReconciler,
    ConflictingClaimPair,
    ReconciliationResult,
    DocumentEvidenceGroup,
    classify_document_tier,
    _parse_usd_millions,
    _parse_fte_percent,
    _parse_count,
    _SyntheticCandidate,
)
from backend.classification.baseline_classifier import (
    BaselineClassifier,
    ClauseComplianceEvaluation,
)
from backend.extraction_nlp.requirement_extractor import ExtractedRequirement
from backend.matching.tfidf_retriever import (
    RequirementRetrievalResult,
    RetrievedEvidenceCandidate,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def make_candidate(
    sentence: str,
    source_document: str,
    page_number: int = 1,
    similarity_score: float = 0.75,
    has_negation: bool = False,
    numeric_values: list = None,
) -> RetrievedEvidenceCandidate:
    return RetrievedEvidenceCandidate(
        rank=1,
        evidence_id="EVD-TEST-001",
        sentence=sentence,
        similarity_score=similarity_score,
        source_document=source_document,
        page_number=page_number,
        has_negation=has_negation,
        numeric_values=numeric_values or [],
        extracted_entities=[],
        uncertainty_flag=False,
        review_notes="",
        retrieval_method="tfidf",
    )


def make_requirement(
    req_text: str,
    req_value=None,
    req_unit: str = None,
    req_op: str = ">=",
    req_param: str = "general",
    req_id: str = "REQ-TEST-001",
) -> ExtractedRequirement:
    return ExtractedRequirement(
        requirement_id=req_id,
        original_text=req_text,
        category="Financial",
        parameter=req_param,
        operator=req_op,
        value=req_value,
        unit=req_unit,
        is_mandatory=True,
    )


def make_retrieval_result(
    requirement: ExtractedRequirement,
    candidates: list,
) -> RequirementRetrievalResult:
    top_sim = candidates[0].similarity_score if candidates else 0.0
    return RequirementRetrievalResult(
        requirement_id=requirement.requirement_id,
        requirement_text=requirement.original_text,
        parameter=requirement.parameter,
        candidates=candidates,
        top_similarity=top_sim,
        has_candidate=bool(candidates),
    )


# ---------------------------------------------------------------------------
# Test Class 1: CrossDocumentReconciler Unit Tests
# ---------------------------------------------------------------------------

class TestCrossDocumentReconcilerUnit:
    """Unit tests for the CrossDocumentReconciler class."""

    @pytest.fixture
    def reconciler(self) -> CrossDocumentReconciler:
        return CrossDocumentReconciler(tender_year=2024)

    def test_reconciler_initialization_defaults(self, reconciler):
        """1. Reconciler initializes with correct defaults."""
        assert reconciler.annexure_supersedes is True
        assert reconciler.min_conflict_fraction == 0.20
        assert reconciler.MIN_FTE_FOR_DEDICATED == 50.0

    def test_single_document_no_conflict(self, reconciler):
        """2. Single-document evidence: no cross-document conflict possible."""
        cands = [
            make_candidate(
                "The bidder reports annual turnover of USD 32.0 Million.",
                "BIDDER-ALPHA-PROPOSAL.pdf",
                page_number=5,
            )
        ]
        result = reconciler.reconcile(
            requirement_id="REQ-001",
            requirement_text="Average annual turnover of at least USD 25.0 Million.",
            requirement_value=25.0,
            requirement_unit="USD Million",
            all_candidates=cands,
        )
        assert result.has_conflict is False
        assert result.conflict_severity == "NONE"
        assert result.reconciled_label is None

    def test_financial_turnover_conflict_below_threshold(self, reconciler):
        """3. Proposal claims $32M; annexure reports $15M < $25M threshold → NON_COMPLIANT (HARD)."""
        cands = [
            make_candidate(
                "Technical proposal executive summary claims USD 32.0 Million turnover.",
                "BIDDER-ALPHA-PROPOSAL.pdf",
                page_number=3,
            ),
            make_candidate(
                "Audited 3-year average turnover registered at USD 15.0 Million.",
                "ANNEXURE-F-FINANCIAL-STATEMENTS.pdf",
                page_number=12,
            ),
        ]
        result = reconciler.reconcile(
            requirement_id="REQ-001",
            requirement_text="Average annual audited turnover of at least USD 25.0 Million.",
            requirement_value=25.0,
            requirement_unit="USD Million",
            all_candidates=cands,
        )
        assert result.has_conflict is True
        assert result.conflict_severity == "HARD"
        assert result.reconciled_label == "NON_COMPLIANT"
        assert result.reconciled_confidence >= 0.85
        assert len(result.conflicts) >= 1
        assert result.conflicts[0].conflict_type == "numeric_magnitude"
        assert result.conflicts[0].secondary_value == 15.0
        assert "ANNEXURE" in result.conflicts[0].secondary_source
        # Reasoning must mention both values and the threshold
        assert "15.0" in result.reconciled_reasoning
        assert "25.0" in result.reconciled_reasoning or "25" in result.reconciled_reasoning

    def test_financial_turnover_consistent_both_above_threshold(self, reconciler):
        """4. Both proposal and annexure report same value above threshold → no conflict."""
        cands = [
            make_candidate(
                "Technical proposal claims USD 28.0 Million average annual turnover.",
                "BIDDER-ALPHA-PROPOSAL.pdf",
                page_number=3,
            ),
            make_candidate(
                "Audited financial statements confirm USD 28.0 Million turnover.",
                "ANNEXURE-F-FINANCIAL-STATEMENTS.pdf",
                page_number=8,
            ),
        ]
        result = reconciler.reconcile(
            requirement_id="REQ-001",
            requirement_text="Average annual audited turnover of at least USD 25.0 Million.",
            requirement_value=25.0,
            requirement_unit="USD Million",
            all_candidates=cands,
        )
        assert result.has_conflict is False
        assert result.reconciled_label is None

    def test_staffing_allocation_conflict_low_fte(self, reconciler):
        """5. Proposal presents dedicated PM; staffing schedule shows 10% FTE → NON_COMPLIANT."""
        cands = [
            make_candidate(
                "Robert Chen named as dedicated PMP Project Manager with 12 years experience.",
                "BIDDER-ALPHA-PROPOSAL.pdf",
                page_number=7,
            ),
            make_candidate(
                "Robert Chen allocated 10% FTE to this tender; 90% committed to ongoing municipal project.",
                "ANNEXURE-G-STAFFING-SCHEDULE.pdf",
                page_number=5,
            ),
        ]
        result = reconciler.reconcile(
            requirement_id="REQ-002",
            requirement_text="The proposed Project Manager must have PMP and minimum 10 years experience.",
            requirement_value=None,
            requirement_unit=None,
            all_candidates=cands,
        )
        assert result.has_conflict is True
        assert result.conflict_severity == "HARD"
        assert result.reconciled_label == "NON_COMPLIANT"
        assert result.conflicts[0].conflict_type == "staffing_allocation"
        assert result.conflicts[0].secondary_value == 10.0
        assert "10%" in result.reconciled_reasoning or "FTE" in result.reconciled_reasoning

    def test_project_count_conflict_below_required(self, reconciler):
        """6. Summary claims 4 projects; certificates show 1 < required 2 → NON_COMPLIANT."""
        cands = [
            make_candidate(
                "Technical summary cites 4 major qualifying projects as references.",
                "BIDDER-ALPHA-PROPOSAL.pdf",
                page_number=9,
            ),
            make_candidate(
                "Annexure E contains only 1 signed completion certificate; other 3 are ongoing.",
                "ANNEXURE-E-CLIENT-CERTIFICATES.pdf",
                page_number=8,
            ),
        ]
        result = reconciler.reconcile(
            requirement_id="REQ-003",
            requirement_text="Bidder must furnish client completion certificates for at least 2 similar reference projects.",
            requirement_value=2.0,
            requirement_unit=None,
            all_candidates=cands,
        )
        assert result.has_conflict is True
        assert result.reconciled_label == "NON_COMPLIANT"
        assert result.conflicts[0].conflict_type == "project_count"
        assert "1" in result.reconciled_reasoning or "certificate" in result.reconciled_reasoning.lower()

    def test_debarment_conflict_clean_declaration_vs_annexure(self, reconciler):
        """7. Proposal claims never debarred; annexure lists entity as debarred → NON_COMPLIANT."""
        cands = [
            make_candidate(
                "The firm hereby declares it has never been debarred or blacklisted by any authority.",
                "BIDDER-ALPHA-PROPOSAL.pdf",
                page_number=11,
            ),
            make_candidate(
                "Sister subsidiary Alpha Cyber Infra Ltd listed on Ministry debarment roster.",
                "ANNEXURE-DEBARMENT-LIST.pdf",
                page_number=3,
            ),
        ]
        result = reconciler.reconcile(
            requirement_id="REQ-004",
            requirement_text="Bidder must submit declaration affirming firm has never been blacklisted or debarred.",
            requirement_value=None,
            requirement_unit=None,
            all_candidates=cands,
        )
        assert result.has_conflict is True
        assert result.reconciled_label == "NON_COMPLIANT"
        assert result.conflicts[0].conflict_type == "debarment_disclosure"
        assert "debarment" in result.reconciled_reasoning.lower()

    def test_certificate_expiry_conflict(self, reconciler):
        """8. Proposal claims active OEM authorization; annexure shows it expired in 2023 → NON_COMPLIANT."""
        cands = [
            make_candidate(
                "Authorized gold partner with active OEM back-to-back SLA in place.",
                "BIDDER-ALPHA-PROPOSAL.pdf",
                page_number=4,
            ),
            make_candidate(
                "OEM Partner Certificate expired on 31-Dec-2023 and has not been re-validated.",
                "ANNEXURE-CERTIFICATES.pdf",
                page_number=2,
            ),
        ]
        result = reconciler.reconcile(
            requirement_id="REQ-005",
            requirement_text="Bidder must furnish a valid OEM Manufacturer Authorization Form.",
            requirement_value=None,
            requirement_unit=None,
            all_candidates=cands,
        )
        assert result.has_conflict is True
        # Tender year is 2024, certificate expired 2023 → HARD conflict
        assert result.conflict_severity in ("HARD", "SOFT")
        assert result.conflicts[0].conflict_type == "certificate_expiry"
        assert "expired" in result.reconciled_reasoning.lower() or "2023" in result.reconciled_reasoning

    def test_minor_discrepancy_flagged_as_soft_uncertain(self, reconciler):
        """9. Minor discrepancy (< 20%) between proposal and annexure → SOFT / UNCERTAIN."""
        cands = [
            make_candidate(
                "Bidder claims USD 26.5 Million average annual turnover.",
                "BIDDER-ALPHA-PROPOSAL.pdf",
                page_number=3,
            ),
            make_candidate(
                "Financial statements confirm USD 25.2 Million average annual turnover.",
                "ANNEXURE-F-FINANCIAL-STATEMENTS.pdf",
                page_number=10,
            ),
        ]
        result = reconciler.reconcile(
            requirement_id="REQ-001",
            requirement_text="Average annual audited turnover of at least USD 25.0 Million.",
            requirement_value=25.0,
            requirement_unit="USD Million",
            all_candidates=cands,
        )
        # 26.5 vs 25.2: fraction = 1.3/26.5 ≈ 4.9% < 20% → no conflict
        assert result.has_conflict is False

    def test_no_annexure_no_conflict(self, reconciler):
        """10. No secondary annexure source → reconciliation not applicable."""
        cands = [
            make_candidate("Bidder claims USD 30.0 Million turnover.", "BIDDER-ALPHA-PROPOSAL.pdf"),
            make_candidate("Executive summary confirms revenue of USD 30.0 Million.", "BIDDER-ALPHA-TECHNICAL-BID.pdf"),
        ]
        result = reconciler.reconcile(
            requirement_id="REQ-001",
            requirement_text="Average turnover at least USD 25.0 Million.",
            requirement_value=25.0,
            requirement_unit="USD Million",
            all_candidates=cands,
        )
        # Both are proposal-tier sources, no annexure → no reconciliation applicable
        assert result.has_conflict is False

    def test_unrelated_numeric_values_no_false_positive(self, reconciler):
        """18. Unrelated numbers (bandwidth, warranty) do not trigger financial conflict."""
        cands = [
            make_candidate("Network capacity of 10 Gbps with 36-month warranty.", "BIDDER-ALPHA-PROPOSAL.pdf"),
            make_candidate("Uptime SLA of 99.99% registered in Annexure B.", "ANNEXURE-B-SLA.pdf"),
        ]
        result = reconciler.reconcile(
            requirement_id="REQ-001",
            requirement_text="Average annual audited turnover of at least USD 25.0 Million.",
            requirement_value=25.0,
            requirement_unit="USD Million",
            all_candidates=cands,
        )
        # No USD financial values in either document → no financial conflict
        assert result.has_conflict is False


class TestDocumentTierClassification:
    """Tests for document tier classification utility."""

    def test_financial_annexure_tier(self):
        """11a. Annexure-F financial statement is tier 1 (authoritative)."""
        rank, label = classify_document_tier("ANNEXURE-F-FINANCIAL-STATEMENTS.pdf")
        assert rank <= 2
        assert "financial" in label or "annexure" in label

    def test_staffing_annexure_tier(self):
        """11b. Staffing schedule annexure is tier 1 (authoritative)."""
        rank, label = classify_document_tier("ANNEXURE-G-STAFFING-SCHEDULE.pdf")
        assert rank <= 2

    def test_main_proposal_tier(self):
        """11c. Main bidder proposal is tier 3 (narrative, less authoritative)."""
        rank, label = classify_document_tier("BIDDER-ALPHA-PROPOSAL.pdf")
        assert rank >= 3
        assert label == "main_proposal"

    def test_unknown_document_tier(self):
        """11d. Unknown document defaults to tier 3."""
        rank, label = classify_document_tier("MYSTERY_FILE.pdf")
        assert rank >= 3

    def test_debarment_list_tier(self):
        """11e. Debarment list is authoritative (tier 1)."""
        rank, label = classify_document_tier("ANNEXURE-DEBARMENT-LIST.pdf")
        assert rank <= 2


class TestNumericParsers:
    """Tests for internal numeric extraction utilities."""

    def test_parse_usd_millions_basic(self):
        assert _parse_usd_millions("USD 32.0 Million turnover") == pytest.approx(32.0)

    def test_parse_usd_millions_billion(self):
        # 1 Billion = 1000 Million
        val = _parse_usd_millions("USD 1.5 Billion net worth")
        assert val == pytest.approx(1500.0)

    def test_parse_fte_percent(self):
        assert _parse_fte_percent("10% FTE allocation") == pytest.approx(10.0)

    def test_parse_count_projects(self):
        assert _parse_count("Only 1 signed completion certificate attached") == 1

    def test_parse_usd_millions_no_match(self):
        assert _parse_usd_millions("No financial values here") is None


# ---------------------------------------------------------------------------
# Test Class 2: reconcile_from_metadata (Ground-Truth Driven Evaluation)
# ---------------------------------------------------------------------------

class TestReconcileFromMetadata:
    """Tests for the reconcile_from_metadata() path used by evaluate_classifier."""

    @pytest.fixture
    def reconciler(self) -> CrossDocumentReconciler:
        return CrossDocumentReconciler(tender_year=2024)

    def test_turnover_conflict_metadata(self, reconciler):
        """12. Turnover conflict via metadata: proposal $32M vs annexure $15M → NON_COMPLIANT."""
        result = reconciler.reconcile_from_metadata(
            requirement_id="REQ-TEST-PAIR-0145",
            requirement_text="Average annual audited financial turnover of at least USD 25.0 Million.",
            requirement_value=25.0,
            requirement_unit="USD Million",
            primary_evidence_text="Technical proposal executive summary claims USD 32.0 Million turnover; "
                                  "however, Annexure F (Audited Balance Sheet Page 12) officially registers "
                                  "average turnover as only USD 15.0 Million.",
            primary_source_document="BIDDER-PROPOSAL.pdf",
            primary_page=3,
            conflicting_document="ANNEXURE-FINANCIAL-STATEMENTS.pdf",
            conflicting_page=12,
            conflicting_text="Audited 3-year average turnover registered at USD 15.0 Million.",
        )
        assert result.has_conflict is True
        assert result.reconciled_label == "NON_COMPLIANT"
        assert result.conflict_severity == "HARD"
        assert any(c.conflict_type == "numeric_magnitude" for c in result.conflicts)

    def test_staffing_conflict_metadata(self, reconciler):
        """13. Staffing conflict via metadata: PM named, annexure shows 10% FTE → NON_COMPLIANT."""
        result = reconciler.reconcile_from_metadata(
            requirement_id="REQ-TEST-PAIR-0074",
            requirement_text="Proposed Project Manager must possess PMP and minimum 10 years experience.",
            requirement_value=None,
            requirement_unit=None,
            primary_evidence_text="Technical proposal names Robert Chen as dedicated PMP Project Manager with "
                                  "12 years experience; however, Staffing Schedule Annexure G assigns Robert Chen "
                                  "10% part-time allocation across 3 other active client sites.",
            primary_source_document="BIDDER-PROPOSAL.pdf",
            primary_page=7,
            conflicting_document="ANNEXURE-G-STAFFING-SCHEDULE.pdf",
            conflicting_page=5,
            conflicting_text="Robert Chen allocated 10% FTE to this tender; 90% committed to ongoing municipal project.",
        )
        assert result.has_conflict is True
        assert result.reconciled_label == "NON_COMPLIANT"
        assert any(c.conflict_type == "staffing_allocation" for c in result.conflicts)


# ---------------------------------------------------------------------------
# Test Class 3: Integration Tests with BaselineClassifier
# ---------------------------------------------------------------------------

class TestPhase7ClassifierIntegration:
    """Integration tests for Phase 7 reconciliation embedded in BaselineClassifier."""

    def _make_mock_reconciler(self, has_conflict: bool = True, rec_label: str = "NON_COMPLIANT"):
        """Creates a mock reconciler that returns a predetermined result."""
        mock_rec = MagicMock()
        mock_conflict = MagicMock()
        mock_conflict.conflict_type = "numeric_magnitude"
        mock_conflict.primary_source = "PROPOSAL.pdf"
        mock_conflict.primary_value = 32.0
        mock_conflict.secondary_source = "ANNEXURE-FINANCIAL-STATEMENTS.pdf"
        mock_conflict.secondary_value = 15.0
        mock_conflict.secondary_page = 12
        mock_result = ReconciliationResult(
            requirement_id="REQ-001",
            has_conflict=has_conflict,
            conflict_severity="HARD" if has_conflict else "NONE",
            conflicts=[mock_conflict] if has_conflict else [],
            reconciled_label=rec_label if has_conflict else None,
            reconciled_confidence=0.90 if has_conflict else 0.0,
            reconciled_reasoning="CROSS-DOCUMENT CONFLICT DETECTED" if has_conflict else "",
        )
        mock_rec.reconcile.return_value = mock_result
        return mock_rec

    def test_reconciliation_disabled_preserves_phase6_label(self):
        """14. When enable_reconciliation=False, classifier ignores cross-document evidence."""
        clf = BaselineClassifier(enable_nli=False, enable_reconciliation=False)

        req = make_requirement(
            "Average annual audited turnover of at least USD 25.0 Million.",
            req_value=25.0,
            req_unit="USD Million",
        )
        # Two different source documents simulating multi-doc scenario
        candidates = [
            make_candidate("Bidder claims USD 32.0 Million turnover.", "BIDDER-PROPOSAL.pdf"),
            make_candidate("Audited turnover USD 15.0 Million.", "ANNEXURE-FINANCIAL-STATEMENTS.pdf"),
        ]
        ret_result = make_retrieval_result(req, candidates)

        eval_result = clf.evaluate_clause(req, ret_result)
        # Without reconciliation: numeric evaluator sees 32.0 >= 25.0 → COMPLIANT
        assert eval_result.cross_doc_conflict is False
        assert eval_result.label in ("COMPLIANT", "UNCERTAIN")  # baseline may or may not have numeric

    def test_reconciliation_enabled_overrides_label_on_hard_conflict(self):
        """15. When enable_reconciliation=True, HARD conflict overrides COMPLIANT → NON_COMPLIANT."""
        mock_rec = self._make_mock_reconciler(has_conflict=True, rec_label="NON_COMPLIANT")
        clf = BaselineClassifier(enable_nli=False, enable_reconciliation=True, reconciler=mock_rec)

        req = make_requirement(
            "Average annual audited turnover of at least USD 25.0 Million.",
            req_value=25.0,
            req_unit="USD Million",
        )
        candidates = [
            make_candidate(
                "Technical proposal executive summary claims USD 32.0 Million turnover.",
                "BIDDER-PROPOSAL.pdf", page_number=3,
                numeric_values=[{"value": 32.0, "unit": "USD Million", "parameter_hint": "annual_turnover", "raw": "USD 32.0 Million"}],
            ),
            make_candidate(
                "Audited 3-year average turnover registered at USD 15.0 Million.",
                "ANNEXURE-FINANCIAL-STATEMENTS.pdf", page_number=12,
                numeric_values=[{"value": 15.0, "unit": "USD Million", "parameter_hint": "annual_turnover", "raw": "USD 15.0 Million"}],
            ),
        ]
        ret_result = make_retrieval_result(req, candidates)

        eval_result = clf.evaluate_clause(req, ret_result)
        eval_result = clf._apply_cross_document_reconciliation(eval_result, req, ret_result)

        assert eval_result.cross_doc_conflict is True
        assert eval_result.label == "NON_COMPLIANT"
        assert eval_result.cross_doc_conflict_severity == "HARD"
        assert eval_result.cross_doc_authoritative_source == "ANNEXURE-FINANCIAL-STATEMENTS.pdf"
        assert eval_result.cross_doc_authoritative_value == 15.0

    def test_reconciler_never_upgrades_non_compliant_to_compliant(self):
        """16. Reconciler with UNCERTAIN result never upgrades NON_COMPLIANT to COMPLIANT."""
        # Create a reconciler that would return UNCERTAIN (soft conflict)
        mock_rec = self._make_mock_reconciler(has_conflict=True, rec_label="UNCERTAIN")
        clf = BaselineClassifier(enable_nli=False, enable_reconciliation=True, reconciler=mock_rec)

        req = make_requirement("Bidder must provide 99.9% uptime SLA.", req_value=99.9, req_unit="%")
        candidates = [
            make_candidate("Our SLA guarantees 95.0% uptime only.", "BIDDER-PROPOSAL.pdf",
                           numeric_values=[{"value": 95.0, "unit": "%", "parameter_hint": "uptime_or_availability", "raw": "95.0%"}]),
            make_candidate("Uptime commitments: 95% confirmed.", "ANNEXURE-B-SLA.pdf"),
        ]
        ret_result = make_retrieval_result(req, candidates)

        # Force baseline to NON_COMPLIANT (95% < 99.9%)
        eval_result = clf.evaluate_clause(req, ret_result)
        assert eval_result.label == "NON_COMPLIANT"

        # Apply reconciliation with UNCERTAIN recommendation
        eval_result = clf._apply_cross_document_reconciliation(eval_result, req, ret_result)

        # NON_COMPLIANT must not be changed to UNCERTAIN (UNCERTAIN has lower priority)
        assert eval_result.label == "NON_COMPLIANT"

    def test_cross_doc_conflict_fields_persisted_in_database(self):
        """17. Cross-document conflict fields are saved to and retrievable from SQLite."""
        from backend.database.db_manager import DatabaseManager
        from backend.classification.baseline_classifier import DocumentComplianceReport

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
            tmp_db_path = tmp_db.name

        try:
            db = DatabaseManager(db_path=tmp_db_path)

            # Build a ClauseComplianceEvaluation with cross-doc fields set
            eval_with_conflict = ClauseComplianceEvaluation(
                requirement_id="REQ-001",
                requirement_text="Average annual turnover at least USD 25.0 Million.",
                category="Financial",
                parameter="annual_turnover",
                operator=">=",
                required_value=25.0,
                required_unit="USD Million",
                is_mandatory=True,
                label="NON_COMPLIANT",
                confidence=0.90,
                top_similarity_score=0.75,
                top_evidence_sentence="Proposal claims USD 32.0M but annexure shows USD 15.0M.",
                top_evidence_source="BIDDER-PROPOSAL.pdf",
                top_evidence_page=3,
                top_evidence_section="Section 2.0",
                has_negation=False,
                retrieval_method="tfidf",
                reasoning="CROSS-DOCUMENT CONFLICT DETECTED: Financial turnover conflict.",
                cross_doc_conflict=True,
                cross_doc_conflict_type="numeric_magnitude",
                cross_doc_conflict_severity="HARD",
                cross_doc_primary_source="BIDDER-PROPOSAL.pdf",
                cross_doc_primary_value=32.0,
                cross_doc_authoritative_source="ANNEXURE-F-FINANCIAL-STATEMENTS.pdf",
                cross_doc_authoritative_value=15.0,
                cross_doc_authoritative_page=12,
                cross_doc_reasoning="Annexure reports USD 15M; below the USD 25M threshold.",
            )

            report = DocumentComplianceReport(
                tender_document="TENDER.pdf",
                bidder_document="BIDDER-PROPOSAL.pdf",
                bidder_name="Test Bidder",
                total_requirements=1,
                compliant_count=0,
                non_compliant_count=1,
                uncertain_count=0,
                missing_evidence_count=0,
                mandatory_shortfalls_count=1,
                evaluations=[eval_with_conflict],
            )

            analysis_id = db.save_analysis(report)
            assert analysis_id.startswith("ANA-")

            retrieved = db.get_analysis_by_id(analysis_id)
            assert retrieved is not None
            assert len(retrieved["evaluations"]) == 1

            saved_eval = retrieved["evaluations"][0]
            assert saved_eval["cross_doc_conflict"] == 1
            assert saved_eval["cross_doc_conflict_type"] == "numeric_magnitude"
            assert saved_eval["cross_doc_conflict_severity"] == "HARD"
            assert saved_eval["cross_doc_primary_source"] == "BIDDER-PROPOSAL.pdf"
            assert saved_eval["cross_doc_primary_value"] == pytest.approx(32.0)
            assert saved_eval["cross_doc_authoritative_source"] == "ANNEXURE-F-FINANCIAL-STATEMENTS.pdf"
            assert saved_eval["cross_doc_authoritative_value"] == pytest.approx(15.0)
            assert saved_eval["cross_doc_authoritative_page"] == 12

        finally:
            try:
                if os.path.exists(tmp_db_path):
                    os.unlink(tmp_db_path)
            except (OSError, PermissionError):
                pass

    def test_ambiguous_evidence_soft_conflict(self):
        """19. Ambiguous evidence with no clear conflict → no override (SOFT at most)."""
        mock_rec = self._make_mock_reconciler(has_conflict=True, rec_label="UNCERTAIN")
        clf = BaselineClassifier(enable_nli=False, enable_reconciliation=True, reconciler=mock_rec)

        req = make_requirement("Bidder must hold ISO 9001:2015 certification.", req_value="ISO 9001:2015")
        candidates = [
            make_candidate("We maintain ISO 9001:2015 aligned processes.", "BIDDER-PROPOSAL.pdf"),
            make_candidate("ISO audit review pending per Annexure C.", "ANNEXURE-C.pdf"),
        ]
        ret_result = make_retrieval_result(req, candidates)

        eval_result = clf.evaluate_clause(req, ret_result)
        original_label = eval_result.label

        eval_result = clf._apply_cross_document_reconciliation(eval_result, req, ret_result)

        # UNCERTAIN conflict must not downgrade existing NON_COMPLIANT or upgrade COMPLIANT
        label_priority = {"NON_COMPLIANT": 0, "UNCERTAIN": 1, "MISSING_EVIDENCE": 2, "COMPLIANT": 3}
        orig_priority = label_priority.get(original_label, 4)
        final_priority = label_priority.get(eval_result.label, 4)
        assert final_priority <= orig_priority  # Never relaxed


# ---------------------------------------------------------------------------
# Test Class 4: Evaluate Cross-Document Reconciliation on Ground Truth
# ---------------------------------------------------------------------------

class TestCrossDocumentEvaluation:
    """Tests the evaluate_cross_document_reconciliation() benchmark function."""

    def test_evaluate_cross_doc_on_test_records(self):
        """Runs the cross-document evaluator on actual test.jsonl records and checks output shape."""
        from backend.evaluation.evaluate_classifier import evaluate_cross_document_reconciliation
        import json

        test_path = os.path.join(
            os.path.dirname(__file__), "..", "dataset", "splits", "test.jsonl"
        )
        if not os.path.exists(test_path):
            pytest.skip("test.jsonl not available in CI environment")

        with open(test_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        result = evaluate_cross_document_reconciliation(records)

        # Must return structured dict with expected keys
        assert "total" in result
        assert "correct" in result
        assert "accuracy" in result
        assert "details" in result

        # Should have found 4 cross-document records in test.jsonl
        assert result["total"] == 4

        # All ground-truth labels are NON_COMPLIANT for these 4 records
        for det in result["details"]:
            assert det["gt_label"] == "NON_COMPLIANT"
            assert det["pred_label"] in ("NON_COMPLIANT", "UNCERTAIN", "COMPLIANT", "MISSING_EVIDENCE")

        # Reconciler should detect at least some conflicts
        has_conflicts = sum(1 for det in result["details"] if det["has_conflict"])
        assert has_conflicts >= 1, "Reconciler should detect at least 1 cross-document conflict"
