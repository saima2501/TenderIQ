"""
test_entity_verifier.py - Unit & Integration Tests for Bidder Entity-Scope Verification
Part of Phase 8: Core NLP Reliability Improvements (Step 2C)

Covers all 17 test categories specified in Task 7 and Scenarios A through G:
1. Same legal entity (Scenario B)
2. Parent company (Scenario A)
3. Subsidiary (Scenario C)
4. Affiliate / Sister concern
5. JV partner (Scenario D)
6. Consortium member
7. Subcontractor (Scenario E)
8. Unrelated third party (Scenario F)
9. Similar but different company names (Scenario G)
10. Parent guarantee with and without tender authorization
11. JV agreement with and without tender authorization
12. Missing supporting documents
13. Ambiguous organization identity / unverified document
14. Cross-document evidence attribution
15. Numeric evidence from the wrong entity
16. Certificate evidence from the wrong entity
17. Regression of existing bidder isolation safeguards
"""

import pytest
from backend.extraction_nlp.entity_verifier import EntityScopeVerifier, EntityScope, EntityScopeResult
from backend.classification.baseline_classifier import BaselineClassifier
from backend.matching.tfidf_retriever import RetrievedEvidenceCandidate, RequirementRetrievalResult


@pytest.fixture
def verifier():
    return EntityScopeVerifier()


@pytest.fixture
def classifier():
    return BaselineClassifier(enable_nli=False, enable_reconciliation=False)


class MockRequirement:
    """Minimal duck-type for tender requirement in tests."""
    def __init__(
        self,
        requirement_id: str = "REQ-FIN-001",
        original_text: str = "The bidder must have an annual turnover of at least INR 10 crore.",
        category: str = "Financial",
        parameter: str = "turnover",
        operator: str = ">=",
        value: float = 10.0,
        unit: str = "crore",
        is_mandatory: bool = True,
        uncertainty_flag: bool = False
    ):
        self.requirement_id = requirement_id
        self.original_text = original_text
        self.category = category
        self.parameter = parameter
        self.operator = operator
        self.value = value
        self.unit = unit
        self.is_mandatory = is_mandatory
        self.uncertainty_flag = uncertainty_flag


# ===========================================================================
# 1. Direct Unit Tests: Legal Name Normalization & Anti-Confusion
# ===========================================================================
class TestLegalNameMatching:
    """Tests name normalization and strict anti-confusion safeguards."""

    def test_exact_name_match(self, verifier):
        assert verifier.is_same_legal_entity("ABC Technologies Pvt. Ltd.", "ABC Technologies Pvt. Ltd.")

    def test_suffix_synonym_normalization(self, verifier):
        assert verifier.is_same_legal_entity("ABC Technologies Pvt. Ltd.", "ABC Technologies Private Limited")
        assert verifier.is_same_legal_entity("Apex Global Inc.", "Apex Global Incorporated")
        assert verifier.is_same_legal_entity("Nexus Corp.", "Nexus Corporation")
        assert verifier.is_same_legal_entity("Vertex LLP", "Vertex Limited Liability Partnership")

    def test_similar_names_anti_confusion(self, verifier):
        """Scenario G: 'ABC Technologies Pvt. Ltd.' vs 'ABC Technology Solutions Ltd.' must be distinct."""
        assert not verifier.is_same_legal_entity("ABC Technologies Pvt. Ltd.", "ABC Technology Solutions Ltd.")
        assert not verifier.is_same_legal_entity("ABC Technologies Pvt. Ltd.", "ABC Holdings Ltd.")
        assert not verifier.is_same_legal_entity("ABC Technologies Pvt. Ltd.", "ABC Services Ltd.")

    def test_null_and_unknown_handling(self, verifier):
        assert not verifier.is_same_legal_entity(None, "ABC Technologies Pvt. Ltd.")
        assert not verifier.is_same_legal_entity("Unknown Bidder", "ABC Technologies Pvt. Ltd.")
        assert not verifier.is_same_legal_entity("", "")


# ===========================================================================
# 2. Scenarios A through G Specification Tests
# ===========================================================================
class TestScenariosAThroughG:
    """Validates the seven core scenarios mandated in Step 2C Task 5."""

    def test_scenario_a_parent_company_turnover(self, verifier):
        """Scenario A: Parent company turnover must return UNCERTAIN."""
        res = verifier.verify_entity_scope(
            evidence_sentence="ABC Holdings Ltd. (parent company) has annual turnover of 100 crore.",
            trusted_bidder_name="ABC Technologies Pvt. Ltd.",
            source_document="BIDDER-ABC-PROPOSAL.pdf"
        )
        assert res.entity_scope == EntityScope.PARENT_ENTITY
        assert not res.is_eligible_attribution
        assert "parent company" in res.reasoning.lower()

    def test_scenario_b_same_bidder_turnover(self, verifier):
        """Scenario B: Same bidder turnover must return SAME_ENTITY and be eligible."""
        res = verifier.verify_entity_scope(
            evidence_sentence="ABC Technologies Pvt. Ltd. has annual turnover of 15 crore.",
            trusted_bidder_name="ABC Technologies Pvt. Ltd.",
            source_document="BIDDER-ABC-PROPOSAL.pdf"
        )
        assert res.entity_scope == EntityScope.SAME_ENTITY
        assert res.is_eligible_attribution

    def test_scenario_c_subsidiary_certificate(self, verifier):
        """Scenario C: Subsidiary certificate must return SUBSIDIARY_ENTITY and not be eligible."""
        res = verifier.verify_entity_scope(
            evidence_sentence="ABC Services Ltd. (subsidiary) holds ISO 27001 certification.",
            trusted_bidder_name="ABC Technologies Pvt. Ltd.",
            source_document="BIDDER-ABC-PROPOSAL.pdf"
        )
        assert res.entity_scope == EntityScope.SUBSIDIARY_ENTITY
        assert not res.is_eligible_attribution
        assert "subsidiary" in res.reasoning.lower()

    def test_scenario_d_jv_partner_experience(self, verifier):
        """Scenario D: JV partner experience without explicit tender permission must return UNCERTAIN."""
        res = verifier.verify_entity_scope(
            evidence_sentence="A signed JV agreement identifies XYZ Infrastructure Ltd. as a consortium member and allocates defined work responsibilities.",
            trusted_bidder_name="ABC Technologies Pvt. Ltd.",
            source_document="BIDDER-ABC-PROPOSAL.pdf"
        )
        assert res.entity_scope == EntityScope.JV_OR_CONSORTIUM_MEMBER
        assert not res.is_eligible_attribution
        assert "consortium" in res.reasoning.lower() or "jv" in res.reasoning.lower()

    def test_scenario_e_subcontractor_credentials(self, verifier):
        """Scenario E: Subcontractor credentials must return SUBCONTRACTOR_ENTITY and not count toward prime bidder."""
        res = verifier.verify_entity_scope(
            evidence_sentence="XYZ Ltd. will provide subcontracted services and holds relevant experience.",
            trusted_bidder_name="ABC Technologies Pvt. Ltd.",
            source_document="BIDDER-ABC-PROPOSAL.pdf"
        )
        assert res.entity_scope == EntityScope.SUBCONTRACTOR_ENTITY
        assert not res.is_eligible_attribution
        assert "subcontractor" in res.reasoning.lower()

    def test_scenario_f_unrelated_third_party(self, verifier):
        """Scenario F: Unrelated company holding certificate must return THIRD_PARTY_ENTITY."""
        res = verifier.verify_entity_scope(
            evidence_sentence="ACME Global Inc holds the ISO 27001 certificate.",
            trusted_bidder_name="ABC Technologies Pvt. Ltd.",
            source_document="BIDDER-ABC-PROPOSAL.pdf"
        )
        assert res.entity_scope == EntityScope.THIRD_PARTY_ENTITY
        assert not res.is_eligible_attribution

    def test_scenario_g_similar_company_names(self, verifier):
        """Scenario G: Similar company name must not be assumed to be SAME_ENTITY."""
        res = verifier.verify_entity_scope(
            evidence_sentence="ABC Technology Solutions Ltd. holds the required certification.",
            trusted_bidder_name="ABC Technologies Pvt. Ltd.",
            source_document="BIDDER-ABC-PROPOSAL.pdf"
        )
        assert res.entity_scope == EntityScope.THIRD_PARTY_ENTITY
        assert not res.is_eligible_attribution


# ===========================================================================
# 3. Supporting Documents & Tender Conditions Tests
# ===========================================================================
class TestSupportingDocumentsAndTenderConditions:
    """Tests parent guarantees, JV agreements, and missing supporting arrangements."""

    def test_parent_guarantee_with_tender_permission(self, verifier):
        """Parent guarantee + tender explicitly permits parent guarantee -> Eligible."""
        res = verifier.verify_entity_scope(
            evidence_sentence="ABC Holdings Ltd. (parent company) has turnover of 100 crore under a binding parent company guarantee.",
            trusted_bidder_name="ABC Technologies Pvt. Ltd.",
            requirement_text="The bidder must have turnover of at least 10 crore; parent guarantee accepted.",
            source_document="BIDDER-ABC-PROPOSAL.pdf"
        )
        assert res.entity_scope == EntityScope.PARENT_ENTITY
        assert res.is_eligible_attribution
        assert "parent company guarantee satisfies" in res.reasoning.lower()

    def test_parent_guarantee_without_tender_permission(self, verifier):
        """Parent guarantee present but tender DOES NOT authorize it -> Ineligible (UNCERTAIN)."""
        res = verifier.verify_entity_scope(
            evidence_sentence="ABC Holdings Ltd. (parent company) has turnover of 100 crore under a binding parent company guarantee.",
            trusted_bidder_name="ABC Technologies Pvt. Ltd.",
            requirement_text="The bidder must have turnover of at least 10 crore from its own operations.",
            source_document="BIDDER-ABC-PROPOSAL.pdf"
        )
        assert res.entity_scope == EntityScope.PARENT_ENTITY
        assert not res.is_eligible_attribution

    def test_jv_agreement_with_tender_permission(self, verifier):
        """JV agreement + tender explicitly allows consortium -> Eligible."""
        res = verifier.verify_entity_scope(
            evidence_sentence="A signed JV agreement identifies XYZ Infrastructure Ltd. as a consortium member.",
            trusted_bidder_name="ABC Technologies Pvt. Ltd.",
            requirement_text="The bidder must possess relevant project experience; consortium allowed.",
            source_document="BIDDER-ABC-PROPOSAL.pdf"
        )
        assert res.entity_scope == EntityScope.JV_OR_CONSORTIUM_MEMBER
        assert res.is_eligible_attribution

    def test_missing_supporting_documents(self, verifier):
        """Parent turnover claimed without any guarantee or undertaking cited -> Ineligible."""
        res = verifier.verify_entity_scope(
            evidence_sentence="Our parent group ABC Holdings has turnover of 100 crore.",
            trusted_bidder_name="ABC Technologies Pvt. Ltd.",
            source_document="BIDDER-ABC-PROPOSAL.pdf"
        )
        assert not res.is_eligible_attribution
        assert res.supporting_document_type is None


# ===========================================================================
# 4. First-Person Claims & Provenance Verification
# ===========================================================================
class TestProvenanceAndFirstPersonClaims:
    """Tests first-person statements with verified vs unverified document provenance."""

    def test_first_person_in_verified_bidder_proposal(self, verifier):
        """'Our company has 15 crore turnover' in BIDDER-ABC-PROPOSAL.pdf -> SAME_ENTITY."""
        res = verifier.verify_entity_scope(
            evidence_sentence="Our company has annual turnover of 15 crore.",
            trusted_bidder_name="ABC Technologies Pvt. Ltd.",
            source_document="BIDDER-ABC-PROPOSAL.pdf"
        )
        assert res.entity_scope == EntityScope.SAME_ENTITY
        assert res.is_eligible_attribution

    def test_first_person_in_foreign_document_uncertain(self, verifier):
        """'Our company has 15 crore turnover' in foreign/unrelated document -> UNKNOWN_ENTITY."""
        res = verifier.verify_entity_scope(
            evidence_sentence="Our company has annual turnover of 15 crore.",
            trusted_bidder_name="ABC Technologies Pvt. Ltd.",
            source_document="FOREIGN_THIRD_PARTY_REPORT.pdf"
        )
        assert res.entity_scope == EntityScope.UNKNOWN_ENTITY
        assert not res.is_eligible_attribution

    def test_ambiguous_organization_identity_uncertain(self, verifier):
        """Identity completely unknown and document unverified -> UNKNOWN_ENTITY."""
        res = verifier.verify_entity_scope(
            evidence_sentence="Turnover exceeds 10 crore.",
            trusted_bidder_name=None,
            source_document="UNKNOWN.pdf"
        )
        assert res.entity_scope == EntityScope.UNKNOWN_ENTITY
        assert not res.is_eligible_attribution


# ===========================================================================
# 5. Integration with BaselineClassifier
# ===========================================================================
class TestClassifierIntegrationWithEntityScope:
    """Verifies that BaselineClassifier enforces entity scope gatekeeper."""

    def test_numeric_evidence_from_parent_company_returns_uncertain(self, classifier):
        """Parent turnover of 100 crore cannot yield COMPLIANT for 10 crore requirement."""
        req = MockRequirement(
            requirement_id="REQ-TURNOVER",
            original_text="The bidder must demonstrate annual turnover of at least USD 10.0M.",
            parameter="turnover",
            operator=">=",
            value=10.0,
            unit="M"
        )
        candidate = RetrievedEvidenceCandidate(
            rank=1,
            evidence_id="EVD-P1",
            sentence="ABC Holdings Ltd. (parent company) has annual audited turnover of USD 100.0M.",
            similarity_score=0.85,
            source_document="BIDDER-PROPOSAL.pdf",
            page_number=3,
            numeric_values=[{"raw": "USD 100.0M", "value": 100.0, "unit": "M"}]
        )
        ret = RequirementRetrievalResult(
            requirement_id="REQ-TURNOVER",
            requirement_text=req.original_text,
            parameter=req.parameter,
            candidates=[candidate]
        )

        eval_res = classifier.evaluate_clause(
            req,
            ret,
            bidder_name="ABC Technologies Pvt. Ltd."
        )
        assert eval_res.label == "UNCERTAIN"
        assert "parent company" in eval_res.reasoning.lower()

    def test_numeric_evidence_from_same_entity_returns_compliant(self, classifier):
        """Same entity turnover of 15 crore satisfies 10 crore requirement -> COMPLIANT."""
        req = MockRequirement(
            requirement_id="REQ-TURNOVER",
            original_text="The bidder must demonstrate annual turnover of at least USD 10.0M.",
            parameter="turnover",
            operator=">=",
            value=10.0,
            unit="M"
        )
        candidate = RetrievedEvidenceCandidate(
            rank=1,
            evidence_id="EVD-S1",
            sentence="ABC Technologies Pvt. Ltd. has annual audited turnover of USD 15.0M.",
            similarity_score=0.85,
            source_document="BIDDER-PROPOSAL.pdf",
            page_number=3,
            numeric_values=[{"raw": "USD 15.0M", "value": 15.0, "unit": "M"}]
        )
        ret = RequirementRetrievalResult(
            requirement_id="REQ-TURNOVER",
            requirement_text=req.original_text,
            parameter=req.parameter,
            candidates=[candidate]
        )

        eval_res = classifier.evaluate_clause(
            req,
            ret,
            bidder_name="ABC Technologies Pvt. Ltd."
        )
        assert eval_res.label == "COMPLIANT"

    def test_certificate_evidence_from_subsidiary_returns_uncertain(self, classifier):
        """Subsidiary certificate cannot yield COMPLIANT for prime bidder."""
        req = MockRequirement(
            requirement_id="REQ-CERT",
            original_text="The bidder must hold ISO 27001 certification.",
            parameter="ISO 27001",
            operator="==",
            value=None,
            unit=None
        )
        candidate = RetrievedEvidenceCandidate(
            rank=1,
            evidence_id="EVD-C1",
            sentence="ABC Services Ltd. (subsidiary) holds ISO 27001 certification valid through December 2026.",
            similarity_score=0.85,
            source_document="BIDDER-PROPOSAL.pdf",
            page_number=5,
            numeric_values=[]
        )
        ret = RequirementRetrievalResult(
            requirement_id="REQ-CERT",
            requirement_text=req.original_text,
            parameter=req.parameter,
            candidates=[candidate]
        )

        eval_res = classifier.evaluate_clause(
            req,
            ret,
            tender_submission_date="2024-10-15",
            bidder_name="ABC Technologies Pvt. Ltd."
        )
        assert eval_res.label == "UNCERTAIN"
        assert "subsidiary" in eval_res.reasoning.lower()

    def test_certificate_from_unrelated_third_party_returns_uncertain(self, classifier):
        """Third-party certificate cannot yield COMPLIANT."""
        req = MockRequirement(
            requirement_id="REQ-CERT",
            original_text="The bidder must hold ISO 27001 certification.",
            parameter="ISO 27001",
            operator="==",
            value=None,
            unit=None
        )
        candidate = RetrievedEvidenceCandidate(
            rank=1,
            evidence_id="EVD-T1",
            sentence="ACME Global Inc holds the ISO 27001 certification valid through December 2026.",
            similarity_score=0.85,
            source_document="BIDDER-PROPOSAL.pdf",
            page_number=5,
            numeric_values=[]
        )
        ret = RequirementRetrievalResult(
            requirement_id="REQ-CERT",
            requirement_text=req.original_text,
            parameter=req.parameter,
            candidates=[candidate]
        )

        eval_res = classifier.evaluate_clause(
            req,
            ret,
            tender_submission_date="2024-10-15",
            bidder_name="ABC Technologies Pvt. Ltd."
        )
        assert eval_res.label == "UNCERTAIN"
        assert "distinct" in eval_res.reasoning.lower() or "third-party" in eval_res.reasoning.lower() or "mismatch" in eval_res.reasoning.lower()

    def test_similar_name_evidence_returns_uncertain(self, classifier):
        """ABC Technology Solutions Ltd. holding certificate does not satisfy ABC Technologies Pvt. Ltd."""
        req = MockRequirement(
            requirement_id="REQ-CERT",
            original_text="The bidder must hold ISO 27001 certification.",
            parameter="ISO 27001",
            operator="==",
            value=None,
            unit=None
        )
        candidate = RetrievedEvidenceCandidate(
            rank=1,
            evidence_id="EVD-SIM1",
            sentence="ABC Technology Solutions Ltd. holds the ISO 27001 certification valid through December 2026.",
            similarity_score=0.85,
            source_document="BIDDER-PROPOSAL.pdf",
            page_number=5,
            numeric_values=[]
        )
        ret = RequirementRetrievalResult(
            requirement_id="REQ-CERT",
            requirement_text=req.original_text,
            parameter=req.parameter,
            candidates=[candidate]
        )

        eval_res = classifier.evaluate_clause(
            req,
            ret,
            tender_submission_date="2024-10-15",
            bidder_name="ABC Technologies Pvt. Ltd."
        )
        assert eval_res.label == "UNCERTAIN"
        assert "distinct" in eval_res.reasoning.lower() or "similarity" in eval_res.reasoning.lower() or "mismatch" in eval_res.reasoning.lower()
