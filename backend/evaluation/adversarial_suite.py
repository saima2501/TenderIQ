"""
adversarial_suite.py - TenderIQ Adversarial and Robustness Stress Testing Suite
Part of Phase 8: Core NLP Reliability Improvements (Step 2F)

Evaluates whether TenderIQ remains reliable when processing difficult, misleading,
contradictory, incomplete, and adversarial procurement documents.

Covers 12 Adversarial Categories:
 1. Negation and misleading compliance claims.
 2. Parent, subsidiary, JV, subcontractor, and unrelated third-party credential attribution.
 3. Similar but legally distinct company names.
 4. Numeric boundary manipulation and ambiguous quantities.
 5. Indian financial unit conversions involving Crore, Lakh, and INR.
 6. Expired, ambiguous, and mismatched certificates.
 7. Contradictory evidence across proposals and annexures.
 8. Missing mandatory documents.
 9. Misleading statements and unsupported declarations.
10. Repeated text, irrelevant paragraphs, and document formatting noise.
11. Conflicting bidder identities.
12. Semantic paraphrases that should not be mistaken for unrelated evidence.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Optional, Any, List, Dict, Tuple

from backend.extraction_nlp.requirement_extractor import RequirementExtractor, ExtractedRequirement
from backend.extraction_nlp.evidence_extractor import EvidenceExtractor
from backend.matching.tfidf_retriever import RequirementRetrievalResult, RetrievedEvidenceCandidate
from backend.classification.baseline_classifier import BaselineClassifier


@dataclass
class AdversarialCase:
    case_id: str
    category: str
    description: str
    requirement_text: str
    parameter: str
    operator: str
    value: Any
    unit: str
    is_mandatory: bool
    bidder_name: str
    evidence_text: str
    source_document: str = "BIDDER-PROPOSAL.pdf"
    page_number: int = 1
    secondary_evidence_text: Optional[str] = None
    secondary_source_document: Optional[str] = None
    secondary_page_number: Optional[int] = None
    tender_submission_deadline: str = "2024-10-15"
    expected_verdict: str = "COMPLIANT"
    safety_rule: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_adversarial_cases() -> List[AdversarialCase]:
    """Returns the complete benchmark of 52 adversarial stress test cases."""
    cases = [
        # ===================================================================
        # 1. Negation and misleading compliance claims
        # ===================================================================
        AdversarialCase(
            case_id="ADV-CAT01-001",
            category="1. Negation & Misleading Claims",
            description="Explicit SLA dispatch exclusion with remote tier caveat",
            requirement_text="The technical support team must provide an on-site engineer within 4 hours of incident escalation.",
            parameter="on-site engineer response time",
            operator="<=",
            value=4.0,
            unit="hours",
            is_mandatory=True,
            bidder_name="Alpha Cloud Systems Ltd",
            evidence_text="For Critical (Priority 1) incidents, our standard support tier does not include on-site dispatch within 4 hours; all on-site visits require a separate emergency surcharge.",
            expected_verdict="NON_COMPLIANT",
            safety_rule="Explicit negation must override semantic retrieval similarity"
        ),
        AdversarialCase(
            case_id="ADV-CAT01-002",
            category="1. Negation & Misleading Claims",
            description="Exclusion of offshore telemetry transfers",
            requirement_text="All patient identifiable records must be stored within the domestic sovereign data center.",
            parameter="sovereign data center residency",
            operator="CONTAINS",
            value="sovereign data center",
            unit="",
            is_mandatory=True,
            bidder_name="Beta HealthTech Ltd",
            evidence_text="Primary patient data is stored locally, but we do not guarantee that off-site telemetry and diagnostic logs will not be transferred to offshore processing hubs.",
            expected_verdict="NON_COMPLIANT",
            safety_rule="Negative polarity regarding mandatory residency must yield NON_COMPLIANT"
        ),
        AdversarialCase(
            case_id="ADV-CAT01-003",
            category="1. Negation & Misleading Claims",
            description="Direct denial of offline air-gapped sync requirement",
            requirement_text="The bidder must provide offline air-gapped data synchronization.",
            parameter="offline air-gapped data synchronization",
            operator="CONTAINS",
            value="offline air-gapped synchronization",
            unit="",
            is_mandatory=True,
            bidder_name="Gamma Cyber Ltd",
            evidence_text="Our enterprise platform does not support offline air-gapped data synchronization, operating strictly as a multi-tenant cloud-native application.",
            expected_verdict="NON_COMPLIANT",
            safety_rule="Explicit lack of supported feature must yield NON_COMPLIANT"
        ),
        AdversarialCase(
            case_id="ADV-CAT01-004",
            category="1. Negation & Misleading Claims",
            description="Accreditation denial with workflow adherence claim",
            requirement_text="The bidder must hold an active accredited ISO 27001 certification.",
            parameter="ISO 27001 certification",
            operator="CONTAINS",
            value="ISO 27001",
            unit="",
            is_mandatory=True,
            bidder_name="Delta Infotech Ltd",
            evidence_text="The bidder does not currently hold an accredited ISO 27001 certification, though our internal policies are modeled on ISO standards.",
            expected_verdict="NON_COMPLIANT",
            safety_rule="NLI contradiction and explicit certificate denial must yield NON_COMPLIANT"
        ),

        # ===================================================================
        # 2. Parent, Subsidiary, JV, Subcontractor & Third-Party Credential Attribution
        # ===================================================================
        AdversarialCase(
            case_id="ADV-CAT02-001",
            category="2. Entity Attribution & Scope",
            description="Parent company turnover cited without mandatory guarantee",
            requirement_text="The bidder must demonstrate an average annual financial turnover of at least INR 50 Crore.",
            parameter="financial turnover",
            operator=">=",
            value=50.0,
            unit="crore",
            is_mandatory=True,
            bidder_name="Alpha Cloud Systems Ltd",
            evidence_text="Our ultimate parent company, Global Megacorp Inc., achieved an annual turnover of INR 300 Crore.",
            expected_verdict="UNCERTAIN",
            safety_rule="Step 2C: Parent turnover without corporate guarantee must yield UNCERTAIN"
        ),
        AdversarialCase(
            case_id="ADV-CAT02-002",
            category="2. Entity Attribution & Scope",
            description="Parent company turnover WITH valid Form A-1 guarantee and Board Resolution",
            requirement_text="The bidder must demonstrate an average annual financial turnover of at least INR 50 Crore. Group holding company turnover may be considered only if backed by an irrevocable Board Resolution and Corporate Financial Guarantee Form A-1.",
            parameter="financial turnover",
            operator=">=",
            value=50.0,
            unit="crore",
            is_mandatory=True,
            bidder_name="Alpha Cloud Systems Ltd",
            evidence_text="Our ultimate parent company Alpha Global Holding Corp achieved turnover of INR 180 Crore, and has executed irrevocable Corporate Guarantee Form A-1 accompanied by certified Board Resolution #AGH-2024-09.",
            expected_verdict="COMPLIANT",
            safety_rule="Step 2C: Parent turnover backed by valid guarantee is legally authorized"
        ),
        AdversarialCase(
            case_id="ADV-CAT02-003",
            category="2. Entity Attribution & Scope",
            description="Sister affiliate experience cited without legal transfer deed",
            requirement_text="The bidder must possess a minimum of 5 years experience in hospital EHR rollouts.",
            parameter="hospital EHR rollout experience",
            operator=">=",
            value=5.0,
            unit="years",
            is_mandatory=True,
            bidder_name="Beta HealthTech Ltd",
            evidence_text="Our sister affiliate Beta Diagnostics LLC has completed 8 years of hospital diagnostic rollouts.",
            expected_verdict="UNCERTAIN",
            safety_rule="Step 2C: Sister company credentials cannot be credited to prime bidder"
        ),
        AdversarialCase(
            case_id="ADV-CAT02-004",
            category="2. Entity Attribution & Scope",
            description="Regional subsidiary turnover cited without consolidation",
            requirement_text="The bidder must furnish a bank solvency certificate of at least INR 10 Crore.",
            parameter="bank solvency certificate",
            operator=">=",
            value=10.0,
            unit="crore",
            is_mandatory=True,
            bidder_name="Gamma Solar Pvt Ltd",
            evidence_text="Our wholly owned subsidiary Gamma Solar Western Region Ltd maintains a bank solvency certificate of INR 15 Crore.",
            expected_verdict="UNCERTAIN",
            safety_rule="Step 2C: Subsidiary solvency cannot be claimed by prime bidder without parent deed"
        ),
        AdversarialCase(
            case_id="ADV-CAT02-005",
            category="2. Entity Attribution & Scope",
            description="Subcontractor ISO certification offered for prime bidder requirement",
            requirement_text="The bidder must hold an active ISO 9001:2015 certification valid through the bid submission date.",
            parameter="ISO 9001:2015 certification",
            operator="CONTAINS",
            value="ISO 9001:2015",
            unit="",
            is_mandatory=True,
            bidder_name="Delta Infotech Ltd",
            evidence_text="Our cabling subcontractor, Delta Wiring Services Ltd, maintains an active accredited ISO 9001:2015 certification valid through 2026.",
            expected_verdict="UNCERTAIN",
            safety_rule="Step 2C: Subcontractor certifications cannot substitute prime bidder compliance"
        ),

        # ===================================================================
        # 3. Similar but Legally Distinct Company Names
        # ===================================================================
        AdversarialCase(
            case_id="ADV-CAT03-001",
            category="3. Similar Legal Entity Names",
            description="Private Limited vs Limited distinction (separate incorporation)",
            requirement_text="The bidder must demonstrate an average annual audited financial turnover of at least INR 25.0 Crore.",
            parameter="financial turnover",
            operator=">=",
            value=25.0,
            unit="crore",
            is_mandatory=True,
            bidder_name="Apex Power Systems Pvt Ltd",
            evidence_text="Apex Power Systems Ltd has reported an audited annual revenue of INR 85 Crore for the fiscal year 2023.",
            expected_verdict="UNCERTAIN",
            safety_rule="Step 2C: Distinct corporate suffix (Pvt Ltd vs Ltd) triggers UNCERTAIN"
        ),
        AdversarialCase(
            case_id="ADV-CAT03-002",
            category="3. Similar Legal Entity Names",
            description="Distinction between generic suffix variants (Technologies vs Solutions)",
            requirement_text="The bidder must possess a minimum of 7 years of corporate operating experience.",
            parameter="operating experience",
            operator=">=",
            value=7.0,
            unit="years",
            is_mandatory=True,
            bidder_name="Summit Technologies Ltd",
            evidence_text="Summit Tech Solutions Ltd was incorporated in 2012, demonstrating 12 years of enterprise operations.",
            expected_verdict="UNCERTAIN",
            safety_rule="Step 2C: Distinct corporate names must not be conflated"
        ),
        AdversarialCase(
            case_id="ADV-CAT03-003",
            category="3. Similar Legal Entity Names",
            description="Private Limited vs LLP entity mismatch",
            requirement_text="The bidder must demonstrate a positive net worth of at least INR 5 Crore.",
            parameter="positive net worth",
            operator=">=",
            value=5.0,
            unit="crore",
            is_mandatory=True,
            bidder_name="Vanguard Logistics Pvt Ltd",
            evidence_text="Vanguard Logistics LLP registered a positive net worth of INR 12 Crore as per chartered accountant certificate.",
            expected_verdict="UNCERTAIN",
            safety_rule="Step 2C: LLP credentials cannot be attributed to Private Limited bidder"
        ),
        AdversarialCase(
            case_id="ADV-CAT03-004",
            category="3. Similar Legal Entity Names",
            description="Exact entity name match succeeds",
            requirement_text="The bidder must demonstrate a minimum of 8 years of operational experience.",
            parameter="operational experience",
            operator=">=",
            value=8.0,
            unit="years",
            is_mandatory=True,
            bidder_name="Nexus Cyber Systems Ltd",
            evidence_text="Nexus Cyber Systems Ltd was incorporated in 2014 and has completed 10 years of specialized enterprise operations.",
            expected_verdict="COMPLIANT",
            safety_rule="Exact legal name match safely attributes corporate experience"
        ),

        # ===================================================================
        # 4. Numeric Boundary Manipulation & Ambiguous Quantities
        # ===================================================================
        AdversarialCase(
            case_id="ADV-CAT04-001",
            category="4. Numeric Boundary & Ambiguity",
            description="Financial turnover barely below boundary (49.8 Cr vs 50.0 Cr)",
            requirement_text="The bidder must demonstrate an average annual audited financial turnover of at least INR 50.0 Crore.",
            parameter="financial turnover",
            operator=">=",
            value=50.0,
            unit="crore",
            is_mandatory=True,
            bidder_name="Alpha Cloud Systems Ltd",
            evidence_text="Audited accounts indicate an average annual turnover of INR 49.8 Crore for FY2021-FY2023.",
            expected_verdict="NON_COMPLIANT",
            safety_rule="Strict mathematical boundary: 49.8 < 50.0 must be NON_COMPLIANT"
        ),
        AdversarialCase(
            case_id="ADV-CAT04-002",
            category="4. Numeric Boundary & Ambiguity",
            description="Vague ambiguous staffing range (8 to 12 engineers vs 10 required)",
            requirement_text="The bidder must commit a dedicated core team of at least 10 certified engineers on-site during deployment.",
            parameter="on-site certified engineers",
            operator=">=",
            value=10.0,
            unit="",
            is_mandatory=True,
            bidder_name="Beta HealthTech Ltd",
            evidence_text="We will provide approximately 8 to 12 certified engineers depending on project workload.",
            expected_verdict="UNCERTAIN",
            safety_rule="Ambiguous numeric ranges with uncertainty qualifier must yield UNCERTAIN"
        ),
        AdversarialCase(
            case_id="ADV-CAT04-003",
            category="4. Numeric Boundary & Ambiguity",
            description="Exact numeric surplus (12 engineers >= 10 required)",
            requirement_text="The bidder must commit a dedicated core team of at least 10 certified engineers on-site during deployment.",
            parameter="on-site certified engineers",
            operator=">=",
            value=10.0,
            unit="",
            is_mandatory=True,
            bidder_name="Beta HealthTech Ltd",
            evidence_text="We have earmarked 12 full-time certified engineers stationed permanently on-site throughout deployment.",
            expected_verdict="COMPLIANT",
            safety_rule="Unambiguous numeric surplus satisfies requirement"
        ),
        AdversarialCase(
            case_id="ADV-CAT04-004",
            category="4. Numeric Boundary & Ambiguity",
            description="Ambiguous 'up to' upper limit on response time constraint",
            requirement_text="The support helpdesk must provide an initial response time of at most 15 minutes.",
            parameter="initial response time",
            operator="<=",
            value=15.0,
            unit="minutes",
            is_mandatory=True,
            bidder_name="Gamma Solar Pvt Ltd",
            evidence_text="Tickets receive attention in up to 30 minutes depending on daytime call volumes.",
            expected_verdict="NON_COMPLIANT",
            safety_rule="Explicit numeric shortfall (30 min > 15 min max) strictly yields NON_COMPLIANT"
        ),
        AdversarialCase(
            case_id="ADV-CAT04-005",
            category="4. Numeric Boundary & Ambiguity",
            description="Uptime availability 99.85% below 99.90% mandatory SLA",
            requirement_text="The hosted infrastructure must maintain a minimum monthly uptime availability of 99.90%.",
            parameter="uptime availability",
            operator=">=",
            value=99.90,
            unit="%",
            is_mandatory=True,
            bidder_name="Delta Infotech Ltd",
            evidence_text="Our platform SLA provides 99.85% guaranteed monthly availability.",
            expected_verdict="NON_COMPLIANT",
            safety_rule="99.85% < 99.90% strictly fails mandatory uptime SLA"
        ),

        # ===================================================================
        # 5. Indian Financial Unit Conversions (Crore, Lakh, INR)
        # ===================================================================
        AdversarialCase(
            case_id="ADV-CAT05-001",
            category="5. Indian Financial Units",
            description="Turnover in Lakhs satisfies Crore requirement (2500 Lakh >= 20 Crore)",
            requirement_text="The bidder must demonstrate an average annual financial turnover of at least INR 20 Crore.",
            parameter="annual financial turnover",
            operator=">=",
            value=20.0,
            unit="crore",
            is_mandatory=True,
            bidder_name="Alpha Cloud Systems Ltd",
            evidence_text="Audited financial statements report an average annual turnover of Rs. 2,500 Lakhs over the last three fiscal years.",
            expected_verdict="COMPLIANT",
            safety_rule="Step 2D: 2,500 Lakh = 25.0 Crore >= 20.0 Crore correctly satisfies requirement"
        ),
        AdversarialCase(
            case_id="ADV-CAT05-002",
            category="5. Indian Financial Units",
            description="Turnover in Lakhs strictly falls short of Crore requirement (1500 Lakh < 20 Crore)",
            requirement_text="The bidder must demonstrate an average annual financial turnover of at least INR 20 Crore.",
            parameter="annual financial turnover",
            operator=">=",
            value=20.0,
            unit="crore",
            is_mandatory=True,
            bidder_name="Beta HealthTech Ltd",
            evidence_text="Audited annual revenue averaged Rs. 1,500 Lakhs over the preceding three fiscal years.",
            expected_verdict="NON_COMPLIANT",
            safety_rule="Step 2D: 1,500 Lakh = 15.0 Crore < 20.0 Crore strictly fails requirement"
        ),
        AdversarialCase(
            case_id="ADV-CAT05-003",
            category="5. Indian Financial Units",
            description="Spelled-out words 'fifty crore rupees' satisfies 50 Crore threshold",
            requirement_text="The bidder must possess a minimum net worth of INR 50 Crore.",
            parameter="minimum net worth",
            operator=">=",
            value=50.0,
            unit="crore",
            is_mandatory=True,
            bidder_name="Gamma Solar Pvt Ltd",
            evidence_text="Statutory auditor certificate verifies that our positive net worth stands at fifty crore rupees.",
            expected_verdict="COMPLIANT",
            safety_rule="Step 2A + Step 2D: 'fifty crore rupees' normalized to 50 Crore"
        ),
        AdversarialCase(
            case_id="ADV-CAT05-004",
            category="5. Indian Financial Units",
            description="Exact boundary match (INR 10 Crore == 1000 Lakh)",
            requirement_text="The bidder must provide a bank solvency certificate of at least INR 10 Crore.",
            parameter="bank solvency certificate",
            operator=">=",
            value=10.0,
            unit="crore",
            is_mandatory=True,
            bidder_name="Delta Infotech Ltd",
            evidence_text="Scheduled commercial bank solvency certificate issued for Rs. 1,000 Lakhs.",
            expected_verdict="COMPLIANT",
            safety_rule="Step 2D: 1000 Lakhs == 10 Crore satisfies >= 10 Crore"
        ),
        AdversarialCase(
            case_id="ADV-CAT05-005",
            category="5. Indian Financial Units",
            description="Unpegged foreign currency without conversion rate yields UNCERTAIN",
            requirement_text="The bidder must demonstrate an annual financial turnover of at least INR 20 Crore.",
            parameter="annual financial turnover",
            operator=">=",
            value=20.0,
            unit="crore",
            is_mandatory=True,
            bidder_name="Delta Infotech Ltd",
            evidence_text="The company achieved an average audited annual turnover of USD 1.8 Million.",
            expected_verdict="UNCERTAIN",
            safety_rule="Step 2D: Unpegged foreign currency without conversion rate cannot be verified"
        ),

        # ===================================================================
        # 6. Expired, Ambiguous & Mismatched Certificates
        # ===================================================================
        AdversarialCase(
            case_id="ADV-CAT06-001",
            category="6. Certificate Posture & Validity",
            description="Certificate expired 9 months prior to bid submission deadline",
            requirement_text="The bidder must hold an active accredited ISO 9001:2015 (Quality Management Systems) certification valid through the bid submission date.",
            parameter="ISO 9001:2015 certification",
            operator="CONTAINS",
            value="ISO 9001:2015",
            unit="",
            is_mandatory=True,
            bidder_name="Alpha Cloud Systems Ltd",
            evidence_text="Our ISO 9001:2015 (Quality Management Systems) certification expired on January 15, 2024, and re-certification audit is currently scheduled.",
            tender_submission_deadline="2024-10-15",
            expected_verdict="NON_COMPLIANT",
            safety_rule="Step 2B: Expired certificate prior to tender deadline must yield NON_COMPLIANT"
        ),
        AdversarialCase(
            case_id="ADV-CAT06-002",
            category="6. Certificate Posture & Validity",
            description="Certificate valid past submission deadline",
            requirement_text="The bidder must hold an active accredited ISO 27001 certification valid through the bid submission date.",
            parameter="ISO 27001 certification",
            operator="CONTAINS",
            value="ISO 27001",
            unit="",
            is_mandatory=True,
            bidder_name="Alpha Cloud Systems Ltd",
            evidence_text="Bidder ISO 27001 accredited certificate issued 10-Jan-2022 is valid through 10-Jan-2026, certified by UKAS-accredited registrar.",
            tender_submission_deadline="2024-10-15",
            expected_verdict="COMPLIANT",
            safety_rule="Step 2B: Verified active certificate valid past deadline yields COMPLIANT"
        ),
        AdversarialCase(
            case_id="ADV-CAT06-003",
            category="6. Certificate Posture & Validity",
            description="Bare unverified certification claim without dates or registration details",
            requirement_text="The bidder must hold an active accredited ISO 27001 certification valid through the bid submission date.",
            parameter="ISO 27001 certification",
            operator="CONTAINS",
            value="ISO 27001",
            unit="",
            is_mandatory=True,
            bidder_name="Beta HealthTech Ltd",
            evidence_text="Our company is ISO 27001 certified across all cloud hosting operations.",
            tender_submission_deadline="2024-10-15",
            expected_verdict="UNCERTAIN",
            safety_rule="Step 2B.1: Bare unverified certification without expiry date yields UNCERTAIN"
        ),
        AdversarialCase(
            case_id="ADV-CAT06-004",
            category="6. Certificate Posture & Validity",
            description="Mismatched ISO standard (ISO 9001 offered when ISO 14001 required)",
            requirement_text="The bidder must hold an active accredited ISO 14001:2015 Environmental Management certification.",
            parameter="ISO 14001:2015 certification",
            operator="CONTAINS",
            value="ISO 14001:2015",
            unit="",
            is_mandatory=True,
            bidder_name="Gamma Solar Pvt Ltd",
            evidence_text="Our company holds an active accredited ISO 9001:2015 Quality Management certificate valid until 2026.",
            tender_submission_deadline="2024-10-15",
            expected_verdict="UNCERTAIN",
            safety_rule="Mismatched accreditation standard cannot satisfy required specification"
        ),
        AdversarialCase(
            case_id="ADV-CAT06-005",
            category="6. Certificate Posture & Validity",
            description="Certificate expires 1 day before bid submission deadline",
            requirement_text="The bidder must hold an active accredited ISO 9001:2015 certification valid through the bid submission date.",
            parameter="ISO 9001:2015 certification",
            operator="CONTAINS",
            value="ISO 9001:2015",
            unit="",
            is_mandatory=True,
            bidder_name="Delta Infotech Ltd",
            evidence_text="ISO 9001:2015 certificate #QA-9921 is valid until 14-Oct-2024.",
            tender_submission_deadline="2024-10-15",
            expected_verdict="NON_COMPLIANT",
            safety_rule="Step 2B: Certificate expiring before bid deadline must yield NON_COMPLIANT"
        ),

        # ===================================================================
        # 7. Contradictory Evidence Across Proposals & Annexures
        # ===================================================================
        AdversarialCase(
            case_id="ADV-CAT07-001",
            category="7. Cross-Document Contradiction",
            description="Executive summary claims USD 32M while audited balance sheet registers USD 15M (req: 25M)",
            requirement_text="The bidder must demonstrate an average annual audited financial turnover of at least USD 25.0 Million over the last three fiscal years.",
            parameter="audited financial turnover",
            operator=">=",
            value=25.0,
            unit="USD Million",
            is_mandatory=True,
            bidder_name="Alpha Cloud Systems Ltd",
            evidence_text="Technical proposal executive summary claims USD 32.0 Million turnover for the preceding three fiscal years.",
            source_document="BIDDER-ALPHA-PROPOSAL.pdf",
            page_number=3,
            secondary_evidence_text="Annexure F (Audited Balance Sheet Page 12) officially registers average turnover as only USD 15.0 Million.",
            secondary_source_document="ANNEXURE-F-BALANCE-SHEET.pdf",
            secondary_page_number=12,
            expected_verdict="NON_COMPLIANT",
            safety_rule="Phase 7 Reconciler: Hard numeric conflict escalates to NON_COMPLIANT"
        ),
        AdversarialCase(
            case_id="ADV-CAT07-002",
            category="7. Cross-Document Contradiction",
            description="Proposal claims dedicated PM while staffing schedule assigns 10% part-time",
            requirement_text="The proposed Project Manager must possess a valid PMP or PRINCE2 certification and a minimum of 10 years of relevant experience.",
            parameter="Project Manager qualification",
            operator=">=",
            value=10.0,
            unit="years",
            is_mandatory=True,
            bidder_name="Alpha Cloud Systems Ltd",
            evidence_text="Technical proposal names Robert Chen as dedicated PMP Project Manager with 12 years experience.",
            source_document="BIDDER-ALPHA-PROPOSAL.pdf",
            page_number=4,
            secondary_evidence_text="Staffing Schedule Annexure G assigns Robert Chen 10% part-time allocation across 3 other active client sites.",
            secondary_source_document="ANNEXURE-G-STAFFING.pdf",
            secondary_page_number=5,
            expected_verdict="NON_COMPLIANT",
            safety_rule="Phase 7 Reconciler: Part-time staffing conflict overrides proposal claim"
        ),
        AdversarialCase(
            case_id="ADV-CAT07-003",
            category="7. Cross-Document Contradiction",
            description="Proposal claims 99.95% uptime SLA while technical annexure disclaims 98.00%",
            requirement_text="The production system hosting environment must guarantee a minimum monthly uptime SLA of 99.90%.",
            parameter="monthly uptime SLA",
            operator=">=",
            value=99.90,
            unit="%",
            is_mandatory=True,
            bidder_name="Beta HealthTech Ltd",
            evidence_text="Our executive proposal commits to a minimum monthly uptime SLA of 99.95%.",
            source_document="BIDDER-BETA-PROPOSAL.pdf",
            page_number=2,
            secondary_evidence_text="SLA Annexure C contract terms restrict guaranteed platform availability to 98.00% uptime with maintenance excluded.",
            secondary_source_document="ANNEXURE-C-SLA.pdf",
            secondary_page_number=8,
            expected_verdict="NON_COMPLIANT",
            safety_rule="Phase 7 Reconciler: Contract annexure conflict overrides executive proposal"
        ),
        AdversarialCase(
            case_id="ADV-CAT07-004",
            category="7. Cross-Document Contradiction",
            description="Harmonious multi-document verification (no conflict)",
            requirement_text="The bidder must demonstrate an average annual audited financial turnover of at least INR 50 Crore.",
            parameter="annual financial turnover",
            operator=">=",
            value=50.0,
            unit="crore",
            is_mandatory=True,
            bidder_name="Gamma Solar Pvt Ltd",
            evidence_text="Technical proposal narrative confirms an average audited financial turnover of INR 75 Crore.",
            source_document="BIDDER-GAMMA-PROPOSAL.pdf",
            page_number=2,
            secondary_evidence_text="Audited Financial Annexure B certifies audited revenue of INR 75 Crore for FY2023-24.",
            secondary_source_document="ANNEXURE-B-AUDITED.pdf",
            secondary_page_number=14,
            expected_verdict="COMPLIANT",
            safety_rule="Harmonious cross-document confirmation establishes compliance"
        ),

        # ===================================================================
        # 8. Missing Mandatory Documents
        # ===================================================================
        AdversarialCase(
            case_id="ADV-CAT08-001",
            category="8. Missing Mandatory Documents",
            description="Mandatory OEM MAF letter omitted from bid package",
            requirement_text="If the bidder is not the original equipment manufacturer (OEM), they must furnish a Tender-Specific Manufacturer Authorization Form (MAF) from the OEM.",
            parameter="tender-specific OEM authorization (MAF)",
            operator="MANDATORY_SUBMISSION",
            value="OEM_MAF_LETTER",
            unit="",
            is_mandatory=True,
            bidder_name="Delta Infotech Ltd",
            evidence_text="",
            expected_verdict="MISSING_EVIDENCE",
            safety_rule="Missing evidence must yield MISSING_EVIDENCE, not NON_COMPLIANT"
        ),
        AdversarialCase(
            case_id="ADV-CAT08-002",
            category="8. Missing Mandatory Documents",
            description="Mandatory EMD bank guarantee or exemption omitted",
            requirement_text="The bidder must submit an Earnest Money Deposit (EMD) bank guarantee or valid GeM MSME exemption certificate.",
            parameter="Earnest Money Deposit (EMD)",
            operator="MANDATORY_SUBMISSION",
            value="EMD_SUBMISSION",
            unit="",
            is_mandatory=True,
            bidder_name="Delta Infotech Ltd",
            evidence_text="",
            expected_verdict="MISSING_EVIDENCE",
            safety_rule="Empty submission for mandatory requirement yields MISSING_EVIDENCE"
        ),
        AdversarialCase(
            case_id="ADV-CAT08-003",
            category="8. Missing Mandatory Documents",
            description="Mandatory Make in India Local Content Declaration Form omitted",
            requirement_text="The bidder must submit a Make in India Local Content Declaration Form certifying minimum 50% Class-I local content.",
            parameter="Make in India Local Content Declaration",
            operator="MANDATORY_SUBMISSION",
            value="LOCAL_CONTENT_DECLARATION",
            unit="",
            is_mandatory=True,
            bidder_name="Delta Infotech Ltd",
            evidence_text="",
            expected_verdict="MISSING_EVIDENCE",
            safety_rule="Empty submission yields MISSING_EVIDENCE"
        ),
        AdversarialCase(
            case_id="ADV-CAT08-004",
            category="8. Missing Mandatory Documents",
            description="Mandatory Non-Debarment notarized undertaking omitted",
            requirement_text="The bidder must submit a solemn declaration affirming that the firm has never been blacklisted or debarred.",
            parameter="solemn declaration of non-debarment",
            operator="MANDATORY_SUBMISSION",
            value="NON_BLACKLISTING_AFFIDAVIT",
            unit="",
            is_mandatory=True,
            bidder_name="Delta Infotech Ltd",
            evidence_text="",
            expected_verdict="MISSING_EVIDENCE",
            safety_rule="Empty submission yields MISSING_EVIDENCE"
        ),

        # ===================================================================
        # 9. Misleading Statements & Unsupported Declarations
        # ===================================================================
        AdversarialCase(
            case_id="ADV-CAT09-001",
            category="9. Misleading Statements",
            description="Alignment claim without formal CMMI Level 5 appraisal",
            requirement_text="The bidder must possess CMMI Dev Level 5 appraisal.",
            parameter="CMMI Dev Level 5 appraisal",
            operator="CONTAINS",
            value="CMMI Dev Level 5",
            unit="",
            is_mandatory=True,
            bidder_name="Alpha Cloud Systems Ltd",
            evidence_text="Our software engineering practices are fully aligned with the intent and principles of CMMI Dev Level 5 guidelines.",
            expected_verdict="UNCERTAIN",
            safety_rule="Alignment/compatibility claims without formal appraisal yield UNCERTAIN"
        ),
        AdversarialCase(
            case_id="ADV-CAT09-002",
            category="9. Misleading Statements",
            description="Compatibility claim without formal ISO 27001 certificate",
            requirement_text="The bidder must hold an active accredited ISO 27001 certification.",
            parameter="ISO 27001 certification",
            operator="CONTAINS",
            value="ISO 27001",
            unit="",
            is_mandatory=True,
            bidder_name="Beta HealthTech Ltd",
            evidence_text="Our security architecture is designed to be fully compatible with ISO 27001 controls.",
            expected_verdict="UNCERTAIN",
            safety_rule="Compatibility claims without certified audit yield UNCERTAIN"
        ),
        AdversarialCase(
            case_id="ADV-CAT09-003",
            category="9. Misleading Statements",
            description="Future intention to obtain mandatory GSTIN registration",
            requirement_text="The bidder must submit an active GSTIN registration certificate.",
            parameter="active GSTIN registration",
            operator="MANDATORY_SUBMISSION",
            value="VALID_GSTIN_FILING",
            unit="",
            is_mandatory=True,
            bidder_name="Gamma Solar Pvt Ltd",
            evidence_text="The company does not currently have GSTIN but will apply for registration upon award of contract.",
            expected_verdict="NON_COMPLIANT",
            safety_rule="Future promise of mandatory statutory filing yields NON_COMPLIANT"
        ),
        AdversarialCase(
            case_id="ADV-CAT09-004",
            category="9. Misleading Statements",
            description="Promotional claim of 'military-grade' without empirical AES standard",
            requirement_text="All data must be encrypted using AES-256.",
            parameter="AES-256 Encryption compliance",
            operator="CONTAINS",
            value="AES-256 Encryption",
            unit="",
            is_mandatory=True,
            bidder_name="Delta Infotech Ltd",
            evidence_text="Robust encryption standards and compliance policies are strictly maintained across our cloud systems.",
            expected_verdict="UNCERTAIN",
            safety_rule="Generic buzzwords without standard specification yield UNCERTAIN"
        ),

        # ===================================================================
        # 10. Repeated Text, Irrelevant Paragraphs & Formatting Noise
        # ===================================================================
        AdversarialCase(
            case_id="ADV-CAT10-001",
            category="10. Document Noise & Boilerplate",
            description="Factual commitment embedded inside repetitive header/footer boilerplate",
            requirement_text="The technical support team must acknowledge critical incidents within 30 minutes.",
            parameter="incident acknowledgement time",
            operator="<=",
            value=30.0,
            unit="minutes",
            is_mandatory=True,
            bidder_name="Alpha Cloud Systems Ltd",
            evidence_text="CONFIDENTIAL BIDDER PROPOSAL - SECTION 8.2 - SLA GUARANTEES. Alpha Cloud Systems Ltd. Page 14 of 95. [Header Noise] We maintain 24/7 network monitoring. For Critical (Priority 1) incidents, our helpdesk acknowledges all tickets within 15 minutes via automated alerting and engineer callback. All rights reserved. [Footer Noise]",
            expected_verdict="COMPLIANT",
            safety_rule="Factual metric within boilerplate is extracted and evaluated"
        ),
        AdversarialCase(
            case_id="ADV-CAT10-002",
            category="10. Document Noise & Boilerplate",
            description="High lexical overlap but zero factual commitment (pure filler)",
            requirement_text="The contractor must provide a comprehensive on-site warranty of at least 36 months.",
            parameter="warranty duration",
            operator=">=",
            value=36.0,
            unit="months",
            is_mandatory=True,
            bidder_name="Beta HealthTech Ltd",
            evidence_text="Warranty and maintenance are vital components of any engineering procurement project. The contractor recognizes the importance of warranty. Warranty coverage ensures high availability. Comprehensive maintenance must be maintained throughout the project lifecycle.",
            expected_verdict="UNCERTAIN",
            safety_rule="Lexical overlap without numeric metrics yields UNCERTAIN"
        ),
        AdversarialCase(
            case_id="ADV-CAT10-003",
            category="10. Document Noise & Boilerplate",
            description="Repeated turnover buzzwords without audited financial figures",
            requirement_text="The bidder must maintain an average annual turnover of at least INR 20 Crore.",
            parameter="annual financial turnover",
            operator=">=",
            value=20.0,
            unit="crore",
            is_mandatory=True,
            bidder_name="Gamma Solar Pvt Ltd",
            evidence_text="Financial Capability Overview. Financial turnover growth in 2021 was strong. Financial turnover in 2022 continued to expand. Financial turnover in 2023 was robust. We are financially capable of executing large contracts.",
            expected_verdict="UNCERTAIN",
            safety_rule="Financial narrative lacking figures yields UNCERTAIN"
        ),
        AdversarialCase(
            case_id="ADV-CAT10-004",
            category="10. Document Noise & Boilerplate",
            description="Committed current staffing exceeds requirement despite historical background noise",
            requirement_text="The bidder must commit at least 5 full-time senior database administrators.",
            parameter="senior database administrators",
            operator=">=",
            value=5.0,
            unit="",
            is_mandatory=True,
            bidder_name="Delta Infotech Ltd",
            evidence_text="Database administration staffing: We have assigned 6 full-time senior certified DBAs dedicated to this project. Note: See appendix for historical staffing in 2018 (3 DBAs), 2019 (4 DBAs), and future projections.",
            expected_verdict="COMPLIANT",
            safety_rule="Current commitment satisfies numeric requirement despite background noise"
        ),

        # ===================================================================
        # 11. Conflicting Bidder Identities
        # ===================================================================
        AdversarialCase(
            case_id="ADV-CAT11-001",
            category="11. Conflicting Bidder Identities",
            description="Exact legal name match with trusted bidder",
            requirement_text="The bidder must demonstrate an annual turnover of at least INR 30 Crore.",
            parameter="annual financial turnover",
            operator=">=",
            value=30.0,
            unit="crore",
            is_mandatory=True,
            bidder_name="Apex Healthcare Solutions Pvt Ltd",
            evidence_text="Prime bidder Apex Healthcare Solutions Pvt Ltd has achieved audited turnover of INR 45 Crore.",
            expected_verdict="COMPLIANT",
            safety_rule="Exact legal name match is eligible for compliance"
        ),
        AdversarialCase(
            case_id="ADV-CAT11-002",
            category="11. Conflicting Bidder Identities",
            description="Foreign LLC entity cited for Indian Private Limited bidder",
            requirement_text="The bidder must demonstrate an annual turnover of at least INR 30 Crore.",
            parameter="annual financial turnover",
            operator=">=",
            value=30.0,
            unit="crore",
            is_mandatory=True,
            bidder_name="Apex Healthcare Solutions Pvt Ltd",
            evidence_text="Apex Global Health Informatics LLC (registered in Delaware) achieved turnover of INR 80 Crore.",
            expected_verdict="UNCERTAIN",
            safety_rule="Step 2C: Discrepancy between Indian Pvt Ltd and Delaware LLC yields UNCERTAIN"
        ),
        AdversarialCase(
            case_id="ADV-CAT11-003",
            category="11. Conflicting Bidder Identities",
            description="Consortium partner experience cited without joint bidding deed",
            requirement_text="The bidder must have executed at least 3 industrial solar microgrid projects.",
            parameter="industrial solar projects",
            operator=">=",
            value=3.0,
            unit="projects",
            is_mandatory=True,
            bidder_name="Zenith Solar Energy Systems Ltd",
            evidence_text="Consortium partner SunPower EPC Infra Ltd has executed 5 industrial solar microgrid projects.",
            expected_verdict="UNCERTAIN",
            safety_rule="Step 2C: Consortium member credentials cited without verified joint deed yield UNCERTAIN"
        ),
        AdversarialCase(
            case_id="ADV-CAT11-004",
            category="11. Conflicting Bidder Identities",
            description="Unrelated third-party corporate certificate cited",
            requirement_text="The bidder must hold an active ISO 27001 certification.",
            parameter="ISO 27001 certification",
            operator="CONTAINS",
            value="ISO 27001",
            unit="",
            is_mandatory=True,
            bidder_name="Tata Communications Ltd",
            evidence_text="Accredited ISO 27001 certificate is issued in the name of Cisco Systems India Pvt Ltd valid until 2026.",
            expected_verdict="UNCERTAIN",
            safety_rule="Step 2C: Unrelated corporate identity yields UNCERTAIN"
        ),

        # ===================================================================
        # 12. Semantic Paraphrases (True Meaning Equivalent)
        # ===================================================================
        AdversarialCase(
            case_id="ADV-CAT12-001",
            category="12. Semantic Paraphrases",
            description="256-bit Advanced Encryption Standard storage matches AES-256 requirement",
            requirement_text="All data must be encrypted using AES-256.",
            parameter="AES-256 Encryption compliance",
            operator="CONTAINS",
            value="AES-256 Encryption",
            unit="",
            is_mandatory=True,
            bidder_name="Alpha Cloud Systems Ltd",
            evidence_text="All databases are secured with military-grade 256-bit Advanced Encryption Standard storage.",
            expected_verdict="COMPLIANT",
            safety_rule="Step 2E: Bidirectional acronym expansion recognizes exact technical standard"
        ),
        AdversarialCase(
            case_id="ADV-CAT12-002",
            category="12. Semantic Paraphrases",
            description="CMMI Maturity Level 5 benchmarked matches CMMI Dev Level 5 requirement",
            requirement_text="The bidder must possess CMMI Dev Level 5 appraisal.",
            parameter="CMMI Dev Level 5 appraisal",
            operator="CONTAINS",
            value="CMMI Dev Level 5",
            unit="",
            is_mandatory=True,
            bidder_name="Beta HealthTech Ltd",
            evidence_text="CMMI Maturity Level 5 appraisal benchmarked and verified under CMMI Institute appraisal #44901 valid through 30-Nov-2025.",
            expected_verdict="COMPLIANT",
            safety_rule="Step 2E: CMMI Maturity Level 5 recognized as semantic equivalent"
        ),
        AdversarialCase(
            case_id="ADV-CAT12-003",
            category="12. Semantic Paraphrases",
            description="Average disaster recovery restoration time 90 mins matches MTTR < 2 hours",
            requirement_text="The contractor must guarantee a Mean Time to Recover (MTTR) under 2 hours for critical service outages.",
            parameter="Mean Time to Recover (MTTR)",
            operator="<",
            value=2.0,
            unit="hours",
            is_mandatory=True,
            bidder_name="Gamma Solar Pvt Ltd",
            evidence_text="Average disaster recovery restoration time is maintained below 90 minutes across all critical services.",
            expected_verdict="COMPLIANT",
            safety_rule="Strict less-than unit normalization (90 min = 1.5 hr < 2 hr) and paraphrase recognition"
        ),
        AdversarialCase(
            case_id="ADV-CAT12-004",
            category="12. Semantic Paraphrases",
            description="24x7x365 multi-tier phone and on-site support matches round-the-clock requirement",
            requirement_text="The contractor must provide round-the-clock technical assistance throughout deployment.",
            parameter="round-the-clock technical assistance",
            operator="CONTAINS",
            value="round-the-clock",
            unit="",
            is_mandatory=True,
            bidder_name="Delta Infotech Ltd",
            evidence_text="Our centralized technical operations center provides 24x7x365 multi-tier engineer phone and on-site support.",
            expected_verdict="COMPLIANT",
            safety_rule="24x7x365 recognizes semantic equivalent of round-the-clock"
        ),
    ]
    return cases


def evaluate_adversarial_suite(classifier: Optional[BaselineClassifier] = None) -> Dict[str, Any]:
    """Executes the complete adversarial benchmark and compiles detailed metrics."""
    evd_ext = EvidenceExtractor()
    if classifier is None:
        classifier = BaselineClassifier(enable_nli=True, enable_reconciliation=True)

    cases = get_adversarial_cases()
    results = []

    category_stats = defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0})
    labels = ["COMPLIANT", "NON_COMPLIANT", "UNCERTAIN", "MISSING_EVIDENCE"]
    confusion_matrix = {gold: {pred: 0 for pred in labels} for gold in labels}

    false_compliant_count = 0
    passed_count = 0

    for c in cases:
        req_obj = ExtractedRequirement(
            requirement_id=c.case_id,
            original_text=c.requirement_text,
            category=c.category,
            parameter=c.parameter,
            operator=c.operator,
            value=c.value,
            unit=c.unit,
            is_mandatory=c.is_mandatory
        )

        cands = []
        if c.evidence_text.strip():
            evd_list = evd_ext.extract_from_text(c.evidence_text, source_document=c.source_document)
            all_numeric = []
            all_entities = []
            has_neg = False
            uncertain = False
            for e in evd_list:
                all_numeric.extend(e.numeric_values)
                all_entities.extend(e.extracted_entities)
                if e.has_negation:
                    has_neg = True
                if e.uncertainty_flag:
                    uncertain = True

            cand = RetrievedEvidenceCandidate(
                rank=1,
                evidence_id=f"EVD-1-{c.case_id}",
                sentence=c.evidence_text,
                similarity_score=0.85,
                source_document=c.source_document,
                page_number=c.page_number,
                has_negation=has_neg,
                numeric_values=all_numeric,
                extracted_entities=all_entities,
                uncertainty_flag=uncertain,
                retrieval_method="semantic"
            )
            cands.append(cand)

        if c.secondary_evidence_text and c.secondary_evidence_text.strip():
            sec_evd_list = evd_ext.extract_from_text(
                c.secondary_evidence_text,
                source_document=c.secondary_source_document or "ANNEXURE.pdf"
            )
            all_sec_numeric = []
            all_sec_entities = []
            sec_has_neg = False
            sec_uncertain = False
            for e in sec_evd_list:
                all_sec_numeric.extend(e.numeric_values)
                all_sec_entities.extend(e.extracted_entities)
                if e.has_negation:
                    sec_has_neg = True
                if e.uncertainty_flag:
                    sec_uncertain = True

            cand2 = RetrievedEvidenceCandidate(
                rank=2,
                evidence_id=f"EVD-2-{c.case_id}",
                sentence=c.secondary_evidence_text,
                similarity_score=0.80,
                source_document=c.secondary_source_document or "ANNEXURE.pdf",
                page_number=c.secondary_page_number or 1,
                has_negation=sec_has_neg,
                numeric_values=all_sec_numeric,
                extracted_entities=all_sec_entities,
                uncertainty_flag=sec_uncertain,
                retrieval_method="semantic"
            )
            cands.append(cand2)

        ret = RequirementRetrievalResult(
            requirement_id=c.case_id,
            requirement_text=c.requirement_text,
            parameter=c.parameter,
            candidates=cands,
            top_similarity=cands[0].similarity_score if cands else 0.0,
            has_candidate=bool(cands)
        )

        nli_pred = None
        if classifier.enable_nli and classifier.nli_classifier and classifier.nli_classifier.is_available and c.evidence_text.strip():
            nli_pred = classifier.nli_classifier.predict_pair(c.requirement_text, c.evidence_text)

        eval_res = classifier.evaluate_clause(
            req_obj,
            ret,
            nli_pred=nli_pred,
            tender_submission_date=c.tender_submission_deadline,
            bidder_name=c.bidder_name
        )

        pred_label = eval_res.label
        gold_label = c.expected_verdict
        is_pass = (pred_label == gold_label)

        if is_pass:
            passed_count += 1
            category_stats[c.category]["passed"] += 1
        else:
            category_stats[c.category]["failed"] += 1
        category_stats[c.category]["total"] += 1

        confusion_matrix[gold_label][pred_label] += 1

        # Safety failure check: False Compliant
        # When expected is NON_COMPLIANT, UNCERTAIN, or MISSING_EVIDENCE, but actual is COMPLIANT
        if gold_label != "COMPLIANT" and pred_label == "COMPLIANT":
            false_compliant_count += 1

        results.append({
            "case_id": c.case_id,
            "category": c.category,
            "description": c.description,
            "expected": gold_label,
            "actual": pred_label,
            "passed": is_pass,
            "reasoning": eval_res.reasoning,
            "safety_rule": c.safety_rule
        })

    total_cases = len(cases)
    accuracy = (passed_count / total_cases) * 100.0 if total_cases > 0 else 0.0

    # Per-class Precision, Recall, F1
    per_class_metrics = {}
    macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
    valid_classes = 0

    for lbl in labels:
        tp = confusion_matrix[lbl][lbl]
        fp = sum(confusion_matrix[other][lbl] for other in labels if other != lbl)
        fn = sum(confusion_matrix[lbl][other] for other in labels if other != lbl)

        prec = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        support = sum(confusion_matrix[lbl].values())
        per_class_metrics[lbl] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support
        }
        if support > 0:
            macro_p += prec
            macro_r += rec
            macro_f1 += f1
            valid_classes += 1

    macro_p = macro_p / valid_classes if valid_classes > 0 else 0.0
    macro_r = macro_r / valid_classes if valid_classes > 0 else 0.0
    macro_f1 = macro_f1 / valid_classes if valid_classes > 0 else 0.0

    return {
        "total_cases": total_cases,
        "passed_cases": passed_count,
        "failed_cases": total_cases - passed_count,
        "accuracy": accuracy,
        "false_compliant_count": false_compliant_count,
        "false_compliant_rate": (false_compliant_count / total_cases) * 100.0 if total_cases > 0 else 0.0,
        "category_stats": dict(category_stats),
        "confusion_matrix": confusion_matrix,
        "per_class_metrics": per_class_metrics,
        "macro_metrics": {
            "precision": macro_p,
            "recall": macro_r,
            "f1": macro_f1
        },
        "results": results
    }


if __name__ == "__main__":
    report = evaluate_adversarial_suite()
    print("=" * 65)
    print(f"ADVERSARIAL STRESS TEST SUMMARY: {report['passed_cases']}/{report['total_cases']} Passed ({report['accuracy']:.1f}%)")
    print(f"Critical Safety Failures (False Compliant): {report['false_compliant_count']}")
    print("=" * 65)

    print("\n--- CATEGORY BREAKDOWN ---")
    for cat, stats in sorted(report["category_stats"].items()):
        acc = (stats["passed"] / stats["total"]) * 100.0
        print(f"  {cat:<45}: {stats['passed']}/{stats['total']} ({acc:.1f}%)")

    print("\n--- PER-CLASS METRICS ---")
    for lbl, m in report["per_class_metrics"].items():
        print(f"  {lbl:<18}: Prec={m['precision']:.1f}%, Rec={m['recall']:.1f}%, F1={m['f1']:.1f}% (Support={m['support']})")

    print(f"\nMacro F1: {report['macro_metrics']['f1']:.1f}%")
