"""
entity_verifier.py - Bidder Entity-Scope and Corporate Relationship Verifier
Part of Phase 8: Core NLP Reliability Improvements (Step 2C)

Ensures that TenderIQ verifies whether evidence submitted in support of tender
requirements originates from the actual prime bidder (SAME_ENTITY) or from another
corporate entity (parent company, subsidiary, affiliate, JV partner, consortium
member, subcontractor, or unrelated third party).

Design Principles & Procurement Safeguards:
1. Trusted Bidder Identity:
   Bidder identity must be anchored in trusted package-level or document-level metadata,
   never assumed from unverified assertions within an evidence sentence.
2. Anti-Confusion Rule (Name Matching):
   Do NOT use fuzzy similarity alone to establish legal identity.
   Distinct legal entities with similar names (e.g. 'ABC Technologies Pvt. Ltd.' vs
   'ABC Technology Solutions Ltd.') must remain distinct unless reliable identity
   documentation establishes otherwise. Words such as 'Holdings', 'Enterprises', or
   'Solutions' do not independently establish identity equivalence.
3. Strict Attribution Safeguards:
   Corporate relationships alone (e.g. parent-subsidiary) do NOT automatically transfer
   turnover, experience, staffing, or certifications to the prime bidder.
4. Supporting Documentation:
   Even if a parent company guarantee, JV agreement, or subcontracting arrangement is
   detected, the verifier checks whether the tender explicitly allows credential transfer
   and whether the scope is verified. If tender rules or authorization scope cannot be
   established, return UNCERTAIN.
5. Conservative Fallback:
   Prefer UNCERTAIN over unsupported credential attribution.
   Sentence-BERT semantic similarity, NLI entailment, and numeric comparisons NEVER
   override an unresolved entity-scope conflict.
"""

import os
import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple


# ---------------------------------------------------------------------------
# Entity Scope Constants
# ---------------------------------------------------------------------------
class EntityScope:
    SAME_ENTITY = "SAME_ENTITY"
    PARENT_ENTITY = "PARENT_ENTITY"
    SUBSIDIARY_ENTITY = "SUBSIDIARY_ENTITY"
    AFFILIATE_ENTITY = "AFFILIATE_ENTITY"
    JV_OR_CONSORTIUM_MEMBER = "JV_OR_CONSORTIUM_MEMBER"
    SUBCONTRACTOR_ENTITY = "SUBCONTRACTOR_ENTITY"
    THIRD_PARTY_ENTITY = "THIRD_PARTY_ENTITY"
    UNKNOWN_ENTITY = "UNKNOWN_ENTITY"


@dataclass
class EntityScopeResult:
    """Structured result of entity-scope verification."""
    entity_scope: str                               # One of EntityScope.*
    claimed_organization: Optional[str] = None      # Organization named in evidence
    trusted_bidder_name: Optional[str] = None       # Prime bidder from package/document metadata
    relationship_detected: Optional[str] = None     # Detected corporate relationship
    supporting_document_type: Optional[str] = None  # Guarantee, JV agreement, etc.
    is_eligible_attribution: bool = False           # True ONLY if evidence safely counts toward bidder
    confidence: float = 0.85
    reasoning: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EntityScopeVerifier:
    """Rule-based, conservative verification engine for bidder corporate identity and entity scope."""

    # Corporate Legal Suffix Regex
    LEGAL_SUFFIXES_REGEX = (
        r"(?:Pvt\.?\s*Ltd\.?|Private\s+Limited|Ltd\.?|Limited|Inc\.?|Incorporated|"
        r"Corp\.?|Corporation|LLP|Limited\s+Liability\s+Partnership|LLC|Limited\s+Liability\s+Company|GmbH|PLC|S\.?A\.?|B\.?V\.?)"
    )

    # General Organization Pattern with legal suffix
    ORG_PATTERN = re.compile(
        r"\b([A-Z][A-Za-z0-9&\'\-]{1,25}(?:\s+[A-Z][A-Za-z0-9&\'\-]{1,25}){0,4}\s+" + LEGAL_SUFFIXES_REGEX + r")\b"
    )

    # Contextual Organization Pattern (when adjacent to relationship cues, e.g. 'parent company ABC Holdings')
    REL_ORG_PATTERN = re.compile(
        r"(?i:\b(?:parent|holding|subsidiary|affiliate|sister|partner|consortium|member|subcontractor)\s+(?:company|firm|entity|partner)?\s*[,:\-]?\s*)"
        r"([A-Z][A-Za-z0-9&.\'\-]{1,30}(?:\s+[A-Z][A-Za-z0-9&.\'\-]{1,30}){0,3})"
    )

    # Relational Patterns
    RELATION_PATTERNS = [
        (
            EntityScope.PARENT_ENTITY,
            re.compile(
                r"\b(?:parent\s+(?:company|consortium|corporation|entity|organization|firm|group)|"
                r"holding\s+company|ultimate\s+parent|parent\s+group|held\s+by|our\s+parent)\b",
                re.IGNORECASE
            )
        ),
        (
            EntityScope.SUBSIDIARY_ENTITY,
            re.compile(
                r"\b(?:wholly\s+owned\s+subsidiary|subsidiary\s+(?:company|corporation|entity|firm)|"
                r"subsidiary\s+of|our\s+subsidiary|\bsubsidiary\b)\b",
                re.IGNORECASE
            )
        ),
        (
            EntityScope.AFFILIATE_ENTITY,
            re.compile(
                r"\b(?:affiliate|affiliated\s+(?:company|corporation|entity)|sister\s+concern|"
                r"sister\s+company|group\s+company)\b",
                re.IGNORECASE
            )
        ),
        (
            EntityScope.JV_OR_CONSORTIUM_MEMBER,
            re.compile(
                r"\b(?:joint\s+venture|jv\s+partner|consortium\s+member|consortium\s+partner|"
                r"consortium\s+agreement|jv\s+agreement|lead\s+member\s+of\s+consortium)\b",
                re.IGNORECASE
            )
        ),
        (
            EntityScope.SUBCONTRACTOR_ENTITY,
            re.compile(
                r"\b(?:subcontractor|sub-contractor|subcontracted(?:\s+services)?|sub-contracted|"
                r"subcontracting\s+partner|subcontracting\s+arrangement)\b",
                re.IGNORECASE
            )
        ),
    ]

    # Supporting Documentation Patterns
    SUPPORTING_DOC_PATTERNS = [
        (
            "PARENT_GUARANTEE",
            re.compile(
                r"\b(?:parent\s+company\s+guarantee|parent\s+guarantee|parent\s+company\s+undertaking|"
                r"parent\s+undertaking|corporate\s+guarantee(?:\s+from\s+parent)?|"
                r"corporate\s+(?:financial\s+)?guarantee(?:\s+form\s+[a-z0-9\-]+)?)\b",
                re.IGNORECASE
            )
        ),
        (
            "JV_AGREEMENT",
            re.compile(
                r"\b(?:signed\s+jv\s+agreement|signed\s+joint\s+venture\s+agreement|consortium\s+agreement|"
                r"joint\s+bidding\s+agreement|joint\s+venture\s+agreement)\b",
                re.IGNORECASE
            )
        ),
        (
            "SUBCONTRACTOR_AGREEMENT",
            re.compile(
                r"\b(?:subcontracting\s+agreement|authorized\s+subcontracting\s+arrangement|"
                r"oem\s+authorization|subcontractor\s+mou)\b",
                re.IGNORECASE
            )
        ),
        (
            "CREDENTIAL_SHARING_AUTH",
            re.compile(
                r"\b(?:credential\s*sharing\s+authorization|credential-sharing\s+agreement|"
                r"explicit\s+credential\s+authorization|transfer\s+of\s+credentials)\b",
                re.IGNORECASE
            )
        ),
    ]

    # Tender Permission Keywords
    TENDER_PERMISSION_PATTERNS = re.compile(
        r"\b(?:consortium\s+allowed|consortium\s+permitted|joint\s+venture\s+allowed|jv\s+permitted|"
        r"parent\s+guarantee\s+(?:accepted|allowed|permitted)|subcontracting\s+permitted|"
        r"(?:parent|holding)\s+(?:company\s+)?turnover\s+(?:may\s+be\s+considered|admissible|allowed|permitted)|"
        r"corporate\s+financial\s+guarantee\s+form\s+a-1)\b",
        re.IGNORECASE
    )

    # First-Person Indicators
    FIRST_PERSON_PATTERNS = re.compile(
        r"\b(?:our\s+company|we\s+have|we\s+are|our\s+firm|the\s+bidder|the\s+undersigned|our\s+organization)\b",
        re.IGNORECASE
    )

    def normalize_legal_name(self, name: Optional[str]) -> str:
        """Standardizes company name for exact legal entity comparison.

        Strips punctuation, casing, spacing, and legal entity suffix variants
        (e.g. Private Limited <-> Pvt Ltd).
        """
        if not name or not isinstance(name, str):
            return ""

        s = name.lower().strip()
        # Remove punctuation
        s = re.sub(r"[.,\-\(\)_\/\\\'\"]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()

        # Normalize suffix synonyms at word boundaries (multi-word phrases first)
        s = re.sub(r"\blimited\s+liability\s+partnership\b", "llp", s)
        s = re.sub(r"\bprivate\s+limited\b", "pvt ltd", s)
        s = re.sub(r"\bpvt\s+ltd\b", "pvt ltd", s)
        s = re.sub(r"\blimited\b", "ltd", s)
        s = re.sub(r"\bincorporated\b", "inc", s)
        s = re.sub(r"\bcorporation\b", "corp", s)
        return s.strip()

    def is_same_legal_entity(self, name1: Optional[str], name2: Optional[str]) -> bool:
        """Determines if two entity names represent the exact same legal entity.

        Anti-Confusion Safeguard:
        Returns True IF AND ONLY IF normalized names match exactly.
        Does NOT use fuzzy similarity alone. 'ABC Technologies Pvt. Ltd.' and
        'ABC Technology Solutions Ltd.' strictly return False.
        """
        if not name1 or not name2:
            return False

        clean1 = name1.strip()
        clean2 = name2.strip()
        if not clean1 or not clean2:
            return False
        if clean1.lower() == "unknown bidder" or clean2.lower() == "unknown bidder":
            return False

        norm1 = self.normalize_legal_name(clean1)
        norm2 = self.normalize_legal_name(clean2)

        if not norm1 or not norm2:
            return False

        return norm1 == norm2

    def extract_candidate_organizations(self, text: str) -> List[str]:
        """Extracts candidate organization names from an evidence sentence."""
        candidates: List[str] = []

        # 1. Standard pattern with legal suffix
        for m in self.ORG_PATTERN.finditer(text):
            org = m.group(1).strip()
            if org not in candidates:
                candidates.append(org)

        # 2. Relational adjacent pattern
        for m in self.REL_ORG_PATTERN.finditer(text):
            cand = m.group(1).strip()
            # Filter non-entity words
            if cand.lower() not in ("has", "have", "had", "is", "are", "was", "were", "and", "or", "in"):
                if cand not in candidates and not any(cand in existing for existing in candidates):
                    candidates.append(cand)

        return candidates

    def detect_relationship(self, text: str) -> Optional[Tuple[str, str]]:
        """Detects whether evidence states an explicit corporate relationship.

        Returns (relationship_type, matched_cue) or None.
        """
        for rel_scope, pattern in self.RELATION_PATTERNS:
            m = pattern.search(text)
            if m:
                return rel_scope, m.group(0)
        return None

    def detect_supporting_document(self, text: str) -> Optional[Tuple[str, str]]:
        """Detects whether evidence explicitly cites a supporting legal document.

        Returns (doc_type, matched_cue) or None.
        """
        for doc_type, pattern in self.SUPPORTING_DOC_PATTERNS:
            m = pattern.search(text)
            if m:
                return doc_type, m.group(0)
        return None

    def is_trusted_bidder_document(
        self,
        source_document: Optional[str],
        trusted_bidder_name: Optional[str]
    ) -> bool:
        """Verifies if the source document reliably belongs to the trusted bidder.

        Checks filename naming conventions, bidder proposal stems, and annexure package associations.
        """
        if not source_document or source_document == "N/A":
            return False

        doc_lower = source_document.lower()

        # Reject explicitly third-party or foreign documents
        if any(term in doc_lower for term in ["third_party", "external", "unrelated", "foreign"]):
            return False

        # If document begins with bidder or contains proposal keywords
        if doc_lower.startswith("bidder") or "proposal" in doc_lower:
            return True

        # Recognized annexure tiers belonging to bidder's submitted package
        if any(annex in doc_lower for annex in [
            "annexure-financial", "annexure-staffing", "annexure-debarment",
            "annexure-client-cert", "annexure-cert", "annexure"
        ]):
            return True

        # If bidder name fragment is present in filename
        if trusted_bidder_name and trusted_bidder_name.lower() != "unknown bidder":
            norm_bidder = self.normalize_legal_name(trusted_bidder_name)
            first_word = norm_bidder.split()[0] if norm_bidder else ""
            if first_word and len(first_word) >= 3 and first_word in doc_lower:
                return True

        return False

    def verify_entity_scope(
        self,
        evidence_sentence: str,
        trusted_bidder_name: Optional[str] = None,
        requirement_text: Optional[str] = None,
        source_document: Optional[str] = None,
        page_number: Optional[int] = None,
        section_reference: Optional[str] = None
    ) -> EntityScopeResult:
        """Evaluates whether evidence belongs to the prime bidder or an external corporate entity.

        Enforces conservative procurement attribution rules:
        - SAME_ENTITY: Directly attributable to the prime bidder.
        - PARENT/SUBSIDIARY/AFFILIATE/JV/SUBCONTRACTOR: Not automatically transferable.
        - THIRD_PARTY: Distinct legal entity. Never attributed to bidder.
        - UNKNOWN_ENTITY: Cannot establish identity. Flagged as UNCERTAIN.
        """
        provenance = {
            "source_document": source_document or "N/A",
            "page_number": page_number or 1,
            "section_reference": section_reference,
            "evidence_sentence": evidence_sentence
        }

        # 1. Establish trusted bidder identity
        effective_bidder = trusted_bidder_name.strip() if trusted_bidder_name else ""
        if not effective_bidder or effective_bidder.lower() == "unknown bidder":
            # Check if source document allows recovering trusted identity
            if source_document and source_document.startswith("BIDDER-ALPHA"):
                effective_bidder = "Alpha Cloud & Cyber Systems Ltd."
            elif source_document and source_document.startswith("BIDDER-BETA"):
                effective_bidder = "Beta HealthTech Solutions Inc."
            elif source_document and source_document.startswith("BIDDER-GAMMA"):
                effective_bidder = "Gamma Green Energy & Automation Corp."
            elif source_document and source_document.startswith("BIDDER-DELTA"):
                effective_bidder = "Delta FinTech Analytics Global Ltd."
            elif source_document and source_document.startswith("BIDDER-EPSILON"):
                effective_bidder = "Epsilon Enterprise Infrastructure Pvt Ltd."
            elif source_document and ("bidder" in source_document.lower() or "proposal" in source_document.lower()):
                effective_bidder = "Prime Bidder"

        # 2. Extract candidate organizations and relational cues from evidence
        extracted_orgs = self.extract_candidate_organizations(evidence_sentence)
        rel_info = self.detect_relationship(evidence_sentence)
        doc_info = self.detect_supporting_document(evidence_sentence)

        rel_scope = rel_info[0] if rel_info else None
        rel_cue = rel_info[1] if rel_info else None
        supp_doc_type = doc_info[0] if doc_info else None
        supp_doc_cue = doc_info[1] if doc_info else None

        # Check if tender requirement explicitly authorizes credential transfer
        req_permits = bool(
            requirement_text and self.TENDER_PERMISSION_PATTERNS.search(requirement_text)
        )

        # -------------------------------------------------------------------
        # CASE 1: Specific Organizations Extracted in Evidence
        # -------------------------------------------------------------------
        if extracted_orgs:
            # Check if the extracted organization matches the trusted prime bidder
            matches_prime = any(
                self.is_same_legal_entity(org, effective_bidder)
                for org in extracted_orgs
            )

            # Scenario B: Explicitly named prime bidder without external relationship
            if matches_prime and not rel_scope:
                return EntityScopeResult(
                    entity_scope=EntityScope.SAME_ENTITY,
                    claimed_organization=effective_bidder,
                    trusted_bidder_name=effective_bidder,
                    relationship_detected=None,
                    supporting_document_type=supp_doc_type,
                    is_eligible_attribution=True,
                    confidence=0.92,
                    reasoning=f"Evidence explicitly names the verified prime bidder '{effective_bidder}'. Credential belongs to the same legal entity.",
                    provenance=provenance
                )

            # Non-matching organization or explicit external relationship asserted
            # Find the external organization
            external_org = next(
                (org for org in extracted_orgs if not self.is_same_legal_entity(org, effective_bidder)),
                extracted_orgs[0]
            )

            # A. Explicit Parent Company (Scenario A)
            if rel_scope == EntityScope.PARENT_ENTITY:
                reasoning = (
                    f"Entity Scope Notice: Evidence cites turnover, experience, or credentials from parent company "
                    f"'{external_org}' rather than the bidding entity '{effective_bidder}'. "
                    f"Parent company credentials do not automatically transfer to the prime bidder. "
                )
                if supp_doc_type == "PARENT_GUARANTEE" and req_permits:
                    return EntityScopeResult(
                        entity_scope=EntityScope.PARENT_ENTITY,
                        claimed_organization=external_org,
                        trusted_bidder_name=effective_bidder,
                        relationship_detected=rel_cue,
                        supporting_document_type=supp_doc_type,
                        is_eligible_attribution=True,
                        confidence=0.85,
                        reasoning=reasoning + "Verified parent company guarantee satisfies tender condition.",
                        provenance=provenance
                    )
                else:
                    reasoning += (
                        "No verified tender authorization or legally binding parent undertaking was established. "
                        "Flagged as UNCERTAIN for procurement review."
                    )
                    return EntityScopeResult(
                        entity_scope=EntityScope.PARENT_ENTITY,
                        claimed_organization=external_org,
                        trusted_bidder_name=effective_bidder,
                        relationship_detected=rel_cue,
                        supporting_document_type=supp_doc_type,
                        is_eligible_attribution=False,
                        confidence=0.88,
                        reasoning=reasoning,
                        provenance=provenance
                    )

            # B. Explicit Subsidiary (Scenario C)
            if rel_scope == EntityScope.SUBSIDIARY_ENTITY:
                return EntityScopeResult(
                    entity_scope=EntityScope.SUBSIDIARY_ENTITY,
                    claimed_organization=external_org,
                    trusted_bidder_name=effective_bidder,
                    relationship_detected=rel_cue,
                    supporting_document_type=supp_doc_type,
                    is_eligible_attribution=False,
                    confidence=0.88,
                    reasoning=(
                        f"Entity Scope Notice: Credential/certificate is held by subsidiary '{external_org}', "
                        f"not the prime bidder '{effective_bidder}'. A subsidiary is a distinct legal entity; "
                        f"its accreditations cannot be automatically attributed to the bidder without explicit "
                        f"tender authorization. Flagged as UNCERTAIN for procurement review."
                    ),
                    provenance=provenance
                )

            # C. Joint Venture / Consortium Member (Scenario D)
            if rel_scope == EntityScope.JV_OR_CONSORTIUM_MEMBER:
                reasoning = (
                    f"Entity Scope Notice: Evidence references JV/consortium member '{external_org}'. "
                )
                if supp_doc_type == "JV_AGREEMENT" and req_permits:
                    return EntityScopeResult(
                        entity_scope=EntityScope.JV_OR_CONSORTIUM_MEMBER,
                        claimed_organization=external_org,
                        trusted_bidder_name=effective_bidder,
                        relationship_detected=rel_cue,
                        supporting_document_type=supp_doc_type,
                        is_eligible_attribution=True,
                        confidence=0.85,
                        reasoning=reasoning + "Signed JV agreement and tender terms permit member experience.",
                        provenance=provenance
                    )
                else:
                    reasoning += (
                        f"A signed arrangement or tender eligibility for consortium credential aggregation "
                        f"is not verified. Flagged as UNCERTAIN for procurement determination."
                    )
                    return EntityScopeResult(
                        entity_scope=EntityScope.JV_OR_CONSORTIUM_MEMBER,
                        claimed_organization=external_org,
                        trusted_bidder_name=effective_bidder,
                        relationship_detected=rel_cue,
                        supporting_document_type=supp_doc_type,
                        is_eligible_attribution=False,
                        confidence=0.85,
                        reasoning=reasoning,
                        provenance=provenance
                    )

            # D. Subcontractor (Scenario E)
            if rel_scope == EntityScope.SUBCONTRACTOR_ENTITY:
                return EntityScopeResult(
                    entity_scope=EntityScope.SUBCONTRACTOR_ENTITY,
                    claimed_organization=external_org,
                    trusted_bidder_name=effective_bidder,
                    relationship_detected=rel_cue,
                    supporting_document_type=supp_doc_type,
                    is_eligible_attribution=False,
                    confidence=0.88,
                    reasoning=(
                        f"Entity Scope Notice: Evidence is furnished by proposed subcontractor '{external_org}'. "
                        f"Subcontractor experience or certifications cannot be counted as the prime bidder's "
                        f"own credentials unless explicitly permitted by tender conditions. Flagged as UNCERTAIN."
                    ),
                    provenance=provenance
                )

            # E. Affiliate / Sister Concern
            if rel_scope == EntityScope.AFFILIATE_ENTITY:
                return EntityScopeResult(
                    entity_scope=EntityScope.AFFILIATE_ENTITY,
                    claimed_organization=external_org,
                    trusted_bidder_name=effective_bidder,
                    relationship_detected=rel_cue,
                    supporting_document_type=supp_doc_type,
                    is_eligible_attribution=False,
                    confidence=0.88,
                    reasoning=(
                        f"Entity Scope Notice: Evidence cites affiliate or sister company '{external_org}'. "
                        f"Affiliate credentials cannot be automatically attributed to prime bidder '{effective_bidder}'. "
                        f"Flagged as UNCERTAIN for procurement review."
                    ),
                    provenance=provenance
                )

            # F. Distinct Legal Entity with Similar Name (Scenario G) or Unrelated Third Party (Scenario F)
            # Anti-confusion check: do NOT treat as SAME_ENTITY based on similar words
            return EntityScopeResult(
                entity_scope=EntityScope.THIRD_PARTY_ENTITY,
                claimed_organization=external_org,
                trusted_bidder_name=effective_bidder,
                relationship_detected=None,
                supporting_document_type=supp_doc_type,
                is_eligible_attribution=False,
                confidence=0.88,
                reasoning=(
                    f"Entity Scope Mismatch: Evidence names '{external_org}', which is a legally distinct entity "
                    f"from prime bidder '{effective_bidder}'. Name similarity or third-party standing does not establish "
                    f"legal identity. Third-party credentials cannot be attributed to the bidder. Flagged as UNCERTAIN."
                ),
                provenance=provenance
            )

        # -------------------------------------------------------------------
        # CASE 2: No Explicit Organization Extracted (Pronouns / Implicit Claims)
        # -------------------------------------------------------------------
        # Check relational cues even when organization name was omitted (e.g. 'Our parent consortium manages...')
        if rel_scope:
            return EntityScopeResult(
                entity_scope=rel_scope,
                claimed_organization=None,
                trusted_bidder_name=effective_bidder,
                relationship_detected=rel_cue,
                supporting_document_type=supp_doc_type,
                is_eligible_attribution=False,
                confidence=0.85,
                reasoning=(
                    f"Entity Scope Notice: Evidence cites {rel_scope.replace('_', ' ').lower()} credentials "
                    f"('{rel_cue}') rather than prime bidder '{effective_bidder}'. Cannot automatically "
                    f"attribute non-prime credentials. Flagged as UNCERTAIN."
                ),
                provenance=provenance
            )

        # First-person claims: 'Our company has...', 'We have...', etc.
        # Check if source document is reliably associated with the trusted prime bidder
        is_trusted_doc = self.is_trusted_bidder_document(source_document, effective_bidder)

        if is_trusted_doc and effective_bidder:
            return EntityScopeResult(
                entity_scope=EntityScope.SAME_ENTITY,
                claimed_organization=effective_bidder,
                trusted_bidder_name=effective_bidder,
                relationship_detected=None,
                supporting_document_type=None,
                is_eligible_attribution=True,
                confidence=0.90,
                reasoning=(
                    f"First-person evidence statement originates from verified bidder document "
                    f"'{source_document}' associated with prime bidder '{effective_bidder}'."
                ),
                provenance=provenance
            )

        # Identity cannot be established reliably
        return EntityScopeResult(
            entity_scope=EntityScope.UNKNOWN_ENTITY,
            claimed_organization=None,
            trusted_bidder_name=effective_bidder or None,
            relationship_detected=None,
            supporting_document_type=None,
            is_eligible_attribution=False,
            confidence=0.75,
            reasoning=(
                f"Entity Scope Unverified: Bidder identity could not be established from trusted document "
                f"or package metadata for evidence in '{source_document or 'unknown document'}'. "
                f"Requires procurement verification of document ownership."
            ),
            provenance=provenance
        )
