"""
cross_document_reconciler.py - Phase 7: Multi-Document Evidence Reconciliation

Detects conflicting evidence claims for the same requirement across a bidder's
multiple submitted documents (main proposal, financial annexures, staffing schedules,
client completion certificates, etc.).

Design Principles:
1. Evidence provenance is always preserved (filename, page_number, section_reference).
2. Authoritative source policy: audited/certified annexures supersede self-declared
   main-proposal narrative claims on quantitative criteria.
3. Conflicting evidence is NEVER silently resolved. It is flagged as UNCERTAIN or
   NON_COMPLIANT depending on evidence quality, with reasoning for auditor review.
4. NLI confidence and semantic similarity NEVER override deterministic numeric conflict.
5. Distinct from MISSING_EVIDENCE: conflicting evidence means something was SUBMITTED
   but is internally inconsistent across documents.
6. A consistent, passing value in ALL available sources → retains the baseline label.

Covered Conflict Types:
A. Numeric Magnitude Conflict: Main proposal claims X; annexure reports Y where Y < threshold.
B. Date/Expiry Conflict: Certificate claimed valid; annexure shows it expired before tender date.
C. Staffing Allocation Conflict: Named PM proposed with 12 yrs; staffing schedule shows 10% FTE.
D. Reference Project Count Conflict: Summary claims N projects; certificates show M < N.
E. Subsidiary Debarment Conflict: Clean record declared; debarment list names related entity.

NOT in scope for Phase 7 (Phase 8+):
- OCR of scanned PDFs
- Fine-tuned NLI models
- Entity co-reference resolution across unrelated entity names
- Cross-tender multi-bidder comparison
"""

import logging
import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source Provenance & Tier Classification
# ---------------------------------------------------------------------------

# Document tiers: lower tier_rank = more authoritative
DOCUMENT_TIER_PATTERNS: List[Tuple[re.Pattern, int, str]] = [
    # Audited financial annexures — highest authority on financial figures
    (re.compile(r"ANNEXURE[-_\s]*F|FINANCIAL[-_\s]*STATEMENT|AUDITED[-_\s]*BALANCE", re.IGNORECASE), 1, "audited_financial"),
    # Staffing schedule annexures — authority on FTE allocation
    (re.compile(r"ANNEXURE[-_\s]*G|STAFFING[-_\s]*SCHEDULE", re.IGNORECASE), 1, "staffing_schedule"),
    # Client completion certificate annexures — authority on project count
    (re.compile(r"ANNEXURE[-_\s]*E|CLIENT[-_\s]*CERT|COMPLETION[-_\s]*CERT", re.IGNORECASE), 1, "completion_certificates"),
    # OEM / MAF certificate annexures
    (re.compile(r"ANNEXURE[-_\s]*[A-Z]|ANNEXURE[-_\s]*CERT", re.IGNORECASE), 1, "annexure_certificate"),
    # Debarment / blacklist lists
    (re.compile(r"DEBARMENT|BLACKLIST|SUSPENSION[-_\s]*LIST", re.IGNORECASE), 1, "debarment_list"),
    # Generic annexures (slightly lower authority than named ones)
    (re.compile(r"ANNEXURE|ANNEX[-_\s]*[A-Z]?", re.IGNORECASE), 2, "generic_annexure"),
    # Main technical proposal / narrative — lowest authority on verified figures
    (re.compile(r"PROPOSAL|TECHNICAL[-_\s]*BID|BID[-_\s]*SUBMISSION", re.IGNORECASE), 3, "main_proposal"),
]

PROPOSAL_LABEL = "main_proposal"
ANNEXURE_TIERS = {"audited_financial", "staffing_schedule", "completion_certificates",
                  "annexure_certificate", "debarment_list", "generic_annexure"}


def classify_document_tier(source_document: str) -> Tuple[int, str]:
    """Returns (tier_rank, tier_label) for a source document filename.

    Lower tier_rank = more authoritative evidence source.
    """
    if not source_document:
        return 5, "unknown"
    for pattern, rank, label in DOCUMENT_TIER_PATTERNS:
        if pattern.search(source_document):
            return rank, label
    # Default: treat as main proposal
    return 3, PROPOSAL_LABEL


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class DocumentEvidenceGroup:
    """All evidence sentences from a single source document relevant to one requirement."""
    source_document: str
    tier_rank: int
    tier_label: str
    sentences: List[str] = field(default_factory=list)
    numeric_values: List[Dict[str, Any]] = field(default_factory=list)  # combined from all sentences
    page_numbers: List[int] = field(default_factory=list)
    section_references: List[Optional[str]] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)

    @property
    def is_authoritative(self) -> bool:
        return self.tier_rank <= 2

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConflictingClaimPair:
    """Represents a detected conflict between two evidence sources for the same criterion."""
    conflict_type: str                        # 'numeric_magnitude', 'staffing_allocation', 'project_count',
                                              # 'certificate_expiry', 'debarment_disclosure', 'general_contradiction'
    criterion: str                            # Description of what is being contested
    primary_source: str                       # Less authoritative source (e.g. main proposal)
    primary_value: Optional[float]
    primary_unit: Optional[str]
    primary_text: str
    primary_page: Optional[int]
    secondary_source: str                     # More authoritative source (e.g. annexure)
    secondary_value: Optional[float]
    secondary_unit: Optional[str]
    secondary_text: str
    secondary_page: Optional[int]
    resolution_policy: str                    # 'annexure_supersedes_proposal', 'flag_for_auditor', 'both_sources_consistent'
    recommended_label: str                    # 'NON_COMPLIANT', 'UNCERTAIN', or 'PRESERVE_BASELINE'
    confidence: float
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReconciliationResult:
    """Output of cross-document reconciliation for a single requirement."""
    requirement_id: str
    has_conflict: bool                        # True if any conflicting claim detected
    conflict_severity: str                    # 'NONE', 'SOFT' (flag for review), 'HARD' (override to NON_COMPLIANT)
    conflicts: List[ConflictingClaimPair] = field(default_factory=list)
    document_groups: List[DocumentEvidenceGroup] = field(default_factory=list)
    reconciled_label: Optional[str] = None   # Override label, or None to preserve baseline
    reconciled_confidence: float = 0.0
    reconciled_reasoning: str = ""
    annexure_sources_checked: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Numeric Extraction Helpers
# ---------------------------------------------------------------------------

# Requires either currency indicator ($ / USD / US$) OR magnitude indicator (Million / Billion / M / B)
# so plain numbers like "36-month" or "99.99%" or "10 Gbps" are never parsed as USD millions.
_FINANCIAL_RE = re.compile(
    r"(?:(?P<curr>USD|US\$|\$)\s*(?P<val1>\d+(?:\.\d+)?)\s*(?P<mult1>Million|Billion|M|B|K)?)|"
    r"(?:(?P<val2>\d+(?:\.\d+)?)\s*(?P<mult2>Million|Billion|M|B))",
    re.IGNORECASE
)

# Matches: "10%" or "10 %"
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")

# Matches: "10% FTE", "10 percent FTE"
_FTE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*FTE|(\d+(?:\.\d+)?)\s*percent\s*FTE", re.IGNORECASE)

# Matches counts: "1 project", "4 major qualifying projects", "1 signed completion certificate"
_COUNT_RE = re.compile(
    r"(\d+)\s+(?:(?:signed|completed|completion|major|qualifying|reference|client|similar)\s+)*(?:project|certificate|contract|deployment|reference)s?",
    re.IGNORECASE
)


# Contextual patterns for audited/certified turnover and revenue
_AUDITED_TURNOVER_PATTERNS = [
    # 'audited ... average turnover registered at USD 6.2 Million'
    re.compile(r'(?:audited|official|certified|statutory)\s+(?:(?:\d+[- ]year\s+)?average\s+)?(?:annual\s+)?turnover\s+(?:officially\s+)?(?:registered\s+)?(?:as|at|of|stands\s+at)?\s*(?:only\s+)?(?:USD|US\$|\$)?\s*(\d+(?:\.\d+)?)\s*(Million|Billion|M|B)?', re.I),
    # 'average annual turnover for the past three fiscal years stands at USD 14.5 Million'
    re.compile(r'(?:average\s+)?(?:annual\s+)?turnover\b.*?\b(?:stands\s+at|is|registers?|registered\s+at|of|at)\s+(?:only\s+)?(?:USD|US\$|\$)?\s*(\d+(?:\.\d+)?)\s*(Million|Billion|M|B)?', re.I),
    # 'USD 32.0 Million turnover / revenue / net worth'
    re.compile(r'(?:USD|US\$|\$)?\s*(\d+(?:\.\d+)?)\s*(Million|Billion|M|B)\s*(?:average\s+)?(?:annual\s+)?(?:turnover|revenue|net\s+worth)', re.I),
    # 'claims USD 32.0 Million'
    re.compile(r'claims?\s+(?:USD|US\$|\$)?\s*(\d+(?:\.\d+)?)\s*(Million|Billion|M|B)', re.I),
]


def _parse_usd_millions(text: str) -> Optional[float]:
    """Extracts the primary USD value in millions from text.
    
    Prefers explicitly contextualized audited/registered turnover figures over
    disputed narrative claims in parenthetical notes. Handles multiple ambiguous
    figures conservatively by returning None.
    """
    if not text:
        return None

    # Step 1: Strip parenthetical disputed notes / disclaimers
    # e.g. '(Note: Narrative claims of USD 14.5 Million in external executive summaries include unexecuted pipeline estimates...)'
    cleaned = re.sub(r'\(Note:[^)]*\)', '', text, flags=re.IGNORECASE)
    cleaned = re.sub(r'\([^)]*(?:narrative|claim|unexecuted|pipeline|dispute)[^)]*\)', '', cleaned, flags=re.IGNORECASE)

    # Step 2: Try explicit contextual turnover patterns in cleaned text
    turnover_matches = []
    for pat in _AUDITED_TURNOVER_PATTERNS:
        for m in pat.finditer(cleaned):
            val = float(m.group(1))
            mult = (m.group(2) or '').upper()
            if mult in ('BILLION', 'B'):
                val *= 1000.0
            elif mult in ('K',):
                val /= 1000.0
            turnover_matches.append(val)

    unique_turnovers = list(dict.fromkeys(turnover_matches))
    if len(unique_turnovers) == 1:
        return unique_turnovers[0]
    elif len(unique_turnovers) > 1:
        # Multiple conflicting audited/turnover figures: return None (uncertain review signal)
        return None

    # Step 3: General financial value extraction from cleaned text
    found_vals = []
    for m in _FINANCIAL_RE.finditer(cleaned):
        val_str = m.group('val1') or m.group('val2')
        mult_str = ((m.group('mult1') or m.group('mult2')) or '').upper()
        if not val_str:
            continue
        v = float(val_str)
        if mult_str in ('BILLION', 'B'):
            v *= 1000.0
        elif mult_str in ('K',):
            v /= 1000.0
        found_vals.append(v)

    if not found_vals:
        return None

    # If all found values are identical or only one distinct value exists
    unique_vals = list(dict.fromkeys(found_vals))
    if len(unique_vals) == 1:
        return unique_vals[0]

    # Multiple distinct numbers without explicit turnover attribution:
    # Ambiguous - return None to signal uncertainty rather than guessing max
    return None


def _parse_fte_percent(text: str) -> Optional[float]:
    """Extracts FTE% from text like '10% FTE'. Returns None if not found."""
    m = _FTE_RE.search(text)
    if m:
        return float(m.group(1) or m.group(2))
    return None


def _parse_count(text: str) -> Optional[int]:
    """Extracts a document count (projects, certificates, etc.) from text."""
    m = _COUNT_RE.search(text)
    if m:
        return int(m.group(1))
    return None


def _parse_year(text: str) -> Optional[int]:
    """Extracts a 4-digit year from text (e.g. expiry date)."""
    m = re.search(r"\b(20\d{2}|19\d{2})\b", text)
    if m:
        return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Main Reconciler Class
# ---------------------------------------------------------------------------

class CrossDocumentReconciler:
    """Detects and reports conflicting evidence across multiple bidder documents.

    Usage in pipeline:
        reconciler = CrossDocumentReconciler()
        result = reconciler.reconcile(
            requirement_id=req_id,
            requirement_text=req_text,
            requirement_value=req_val,
            requirement_unit=req_unit,
            all_candidates=all_top_k_candidates,  # from all evidence across docs
        )
    """

    # Minimum FTE % to consider a named person "dedicated" to this tender
    MIN_FTE_FOR_DEDICATED: float = 50.0

    def __init__(
        self,
        annexure_supersedes_proposal: bool = True,
        min_magnitude_conflict_fraction: float = 0.20,  # >20% discrepancy flags as conflict
        tender_year: Optional[int] = None,              # For expiry checks; defaults to current year
    ):
        self.annexure_supersedes = annexure_supersedes_proposal
        self.min_conflict_fraction = min_magnitude_conflict_fraction
        import datetime
        self.tender_year: int = tender_year or datetime.date.today().year

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reconcile(
        self,
        requirement_id: str,
        requirement_text: str,
        requirement_value: Optional[float],
        requirement_unit: Optional[str],
        all_candidates: List[Any],  # List[RetrievedEvidenceCandidate]
    ) -> ReconciliationResult:
        """Reconciles all retrieved evidence candidates across multiple source documents.

        Args:
            requirement_id:    Unique ID of the tender requirement being evaluated.
            requirement_text:  Full text of the tender requirement clause.
            requirement_value: Numeric threshold from the requirement (if any).
            requirement_unit:  Unit of the numeric threshold.
            all_candidates:    List of RetrievedEvidenceCandidate objects from all
                               retrieved evidence sentences (from TF-IDF or Semantic).

        Returns:
            ReconciliationResult with conflict details, recommended label override, and reasoning.
        """
        # Group candidates by source document
        groups = self._group_by_source(all_candidates)

        if len(groups) < 2:
            # Only one source document — no cross-document conflict possible
            return ReconciliationResult(
                requirement_id=requirement_id,
                has_conflict=False,
                conflict_severity="NONE",
                document_groups=groups,
                reconciled_label=None,
                reconciled_reasoning="Only one source document present; no cross-document reconciliation applicable.",
            )

        # Separate annexure groups from proposal groups
        annexure_groups = [g for g in groups if g.tier_rank <= 2]
        proposal_groups = [g for g in groups if g.tier_rank >= 3]

        if not annexure_groups:
            return ReconciliationResult(
                requirement_id=requirement_id,
                has_conflict=False,
                conflict_severity="NONE",
                document_groups=groups,
                reconciled_label=None,
                reconciled_reasoning="No annexure or certified secondary source present; cross-document reconciliation not applicable.",
            )

        # Run conflict detectors
        conflicts: List[ConflictingClaimPair] = []
        req_text_lower = requirement_text.lower()

        # Determine which conflict type(s) to attempt based on requirement context
        is_financial = (
            requirement_value is not None
            and requirement_unit is not None
            and "usd" in requirement_unit.lower()
        )
        is_staffing = "project manager" in req_text_lower or "fte" in req_text_lower or "staffing" in req_text_lower
        is_project_count = (
            "completion certificate" in req_text_lower
            or "reference project" in req_text_lower
            or "similar project" in req_text_lower
        )
        is_debarment = "debarred" in req_text_lower or "blacklisted" in req_text_lower or "suspended" in req_text_lower
        is_certificate_expiry = "valid" in req_text_lower or "authorization" in req_text_lower or "accredited" in req_text_lower
        is_sla_uptime = ("uptime" in req_text_lower or "sla" in req_text_lower or "availability" in req_text_lower) and requirement_value is not None

        for annexure_grp in annexure_groups:
            for proposal_grp in proposal_groups:
                if is_financial:
                    c = self._check_financial_conflict(proposal_grp, annexure_grp, requirement_value, requirement_unit)
                    if c:
                        conflicts.append(c)

                if is_sla_uptime:
                    c = self._check_sla_uptime_conflict(proposal_grp, annexure_grp, requirement_value)
                    if c:
                        conflicts.append(c)

                if is_staffing:
                    c = self._check_staffing_conflict(proposal_grp, annexure_grp)
                    if c:
                        conflicts.append(c)

                if is_project_count:
                    c = self._check_project_count_conflict(proposal_grp, annexure_grp, requirement_value)
                    if c:
                        conflicts.append(c)

                if is_debarment:
                    c = self._check_debarment_conflict(proposal_grp, annexure_grp)
                    if c:
                        conflicts.append(c)

                if is_certificate_expiry:
                    c = self._check_certificate_expiry_conflict(proposal_grp, annexure_grp)
                    if c:
                        conflicts.append(c)

        if not conflicts:
            return ReconciliationResult(
                requirement_id=requirement_id,
                has_conflict=False,
                conflict_severity="NONE",
                document_groups=groups,
                reconciled_label=None,
                reconciled_reasoning="No conflicting claims detected across available source documents.",
                annexure_sources_checked=[g.source_document for g in annexure_groups],
            )

        # Determine severity and recommended label from collected conflicts
        hard_conflicts = [c for c in conflicts if c.recommended_label == "NON_COMPLIANT"]
        soft_conflicts = [c for c in conflicts if c.recommended_label == "UNCERTAIN"]

        if hard_conflicts:
            severity = "HARD"
            # Use first (highest confidence) hard conflict
            hard_conflicts.sort(key=lambda x: x.confidence, reverse=True)
            chosen = hard_conflicts[0]
            rec_label = "NON_COMPLIANT"
            rec_conf = chosen.confidence
            rec_reason = (
                f"CROSS-DOCUMENT CONFLICT DETECTED (Phase 7): "
                f"{chosen.reasoning} "
                f"[Authoritative source: '{chosen.secondary_source}' p.{chosen.secondary_page}]"
            )
        else:
            severity = "SOFT"
            soft_conflicts.sort(key=lambda x: x.confidence, reverse=True)
            chosen = soft_conflicts[0]
            rec_label = "UNCERTAIN"
            rec_conf = chosen.confidence
            rec_reason = (
                f"CROSS-DOCUMENT DISCREPANCY FLAGGED (Phase 7): "
                f"{chosen.reasoning} "
                f"[Secondary source: '{chosen.secondary_source}' p.{chosen.secondary_page}] "
                f"Requires procurement officer verification."
            )

        return ReconciliationResult(
            requirement_id=requirement_id,
            has_conflict=True,
            conflict_severity=severity,
            conflicts=conflicts,
            document_groups=groups,
            reconciled_label=rec_label,
            reconciled_confidence=rec_conf,
            reconciled_reasoning=rec_reason,
            annexure_sources_checked=[g.source_document for g in annexure_groups],
        )

    # ------------------------------------------------------------------
    # Evidence Grouping
    # ------------------------------------------------------------------

    def _group_by_source(self, candidates: List[Any]) -> List[DocumentEvidenceGroup]:
        """Groups evidence candidates by their source document."""
        groups_map: Dict[str, DocumentEvidenceGroup] = {}

        for cand in candidates:
            src = getattr(cand, "source_document", "UNKNOWN") or "UNKNOWN"
            if src not in groups_map:
                tier_rank, tier_label = classify_document_tier(src)
                groups_map[src] = DocumentEvidenceGroup(
                    source_document=src,
                    tier_rank=tier_rank,
                    tier_label=tier_label,
                )
            grp = groups_map[src]
            sentence = getattr(cand, "sentence", "") or ""
            if sentence:
                grp.sentences.append(sentence)
            page = getattr(cand, "page_number", None)
            if page is not None:
                grp.page_numbers.append(page)
            sec = getattr(cand, "section_reference", None)
            grp.section_references.append(sec)
            evid_id = getattr(cand, "evidence_id", "")
            grp.evidence_ids.append(evid_id)
            # Merge numeric values from the candidate
            nums = getattr(cand, "numeric_values", []) or []
            grp.numeric_values.extend(nums)

        return list(groups_map.values())

    # ------------------------------------------------------------------
    # Conflict Detectors
    # ------------------------------------------------------------------

    def _check_financial_conflict(
        self,
        proposal_grp: DocumentEvidenceGroup,
        annexure_grp: DocumentEvidenceGroup,
        requirement_value: Optional[float],
        requirement_unit: Optional[str],
    ) -> Optional[ConflictingClaimPair]:
        """Detects numeric magnitude conflict in financial turnover / net worth claims."""
        proposal_text = " ".join(proposal_grp.sentences)
        annexure_text = " ".join(annexure_grp.sentences)

        proposal_val = _parse_usd_millions(proposal_text)
        annexure_val = _parse_usd_millions(annexure_text)

        if proposal_val is None or annexure_val is None:
            return None
        if abs(proposal_val - annexure_val) < 0.01:
            return None  # Values are consistent

        # Calculate fractional discrepancy
        larger = max(proposal_val, annexure_val)
        if larger == 0:
            return None
        fraction = abs(proposal_val - annexure_val) / larger

        if fraction < self.min_conflict_fraction:
            return None  # Discrepancy is minor (rounding / currency translation)

        proposal_page = proposal_grp.page_numbers[0] if proposal_grp.page_numbers else None
        annexure_page = annexure_grp.page_numbers[0] if annexure_grp.page_numbers else None

        # Determine if annexure value fails the requirement threshold
        if requirement_value is not None and annexure_val < requirement_value:
            rec_label = "NON_COMPLIANT"
            conf = 0.90
            rsn = (
                f"Financial turnover conflict: Main proposal claims USD {proposal_val:.1f}M "
                f"but audited annexure '{annexure_grp.source_document}' reports only USD {annexure_val:.1f}M "
                f"(discrepancy: {fraction*100:.0f}%). "
                f"Annexure value USD {annexure_val:.1f}M is below the required threshold of "
                f"USD {requirement_value:.1f}M. Annexure supersedes self-declared proposal figure."
            )
        else:
            # Annexure value also passes the threshold, but there is still a material discrepancy
            rec_label = "UNCERTAIN"
            conf = 0.75
            rsn = (
                f"Financial turnover discrepancy: Main proposal claims USD {proposal_val:.1f}M; "
                f"annexure '{annexure_grp.source_document}' registers USD {annexure_val:.1f}M "
                f"({fraction*100:.0f}% difference). Both figures nominally clear the threshold, "
                f"but the conflicting declarations require auditor verification of the audited balance sheet."
            )

        return ConflictingClaimPair(
            conflict_type="numeric_magnitude",
            criterion="annual_turnover_usd_millions",
            primary_source=proposal_grp.source_document,
            primary_value=proposal_val,
            primary_unit="USD Million",
            primary_text=proposal_text[:300],
            primary_page=proposal_page,
            secondary_source=annexure_grp.source_document,
            secondary_value=annexure_val,
            secondary_unit="USD Million",
            secondary_text=annexure_text[:300],
            secondary_page=annexure_page,
            resolution_policy="annexure_supersedes_proposal",
            recommended_label=rec_label,
            confidence=conf,
            reasoning=rsn,
        )


    def _check_sla_uptime_conflict(
        self,
        proposal_grp: DocumentEvidenceGroup,
        annexure_grp: DocumentEvidenceGroup,
        requirement_value: Optional[float],
    ) -> Optional[ConflictingClaimPair]:
        """Detects conflict between proposal SLA claims and binding annexure SLA terms."""
        proposal_text = " ".join(proposal_grp.sentences)
        annexure_text = " ".join(annexure_grp.sentences)

        p_match = re.search(r"(\d+(?:\.\d+)?)\s*%", proposal_text)
        a_match = re.search(r"(\d+(?:\.\d+)?)\s*%", annexure_text)

        if not p_match or not a_match:
            return None

        p_val = float(p_match.group(1))
        a_val = float(a_match.group(1))

        if abs(p_val - a_val) < 0.05:
            return None

        req_val = float(requirement_value) if requirement_value is not None else 99.0
        if a_val < req_val:
            rec_label = "NON_COMPLIANT"
            conf = 0.89
            rsn = (
                f"SLA availability conflict: Executive proposal commits to {p_val:.2f}% uptime; "
                f"however, binding SLA terms in annexure '{annexure_grp.source_document}' restrict "
                f"guaranteed availability to {a_val:.2f}%, failing the mandatory requirement of {req_val:.2f}%."
            )
        else:
            rec_label = "UNCERTAIN"
            conf = 0.75
            rsn = (
                f"SLA availability discrepancy: Proposal claims {p_val:.2f}% uptime, while annexure "
                f"'{annexure_grp.source_document}' specifies {a_val:.2f}% uptime."
            )

        proposal_page = proposal_grp.page_numbers[0] if proposal_grp.page_numbers else None
        annexure_page = annexure_grp.page_numbers[0] if annexure_grp.page_numbers else None

        return ConflictingClaimPair(
            conflict_type="sla_uptime_discrepancy",
            criterion="monthly_uptime_sla",
            primary_source=proposal_grp.source_document,
            primary_value=p_val,
            primary_unit="%",
            primary_text=proposal_text[:300],
            primary_page=proposal_page,
            secondary_source=annexure_grp.source_document,
            secondary_value=a_val,
            secondary_unit="%",
            secondary_text=annexure_text[:300],
            secondary_page=annexure_page,
            resolution_policy="annexure_supersedes_proposal",
            recommended_label=rec_label,
            confidence=conf,
            reasoning=rsn,
        )

    def _check_staffing_conflict(
        self,
        proposal_grp: DocumentEvidenceGroup,
        annexure_grp: DocumentEvidenceGroup,
    ) -> Optional[ConflictingClaimPair]:
        """Detects FTE allocation conflict for named personnel (e.g. Project Manager)."""
        annexure_text = " ".join(annexure_grp.sentences)
        proposal_text = " ".join(proposal_grp.sentences)

        # Look for FTE allocation in annexure
        fte = _parse_fte_percent(annexure_text)
        if fte is None:
            # Also try raw percent mention in context of annexure
            m = _PERCENT_RE.search(annexure_text)
            if m:
                fte = float(m.group(1))

        if fte is None:
            return None

        # If the annexure reports very low FTE (part-time allocation), it's a conflict
        if fte >= self.MIN_FTE_FOR_DEDICATED:
            return None  # Person appears sufficiently allocated

        proposal_page = proposal_grp.page_numbers[0] if proposal_grp.page_numbers else None
        annexure_page = annexure_grp.page_numbers[0] if annexure_grp.page_numbers else None

        rsn = (
            f"Staffing allocation conflict: Proposal presents dedicated Project Manager, "
            f"but staffing schedule annexure '{annexure_grp.source_document}' reports only "
            f"{fte:.0f}% FTE allocation to this tender — indicating part-time commitment. "
            f"Effective availability does not meet 'dedicated' or 'proposed' Project Manager criteria."
        )

        return ConflictingClaimPair(
            conflict_type="staffing_allocation",
            criterion="project_manager_dedication",
            primary_source=proposal_grp.source_document,
            primary_value=100.0,
            primary_unit="% FTE (claimed dedicated)",
            primary_text=proposal_text[:300],
            primary_page=proposal_page,
            secondary_source=annexure_grp.source_document,
            secondary_value=fte,
            secondary_unit="% FTE",
            secondary_text=annexure_text[:300],
            secondary_page=annexure_page,
            resolution_policy="annexure_supersedes_proposal",
            recommended_label="NON_COMPLIANT",
            confidence=0.87,
            reasoning=rsn,
        )

    def _check_project_count_conflict(
        self,
        proposal_grp: DocumentEvidenceGroup,
        annexure_grp: DocumentEvidenceGroup,
        requirement_value: Optional[float],
    ) -> Optional[ConflictingClaimPair]:
        """Detects conflict between claimed project count and certificated project count."""
        proposal_text = " ".join(proposal_grp.sentences)
        annexure_text = " ".join(annexure_grp.sentences)

        proposal_count = _parse_count(proposal_text)
        annexure_count = _parse_count(annexure_text)

        if proposal_count is None or annexure_count is None:
            return None
        if proposal_count == annexure_count:
            return None

        proposal_page = proposal_grp.page_numbers[0] if proposal_grp.page_numbers else None
        annexure_page = annexure_grp.page_numbers[0] if annexure_grp.page_numbers else None

        # Annexure count is the certified figure
        req_count = int(requirement_value) if requirement_value is not None else 0

        if annexure_count < req_count:
            rec_label = "NON_COMPLIANT"
            conf = 0.88
            rsn = (
                f"Reference project count conflict: Proposal summary claims {proposal_count} qualifying projects; "
                f"however, client completion certificates in '{annexure_grp.source_document}' "
                f"only include valid signed documentation for {annexure_count} project(s). "
                f"Certified count {annexure_count} falls below the required {req_count}. "
                f"Uncertified / in-progress references cannot be counted."
            )
        else:
            # Annexure count >= requirement but disagrees with proposal
            rec_label = "UNCERTAIN"
            conf = 0.72
            rsn = (
                f"Reference project count discrepancy: Proposal claims {proposal_count} projects; "
                f"annexure '{annexure_grp.source_document}' shows {annexure_count} certified completions. "
                f"Annexure count {annexure_count} nominally meets the threshold of {req_count}, "
                f"but the inconsistency between narrative claim and certificate count warrants auditor verification."
            )

        return ConflictingClaimPair(
            conflict_type="project_count",
            criterion="similar_reference_project_count",
            primary_source=proposal_grp.source_document,
            primary_value=float(proposal_count),
            primary_unit="projects (claimed)",
            primary_text=proposal_text[:300],
            primary_page=proposal_page,
            secondary_source=annexure_grp.source_document,
            secondary_value=float(annexure_count),
            secondary_unit="projects (certified)",
            secondary_text=annexure_text[:300],
            secondary_page=annexure_page,
            resolution_policy="annexure_supersedes_proposal",
            recommended_label=rec_label,
            confidence=conf,
            reasoning=rsn,
        )

    def _check_debarment_conflict(
        self,
        proposal_grp: DocumentEvidenceGroup,
        annexure_grp: DocumentEvidenceGroup,
    ) -> Optional[ConflictingClaimPair]:
        """Detects debarment / blacklisting contradiction between clean declaration and debarment list."""
        annexure_text = " ".join(annexure_grp.sentences)
        proposal_text = " ".join(proposal_grp.sentences)

        annexure_lower = annexure_text.lower()
        proposal_lower = proposal_text.lower()

        # Annexure must mention debarment/blacklisting of some entity
        annexure_has_debarment = bool(
            re.search(r"\b(debarr?ed|debarment|blacklisted|blacklisting|suspended|suspension|red.flag)\b", annexure_lower)
        )
        # Proposal must claim clean status
        proposal_claims_clean = bool(
            re.search(r"\b(never\s+been\s+debarred|never\s+been\s+blacklisted|clean\s+record|no\s+debarment)\b", proposal_lower)
        )

        if not annexure_has_debarment:
            return None  # Annexure doesn't mention debarment

        proposal_page = proposal_grp.page_numbers[0] if proposal_grp.page_numbers else None
        annexure_page = annexure_grp.page_numbers[0] if annexure_grp.page_numbers else None

        if proposal_claims_clean:
            rsn = (
                f"Debarment declaration conflict: Proposal declares clean debarment record, "
                f"but annexure '{annexure_grp.source_document}' contains a debarment-related disclosure. "
                f"Conflicting debarment records require procurement officer investigation. "
                f"If a related entity (subsidiary, affiliate) appears on the debarment list, "
                f"qualification may be compromised under joint-venture provisions."
            )
            rec_label = "NON_COMPLIANT"
            conf = 0.82
        else:
            rsn = (
                f"Debarment disclosure in annexure '{annexure_grp.source_document}': "
                f"Debarment or blacklisting mentioned in annexure. "
                f"Proposal text does not explicitly deny debarment. Requires auditor review."
            )
            rec_label = "UNCERTAIN"
            conf = 0.70

        return ConflictingClaimPair(
            conflict_type="debarment_disclosure",
            criterion="debarment_clean_record",
            primary_source=proposal_grp.source_document,
            primary_value=None,
            primary_unit=None,
            primary_text=proposal_text[:300],
            primary_page=proposal_page,
            secondary_source=annexure_grp.source_document,
            secondary_value=None,
            secondary_unit=None,
            secondary_text=annexure_text[:300],
            secondary_page=annexure_page,
            resolution_policy="annexure_supersedes_proposal",
            recommended_label=rec_label,
            confidence=conf,
            reasoning=rsn,
        )

    def _check_certificate_expiry_conflict(
        self,
        proposal_grp: DocumentEvidenceGroup,
        annexure_grp: DocumentEvidenceGroup,
    ) -> Optional[ConflictingClaimPair]:
        """Detects expired certificate referenced in proposal but shown as expired in annexure."""
        annexure_text = " ".join(annexure_grp.sentences)
        proposal_text = " ".join(proposal_grp.sentences)

        annexure_lower = annexure_text.lower()

        # Check if annexure mentions expiry/expiration explicitly
        expiry_match = re.search(
            r"\b(expired?|expir(?:ed|y|ation)|not\s+re.validat\w+|lapsed)\b",
            annexure_lower
        )
        if not expiry_match:
            return None

        # Extract a year from the annexure if possible (e.g. "expired on December 31, 2023")
        expiry_year = _parse_year(annexure_text)
        expired_before_tender = (expiry_year is not None and expiry_year < self.tender_year)

        proposal_page = proposal_grp.page_numbers[0] if proposal_grp.page_numbers else None
        annexure_page = annexure_grp.page_numbers[0] if annexure_grp.page_numbers else None

        if expired_before_tender:
            rec_label = "NON_COMPLIANT"
            conf = 0.85
            rsn = (
                f"Certificate expiry conflict: Proposal claims an active authorization or accreditation, "
                f"but annexure '{annexure_grp.source_document}' indicates the certificate expired in {expiry_year} "
                f"(prior to the tender year {self.tender_year}). An expired certificate cannot satisfy an 'active' requirement."
            )
        else:
            rec_label = "UNCERTAIN"
            conf = 0.72
            rsn = (
                f"Certificate validity concern: Annexure '{annexure_grp.source_document}' references expiry or lapse "
                f"of a certification. Cannot confirm the certificate was valid at tender submission date. "
                f"Requires auditor verification of current validity status."
            )

        return ConflictingClaimPair(
            conflict_type="certificate_expiry",
            criterion="active_valid_certificate",
            primary_source=proposal_grp.source_document,
            primary_value=None,
            primary_unit=None,
            primary_text=proposal_text[:300],
            primary_page=proposal_page,
            secondary_source=annexure_grp.source_document,
            secondary_value=float(expiry_year) if expiry_year else None,
            secondary_unit="expiry_year",
            secondary_text=annexure_text[:300],
            secondary_page=annexure_page,
            resolution_policy="annexure_supersedes_proposal",
            recommended_label=rec_label,
            confidence=conf,
            reasoning=rsn,
        )

    # ------------------------------------------------------------------
    # Reconcile from dataset record (for evaluation harness)
    # ------------------------------------------------------------------

    def reconcile_from_metadata(
        self,
        requirement_id: str,
        requirement_text: str,
        requirement_value: Optional[float],
        requirement_unit: Optional[str],
        primary_evidence_text: str,
        primary_source_document: str,
        primary_page: Optional[int],
        conflicting_document: str,
        conflicting_page: Optional[int],
        conflicting_text: str,
    ) -> ReconciliationResult:
        """Reconciles using explicit metadata from dataset records (for evaluation).

        This is used when the conflict metadata is provided directly (e.g. from
        dataset/splits/test.jsonl contradiction_metadata field) rather than inferred
        from multiple retrieved candidates. It creates synthetic DocumentEvidenceGroup
        objects from the provided fields and runs the same reconciliation logic.
        """
        # Build synthetic candidate objects
        primary_cand = _SyntheticCandidate(
            sentence=primary_evidence_text,
            source_document=primary_source_document,
            page_number=primary_page or 1,
            section_reference=None,
            evidence_id="SYNTH-PRIMARY-001",
            numeric_values=[],
        )
        conflicting_cand = _SyntheticCandidate(
            sentence=conflicting_text,
            source_document=conflicting_document,
            page_number=conflicting_page or 1,
            section_reference=None,
            evidence_id="SYNTH-CONFLICT-001",
            numeric_values=[],
        )
        return self.reconcile(
            requirement_id=requirement_id,
            requirement_text=requirement_text,
            requirement_value=requirement_value,
            requirement_unit=requirement_unit,
            all_candidates=[primary_cand, conflicting_cand],
        )


# ---------------------------------------------------------------------------
# Helper: Synthetic candidate for evaluation harness
# ---------------------------------------------------------------------------

class _SyntheticCandidate:
    """Minimal duck-type of RetrievedEvidenceCandidate for metadata-driven evaluation."""

    def __init__(
        self,
        sentence: str,
        source_document: str,
        page_number: int,
        section_reference: Optional[str],
        evidence_id: str,
        numeric_values: Optional[List[Dict[str, Any]]] = None,
    ):
        self.sentence = sentence
        self.source_document = source_document
        self.page_number = page_number
        self.section_reference = section_reference
        self.evidence_id = evidence_id
        self.numeric_values = numeric_values or []
        self.similarity_score = 0.5
        self.has_negation = False
        self.extracted_entities = []
        self.uncertainty_flag = False
        self.review_notes = ""
        self.retrieval_method = "synthetic"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentence": self.sentence,
            "source_document": self.source_document,
            "page_number": self.page_number,
            "section_reference": self.section_reference,
            "evidence_id": self.evidence_id,
        }
