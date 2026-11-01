"""
certificate_verifier.py - Accurate Certificate Validity & Posture Verification
Part of Phase 8: Core NLP Reliability Improvements (Step 2B & 2B.1)

Reliability improvements:
1. Bare certification statements without verified expiry details evaluate to UNCERTAIN.
2. Historical expiry dates are compared against tender submission deadline:
   - If expiry_date >= submission_deadline -> Passes date check (COMPLIANT if certified)
   - If expiry_date < submission_deadline -> NON_COMPLIANT
   - If submission_deadline is unavailable -> UNCERTAIN
   - If stated expired without date or deadline -> UNCERTAIN
3. Multiple certificates in a document/sentence are segmented and associated with their
   own individual expiry dates, supporting separate clauses, paragraphs, and reversed order.
4. Ambiguous certificate-date pairings route to UNCERTAIN.
5. Compatibility and alignment claims strictly route to UNCERTAIN (never COMPLIANT).
6. Full evidence quotation and provenance preservation.
"""

import re
import calendar
from dataclasses import dataclass, asdict
from datetime import date, datetime
from typing import Optional, Tuple, Dict, Any, Union, List


MONTH_MAP: Dict[str, int] = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


@dataclass
class CertificateClaim:
    """Structured representation of an extracted certificate claim from bidder evidence."""
    standard_name: str
    claim_type: str                   # 'CERTIFIED', 'COMPATIBILITY', 'AMBIGUOUS', 'EXPIRED', 'DENIED', 'UNKNOWN'
    issue_date: Optional[date] = None
    expiry_date: Optional[date] = None
    raw_text: str = ""
    matched_phrase: str = ""
    is_ambiguous_association: bool = False
    source_document: Optional[str] = None
    page_number: Optional[int] = None
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.issue_date:
            d["issue_date"] = self.issue_date.isoformat()
        if self.expiry_date:
            d["expiry_date"] = self.expiry_date.isoformat()
        return d


class CertificateVerifier:
    """Verifier for certification claims, standards adherence, and certificate validity dates."""

    # Standard recognition regex
    STANDARD_REGEX = (
        r"\b("
        r"ISO(?:\s*[/IEC]*\s*\d+(?::\d+)?)?"
        r"|SOC\s*[123]"
        r"|CMMI(?:\s*Level\s*\d+)?"
        r"|PMP"
        r"|PRINCE2"
        r"|HIPAA"
        r"|PCI-DSS"
        r"|Udyam"
        r"|GSTIN"
        r")\b"
    )

    # Explicit denial / negation patterns
    DENIAL_PATTERNS = [
        r"\b(?:does|do|did)\s+not\s+(?:currently\s+)?(?:hold|possess)\b",
        r"\b(?:is|are|was|were|am)\s+not\s+(?:currently\s+)?(?:certified|accredited)\b",
        r"\b(?:have|has|had)\s+not\s+(?:currently\s+|yet\s+)?(?:obtained|received|held|achieved)\b",
        r"\b(?:not\s+(?:currently\s+)?certified|not\s+(?:currently\s+)?accredited)\b",
        r"\byet\s+to\s+obtain\b",
        r"\bpending\s+certification\b",
        r"\bunaccredited\b",
        r"\bno\s+(?:active\s+)?(?:iso|soc|pmp|cmmi|accreditation|certificate)\b",
    ]

    # Explicit expiry patterns
    EXPIRY_PATTERNS = [
        r"\b(?:expired\s+on|has\s+expired|is\s+expired|certificate\s+expired|certification\s+expired|past\s+validity|lapsed)\b",
    ]

    # Compatibility / alignment / adherence patterns (NEVER proof of certification)
    COMPATIBILITY_PATTERNS = [
        r"\bcompatible\s+with\b",
        r"\bcompatibility\s+with\b",
        r"\baligned\s+with\b",
        r"\balignment\s+with\b",
        r"\bfollows?\s+.*guidelines?\b",
        r"\bdesigned\s+according\s+to\b",
        r"\bin\s+accordance\s+with\b",
        r"\bbased\s+on\b",
        r"\bconforming\s+to\b",
        r"\bbuilt\s+to\s+comply\s+with\b",
        r"\badheres?\s+to\s+.*principles?\b",
        r"\badhering\s+to\s+.*principles?\b",
    ]

    # Ambiguous claims (require auditor review)
    AMBIGUOUS_PATTERNS = [
        r"\b(?:comply|complies|compliance)\s+with\b",
        r"\bfollowing\b",
        r"\bpractices\b",
        r"\bguidelines?\b",
        r"-level\b",
        r"\bprinciples?\b",
        r"\bframework\b",
    ]

    # Actual certification claims
    CERTIFIED_PATTERNS = [
        r"\b(?:hold\s+(?:an?\s+)?(?:active\s+)?(?:valid\s+)?|holds\b|holding\b|held\b|possess\b|possesses\b|possessing\b|obtained\b)\b",
        r"\bcertified\b",
        r"\baccredited\b",
        r"\baccreditation\s+certificate\b",
        r"\bcertificate\b",
        r"\bofficial\s+.*certificate\b",
        r"\benclosed\s+is\s+our\s+official\b",
    ]

    # Context cues for expiry dates
    EXPIRY_DATE_CUES = [
        r"valid\s+(?:through|until|up\s+to|till|upto|to)\b",
        r"expir(?:es|ed|ing|y)(?:\s+on|\s+date\s*[:\-]?)?\b",
        r"expiration(?:\s+date\s*[:\-]?)?\b",
        r"renewal\s+date\s*[:\-]?\b",
        r"validity(?:\s*[:\-]|\s+through|\s+until|\s+to|\s+period)?\b",
    ]

    # Context cues for issue dates
    ISSUE_DATE_CUES = [
        r"issued(?:\s+(?:on|in))?\b",
        r"issue\s+date\s*[:\-]?\b",
        r"effective\s+(?:from|date\s*[:\-]?)?\b",
        r"awarded\s+on\b",
        r"certified\s+(?:on|since)\b",
        r"valid\s+from\b",
    ]

    @staticmethod
    def _parse_month(m_str: str) -> Optional[int]:
        return MONTH_MAP.get(m_str.lower().strip())

    @staticmethod
    def _last_day_of_month(year: int, month: int) -> int:
        return calendar.monthrange(year, month)[1]

    @classmethod
    def parse_date_value(cls, d_val: Union[str, date, datetime]) -> Optional[date]:
        """Converts date, datetime, or date string into a datetime.date object."""
        if d_val is None:
            return None
        if isinstance(d_val, datetime):
            return d_val.date()
        if isinstance(d_val, date):
            return d_val
        if not isinstance(d_val, str):
            return None

        clean_str = d_val.strip()
        # YYYY-MM-DD
        m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", clean_str)
        if m:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))

        # DD/MM/YYYY or DD-MM-YYYY
        m = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$", clean_str)
        if m:
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if mo > 12 and d <= 12:
                d, mo = mo, d
            return date(y, mo, d)

        # DD Month YYYY
        m = re.match(r"^(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\s+(\d{4})$", clean_str)
        if m:
            mo = cls._parse_month(m.group(2))
            if mo:
                return date(int(m.group(3)), mo, int(m.group(1)))

        # Month DD, YYYY
        m = re.match(r"^([A-Za-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?(?:,)?\s+(\d{4})$", clean_str)
        if m:
            mo = cls._parse_month(m.group(1))
            if mo:
                return date(int(m.group(3)), mo, int(m.group(2)))

        return None

    def extract_dates(self, text: str) -> Tuple[Optional[date], Optional[date]]:
        """Extracts (issue_date, expiry_date) from certificate sentence.

        Uses preceding contextual cues to differentiate issue and expiry dates.
        """
        issue_date: Optional[date] = None
        expiry_date: Optional[date] = None

        date_patterns = [
            # 1. YYYY-MM-DD
            (r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b",
             lambda m: date(int(m.group(1)), int(m.group(2)), int(m.group(3)))),
            # 2. DD-Month-YYYY or DD Month YYYY
            (r"\b(\d{1,2})(?:st|nd|rd|th)?[\s\-]+([A-Za-z]{3,9})[\s\-]+(\d{4})\b",
             lambda m: date(int(m.group(3)), self._parse_month(m.group(2)), int(m.group(1)))
             if self._parse_month(m.group(2)) else None),
            # 3. Month DD, YYYY or Month DD YYYY
            (r"\b([A-Za-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?(?:,)?\s+(\d{4})\b",
             lambda m: date(int(m.group(3)), self._parse_month(m.group(1)), int(m.group(2)))
             if self._parse_month(m.group(1)) else None),
            # 4. DD/MM/YYYY
            (r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b",
             lambda m: (
                 date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
                 if int(m.group(2)) <= 12 and int(m.group(1)) <= 31
                 else (date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
                       if int(m.group(1)) <= 12 and int(m.group(2)) <= 31 else None)
             )),
            # 5. Month YYYY (e.g. December 2026) -> end of month for expiry
            (r"\b([A-Za-z]{3,9})\s+(\d{4})\b",
             lambda m: date(int(m.group(2)), self._parse_month(m.group(1)),
                            self._last_day_of_month(int(m.group(2)), self._parse_month(m.group(1))))
             if self._parse_month(m.group(1)) else None),
            # 6. Standalone Year with validity cue (e.g. valid through 2026)
            (r"(?:valid\s+(?:through|until|up\s+to|till)\s+)(\d{4})\b",
             lambda m: date(int(m.group(1)), 12, 31)),
        ]

        found_dates: List[Tuple[Tuple[int, int], date]] = []
        for pattern, extractor in date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    dt = extractor(match)
                    if dt:
                        span = match.span()
                        if not any(max(span[0], s[0]) < min(span[1], s[1]) for s, _ in found_dates):
                            found_dates.append((span, dt))
                except Exception:
                    continue

        found_dates.sort(key=lambda x: x[0][0])

        for span, dt in found_dates:
            start_ctx = max(0, span[0] - 50)
            preceding_text = text[start_ctx:span[0]].lower()

            is_expiry = any(re.search(cue, preceding_text, re.IGNORECASE) for cue in self.EXPIRY_DATE_CUES)
            is_issue = any(re.search(cue, preceding_text, re.IGNORECASE) for cue in self.ISSUE_DATE_CUES)

            if is_expiry:
                expiry_date = dt
            elif is_issue and not issue_date:
                issue_date = dt
            else:
                if not expiry_date and any(w in preceding_text for w in ["through", "until", "to", "expired"]):
                    expiry_date = dt
                elif not expiry_date and not issue_date:
                    expiry_date = dt

        return issue_date, expiry_date

    def classify_claim_posture(self, text: str) -> str:
        """Classifies the posture of a certification claim sentence."""
        text_lower = text.lower()

        # 1. Explicit denial takes precedence
        if any(re.search(p, text_lower) for p in self.DENIAL_PATTERNS):
            return "DENIED"

        # 2. Explicit expiry
        if any(re.search(p, text_lower) for p in self.EXPIRY_PATTERNS):
            return "EXPIRED"

        # 3. Compatibility / alignment / adherence (NEVER treated as certified)
        if any(re.search(p, text_lower) for p in self.COMPATIBILITY_PATTERNS):
            return "COMPATIBILITY"

        # 4. Actual certification vs ambiguous claims
        has_cert = any(re.search(p, text_lower) for p in self.CERTIFIED_PATTERNS)
        has_ambig = any(re.search(p, text_lower) for p in self.AMBIGUOUS_PATTERNS)

        if has_cert and not has_ambig:
            return "CERTIFIED"
        elif has_cert and has_ambig:
            if re.search(r"\b(?:certified|accredited|certificate|license)\b", text_lower):
                return "CERTIFIED"
            return "AMBIGUOUS"
        elif has_ambig:
            return "AMBIGUOUS"

        return "UNKNOWN"

    def extract_deadline_from_requirement(self, requirement_text: str) -> Optional[date]:
        """Extracts explicit submission deadline from requirement clause if present."""
        m = re.search(
            r"(?:valid\s+(?:on|through|until|up\s+to|as\s+of)|deadline(?:\s+of)?|submission\s+date\s*[:\-]?|by)\s+([A-Za-z0-9\s,\/\-]+)",
            requirement_text,
            re.IGNORECASE
        )
        if m:
            candidate_str = m.group(1).strip().rstrip(".")
            _, exp = self.extract_dates(candidate_str)
            if exp:
                return exp
            dt = self.parse_date_value(candidate_str)
            if dt:
                return dt
        return None

    def _split_sentence_by_standards(self, sentence: str, matches: List[Any]) -> List[Tuple[str, str]]:
        """Segments a multi-standard sentence into standard-specific clauses.

        Returns a list of (standard_name, clause_text) tuples.
        """
        if len(matches) <= 1:
            std = matches[0].group(1) if matches else "Accreditation"
            return [(std, sentence)]

        boundaries = [0]
        for i in range(len(matches) - 1):
            m1 = matches[i]
            m2 = matches[i + 1]
            intervening = sentence[m1.end():m2.start()]
            # Look for conjunctions or punctuation: ';', ',', 'and', 'as well as', 'while'
            conj = re.search(r"(?:;\s*|,\s*(?:and\s+)?|\s+and\s+|\s+as\s+well\s+as\s+|\s+while\s+)", intervening, re.IGNORECASE)
            if conj:
                split_point = m1.end() + conj.end()
            else:
                split_point = (m1.end() + m2.start()) // 2
            boundaries.append(split_point)
        boundaries.append(len(sentence))

        clauses: List[Tuple[str, str]] = []
        for i in range(len(matches)):
            std = matches[i].group(1)
            clause_text = sentence[boundaries[i]:boundaries[i + 1]].strip()
            clauses.append((std, clause_text))
        return clauses

    def extract_all_certificate_claims(
        self,
        text: str,
        source_document: Optional[str] = None,
        page_number: Optional[int] = None
    ) -> List[CertificateClaim]:
        """Extracts all certificate claims and associates each standard with its own validity dates.

        Supports:
        - Multiple certificates across separate sentences.
        - Multiple certificates within a single sentence or clause.
        - Preserves sentence boundaries and prevents date cross-contamination.
        - Flags ambiguous pairings (e.g. multiple standards sharing unsegmented dates).
        """
        claims: List[CertificateClaim] = []

        # Split text into sentences
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)

        for sent in sentences:
            sent_clean = sent.strip()
            if not sent_clean:
                continue

            matches = list(re.finditer(self.STANDARD_REGEX, sent_clean, re.IGNORECASE))
            if matches:
                # Filter out certificate registration IDs like "Cert No: ISO-98421-B"
                filtered_matches = []
                for m in matches:
                    prefix = sent_clean[max(0, m.start() - 25):m.start()]
                    if re.search(r"(?:cert(?:ificate)?|licen[sc]e|reg(?:istration)?)\s*(?:no\.?|num(?:ber)?|#)?\s*[:\-]?\s*$", prefix, re.IGNORECASE):
                        continue
                    filtered_matches.append(m)

                # If filtered matches has specific ISO standard (e.g. ISO 9001, ISO 27001), drop bare 'ISO'
                has_specific_iso = any(re.match(r"^ISO\s*[/IEC]*\s*\d+", m.group(1), re.IGNORECASE) for m in filtered_matches)
                if has_specific_iso:
                    filtered_matches = [
                        m for m in filtered_matches
                        if not re.match(r"^ISO$", m.group(1).strip(), re.IGNORECASE)
                    ]
                matches = filtered_matches

            if not matches:
                # Check if sentence has general certification claim without named standard
                posture = self.classify_claim_posture(sent_clean)
                if posture != "UNKNOWN":
                    iss, exp = self.extract_dates(sent_clean)
                    claims.append(CertificateClaim(
                        standard_name="Accredited Standard",
                        claim_type=posture,
                        issue_date=iss,
                        expiry_date=exp,
                        raw_text=sent_clean,
                        source_document=source_document,
                        page_number=page_number
                    ))
                continue

            # Check if sentence has ambiguous 'respectively' or coordinated standards sharing dates
            is_ambiguous = bool(re.search(r"\brespectively\b", sent_clean, re.IGNORECASE))

            if len(matches) > 1 and is_ambiguous:
                # Multiple standards with ambiguous distributed dates
                for m in matches:
                    std_name = m.group(1)
                    claims.append(CertificateClaim(
                        standard_name=std_name,
                        claim_type="AMBIGUOUS",
                        issue_date=None,
                        expiry_date=None,
                        raw_text=sent_clean,
                        is_ambiguous_association=True,
                        source_document=source_document,
                        page_number=page_number,
                        reasoning=f"Ambiguous certificate date pairing: '{sent_clean}'."
                    ))
                continue

            # Segment sentence by standard clauses
            std_clauses = self._split_sentence_by_standards(sent_clean, matches)

            for std_name, clause_text in std_clauses:
                iss_dt, exp_dt = self.extract_dates(clause_text)

                # Fallback: if clause didn't catch the date but the full sentence has only ONE date,
                # check if standards were coordinated (e.g. 'Our ISO 9001 and ISO 27001 are valid to 2025')
                if not exp_dt:
                    full_iss, full_exp = self.extract_dates(sent_clean)
                    if full_exp and not any("," in sent_clean for _ in [0]):
                        exp_dt = full_exp
                        iss_dt = full_iss

                posture = self.classify_claim_posture(clause_text)
                if posture == "UNKNOWN":
                    posture = self.classify_claim_posture(sent_clean)

                claims.append(CertificateClaim(
                    standard_name=std_name,
                    claim_type=posture,
                    issue_date=iss_dt,
                    expiry_date=exp_dt,
                    raw_text=clause_text,
                    source_document=source_document,
                    page_number=page_number
                ))

        return claims

    def verify_certificate(
        self,
        requirement_text: str,
        evidence_sentence: str,
        req_standard: Optional[str] = None,
        tender_submission_date: Optional[Union[str, date]] = None
    ) -> Tuple[str, float, str, CertificateClaim]:
        """Performs robust, auditable verification of certificate claim, validity, and submission deadline.

        Reliability safeguards (Step 2B.1):
        - Bare certification claims without expiry date or verifiable details return UNCERTAIN.
        - Historical expiry is strictly compared against tender_submission_date.
        - Expiry dates are isolated per standard, preventing cross-standard contamination.
        - Ambiguous associations return UNCERTAIN.
        """
        # 1. Extract all structured certificate claims from evidence
        claims = self.extract_all_certificate_claims(evidence_sentence)

        # 2. Determine target standard name
        std_name = req_standard or ""
        if not std_name:
            m = re.search(self.STANDARD_REGEX, requirement_text, re.IGNORECASE)
            std_name = m.group(1) if m else "Accredited Standard"

        # 3. Match the claim for target standard
        target_norm = std_name.lower().replace(":", "").replace("-", "").replace("/", "")
        matched_claim: Optional[CertificateClaim] = None

        for c in claims:
            c_norm = c.standard_name.lower().replace(":", "").replace("-", "").replace("/", "")
            # Check for substring token match (e.g. '27001' in 'iso27001')
            std_digits = re.findall(r"\d+", std_name)
            if std_digits and any(d in c_norm for d in std_digits):
                matched_claim = c
                break
            if c_norm in target_norm or target_norm in c_norm:
                matched_claim = c
                break

        if not matched_claim:
            if claims:
                matched_claim = claims[0]
            else:
                posture = self.classify_claim_posture(evidence_sentence)
                iss, exp = self.extract_dates(evidence_sentence)
                matched_claim = CertificateClaim(
                    standard_name=std_name,
                    claim_type=posture,
                    issue_date=iss,
                    expiry_date=exp,
                    raw_text=evidence_sentence
                )

        claim = matched_claim
        claim_type = claim.claim_type
        expiry_dt = claim.expiry_date

        # 4. Determine tender submission deadline
        parsed_deadline = self.parse_date_value(tender_submission_date)
        if parsed_deadline is None:
            parsed_deadline = self.extract_deadline_from_requirement(requirement_text)

        # -------------------------------------------------------------------
        # SAFEGUARD 1: Ambiguous Association
        # -------------------------------------------------------------------
        if claim.is_ambiguous_association:
            claim.reasoning = (
                f"Ambiguous certificate date association: multiple standards mentioned without "
                f"verifiable date pairing in '{evidence_sentence}'. Requires manual auditor review."
            )
            return "UNCERTAIN", 0.70, claim.reasoning, claim

        # -------------------------------------------------------------------
        # SAFEGUARD 2: Explicit Denial / Negation
        # -------------------------------------------------------------------
        if claim_type == "DENIED":
            claim.reasoning = (
                f"Proposal claim explicitly denies holding required accreditation '{std_name}': "
                f"'{claim.raw_text}'."
            )
            return "NON_COMPLIANT", 0.90, claim.reasoning, claim

        # -------------------------------------------------------------------
        # SAFEGUARD 3: Compatibility / Alignment / Guideline Adherence
        # -------------------------------------------------------------------
        if claim_type == "COMPATIBILITY":
            claim.reasoning = (
                f"Proposal asserts compatibility, alignment, or design adhering to '{std_name}' principles/guidelines "
                f"('{claim.raw_text}') rather than furnishing a verified accredited certificate. "
                f"Compatibility is not proof of certification. Requires procurement officer review."
            )
            return "UNCERTAIN", 0.75, claim.reasoning, claim

        # -------------------------------------------------------------------
        # SAFEGUARD 4: Ambiguous Standard Claims
        # -------------------------------------------------------------------
        if claim_type == "AMBIGUOUS":
            claim.reasoning = (
                f"Proposal asserts informal compliance or practices related to '{std_name}' ('{claim.raw_text}') "
                f"without furnishing an official accredited certificate. Flagged for auditor review."
            )
            return "UNCERTAIN", 0.70, claim.reasoning, claim

        # -------------------------------------------------------------------
        # SAFEGUARD 5: Date Comparison (Handles both CERTIFIED and EXPIRED claims)
        # -------------------------------------------------------------------
        # Case A & B: Both Expiry Date and Submission Deadline are available
        if expiry_dt and parsed_deadline:
            if expiry_dt >= parsed_deadline:
                # Case A: Certificate was valid at submission deadline (inclusive)
                claim.reasoning = (
                    f"Verified accreditation certificate for '{std_name}' was valid through "
                    f"{expiry_dt.strftime('%Y-%m-%d')} at the time of tender submission and "
                    f"satisfies tender submission deadline (deadline: {parsed_deadline.strftime('%Y-%m-%d')})."
                )
                return "COMPLIANT", 0.90, claim.reasoning, claim
            else:
                # Case B: Certificate expired before tender submission deadline
                claim.reasoning = (
                    f"Accreditation certificate for '{std_name}' expired on {expiry_dt.strftime('%Y-%m-%d')}, "
                    f"prior to the tender submission deadline of {parsed_deadline.strftime('%Y-%m-%d')}."
                )
                return "NON_COMPLIANT", 0.90, claim.reasoning, claim

        # Case C: Expiry date is present, but Submission Deadline is UNAVAILABLE
        if expiry_dt and not parsed_deadline:
            claim.reasoning = (
                f"Bidder provides accreditation certificate for '{std_name}' with expiry date "
                f"{expiry_dt.strftime('%Y-%m-%d')}, but tender submission deadline is unavailable in "
                f"tender metadata to verify active validity at tender submission. Flagged for auditor review."
            )
            return "UNCERTAIN", 0.75, claim.reasoning, claim

        # Case D & Task 1: Expiry date is MISSING
        if not expiry_dt:
            if claim_type == "EXPIRED":
                # Case D: Statement says expired, but has no date and no deadline
                claim.reasoning = (
                    f"Bidder states certificate is expired ('{claim.raw_text}'), but no expiration date "
                    f"or tender submission deadline is available to establish whether non-compliance "
                    f"applies to the tendered submission date. Flagged for procurement review."
                )
                return "UNCERTAIN", 0.70, claim.reasoning, claim

            # Task 1: Bare certification claim without expiry date
            # Check if independently verifiable certificate registration is present
            has_cert_id = bool(re.search(r"\b(?:cert(?:\s*no)?|license|registration)\s*[:#]\s*[A-Za-z0-9\-]+", evidence_sentence, re.IGNORECASE))
            if has_cert_id and parsed_deadline:
                claim.reasoning = (
                    f"Official accreditation certificate registration confirmed for '{std_name}' "
                    f"in bidder proposal: '{claim.raw_text}'. Expiry date missing; procurement review advised."
                )
                return "UNCERTAIN", 0.75, claim.reasoning, claim

            # Bare claim without certificate details or expiry date
            claim.reasoning = (
                f"Bidder asserts '{std_name}' certification ('{claim.raw_text}'), but no certificate "
                f"expiry date or independently verifiable certificate registration details were provided "
                f"in proposal evidence. Cannot verify active validity without physical certificate review."
            )
            return "UNCERTAIN", 0.75, claim.reasoning, claim

        # -------------------------------------------------------------------
        # Default Fallback for Unrecognized Posture
        # -------------------------------------------------------------------
        claim.reasoning = (
            f"Uncertain or unverified accreditation statement for '{std_name}': '{evidence_sentence}'. "
            f"Requires manual auditor verification."
        )
        return "UNCERTAIN", 0.60, claim.reasoning, claim
