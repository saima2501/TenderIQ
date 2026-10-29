"""
requirement_extractor.py - Structured Tender Requirement Extraction Pipeline
Part of Phase 3: Requirement & Evidence Extraction

Features:
1. Extracts structured requirement clauses from procurement tender documents (RFPs, IFBs).
2. Decomposes each requirement into:
   - requirement_id (clause/section or sequential ID)
   - original_text (unmodified clause sentence)
   - category (Technical, Financial, Experience, Statutory, SLA & Warranty, etc.)
   - parameter (e.g. uptime SLA, turnover, throughput, encryption, warranty)
   - operator (>=, <=, ==, CONTAINS, MANDATORY_SUBMISSION, NONE)
   - value (numeric threshold or required standard name)
   - unit (%, USD Million, Gbps, years, months, minutes, etc.)
   - is_mandatory (True for must/shall, False for should/preferred, None if ambiguous)
   - source_document & page_number (1-indexed)
   - uncertainty_flag & review_notes (marks uncertain or incomplete statements for review)
3. Transparent rule-based, regex-driven NLP with optional spaCy integration.
   - Does NOT claim pre-trained model training.
   - Auditable, explainable, and deterministic.
4. Seamless integration with Phase 2 PDFExtractor and ExtractedDocument.
"""

import os
import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union, Tuple

# Optional spaCy integration with graceful fallback
try:
    import spacy
    try:
        _nlp = spacy.blank("en")
        if "sentencizer" not in _nlp.pipe_names:
            _nlp.add_pipe("sentencizer")
        SPACY_AVAILABLE = True
    except Exception:
        _nlp = None
        SPACY_AVAILABLE = False
except ImportError:
    spacy = None
    _nlp = None
    SPACY_AVAILABLE = False


@dataclass
class ExtractedRequirement:
    """Represents a single structured tender requirement extracted from text."""
    requirement_id: str                      # e.g. "REQ-IT-001-01", "REQ-2.1", "REQ-001"
    original_text: str                       # Full clause sentence
    category: str                            # Functional procurement category
    parameter: str                           # Extracted parameter / criterion name
    operator: str                            # Comparison operator: '>=', '<=', '==', 'CONTAINS', 'MANDATORY_SUBMISSION', 'NONE'
    value: Optional[Union[float, str]] = None # Numeric threshold or specific standard name
    unit: Optional[str] = None               # Unit of measurement (%, USD Million, Gbps, years, etc.)
    is_mandatory: Optional[bool] = None      # True for mandatory, False for optional/preferred, None for ambiguous
    source_document: str = "N/A"             # Tender PDF filename
    page_number: Optional[int] = None        # 1-indexed page number
    section_reference: Optional[str] = None  # Heading or clause reference (e.g. "2.0 Financial Qualification")
    uncertainty_flag: bool = False           # True if ambiguous, incomplete, or missing explicit metrics
    review_notes: str = ""                   # Rationale for extraction and review recommendations

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RequirementExtractionReport:
    """Summary report of requirements extracted from a tender document."""
    source_document: str
    total_pages: int
    total_requirements: int
    mandatory_count: int
    optional_count: int
    uncertain_count: int
    nlp_engine: str
    requirements: List[ExtractedRequirement] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["requirements"] = [r.to_dict() for r in self.requirements]
        return d


class RequirementExtractor:
    """Rule-based and regex-driven procurement tender requirement extractor."""

    # 1. Modality Keywords (Modal auxiliary verbs & procurement mandates)
    MANDATORY_MODAL_PATTERNS = [
        r"\bmust\b",
        r"\bshall\b",
        r"\bis\s+required\s+to\b",
        r"\bare\s+required\s+to\b",
        r"\bmandatory\b",
        r"\bcompulsory\b",
        r"\bstrictly\s+required\b",
        r"\bessential\b",
        r"\bsolemn\s+declaration\b",
        r"\bcontractually\s+guarantee\b",
        r"\bnon-negotiable\b",
    ]

    # Mandatory threshold indicators establishing non-negotiable minimum criteria
    MANDATORY_THRESHOLD_PATTERNS = [
        r"\bminimum(?:\s+of)?\b",
        r"\bat\s+least\b",
        r"\bnot\s+less\s+than\b",
        r"\bno\s+less\s+than\b",
        r"\bturnover\s+threshold\b",
        r"\beligibility\s+threshold\b",
        r"\bqualification\s+threshold\b",
    ]

    OPTIONAL_PATTERNS = [
        r"\bshould\b",
        r"\bpreferred\b",
        r"\bpreferable\b",
        r"\bdesirable\b",
        r"\brecommended\b",
        r"\boptional\b",
        r"\bmay\b",
        r"\bencouraged\b",
        r"\bis\s+advised\b",
    ]

    # 2. Operator Patterns
    OPERATOR_GTE_PATTERNS = [
        r"\bminimum(?:\s+of)?\b",
        r"\bat\s+least\b",
        r"\bnot\s+less\s+than\b",
        r"\bno\s+less\s+than\b",
        r"\bor\s+more\b",
        r"\bexceeding\b",
        r"\bgreater\s+than\s+or\s+equal\s+to\b",
        r">=",
    ]

    OPERATOR_LTE_PATTERNS = [
        r"\b(?:must\s+|shall\s+|does\s+|do\s+|should\s+)?not\s+exceed\b",
        r"\bnot\s+to\s+exceed\b",
        r"\bcannot\s+exceed\b",
        r"\bmaximum(?:\s+of)?\b",
        r"\bat\s+most\b",
        r"\bnot\s+more\s+than\b",
        r"\bno\s+more\s+than\b",
        r"\bwithin\b",
        r"\binside\b",
        r"\bup\s+to\b",
        r"\bless\s+than\s+or\s+equal\s+to\b",
        r"<=",
    ]

    OPERATOR_EQ_PATTERNS = [
        r"\bexactly\b",
        r"\bequal\s+to\b",
        r"\bstrictly\s+equal\s+to\b",
        r"==",
    ]

    # 3. Known Standards & Certifications
    STANDARDS_MAP = {
        "ISO/IEC 27001": "ISO/IEC 27001:2022",
        "ISO 27001": "ISO/IEC 27001:2022",
        "ISO 9001": "ISO 9001:2015",
        "SOC 2 TYPE II": "SOC 2 Type II",
        "SOC 2": "SOC 2 Type II",
        "CMMI LEVEL 5": "CMMI Level 5",
        "CMMI LEVEL 3": "CMMI Level 3",
        "PMP": "PMP Certification",
        "PRINCE2": "PRINCE2 Certification",
        "AES-256": "AES-256 Encryption",
        "TLS 1.3": "TLS 1.3 Encryption",
    }

    # 4. Incomplete Clause Triggers (trailing prepositions/conjunctions)
    INCOMPLETE_TAIL_REGEX = re.compile(
        r"(?:\b(to|with|regarding|for|at|in|on|of|by|and|or|such\s+as)\s*[\.\?!]?\s*$)",
        re.IGNORECASE
    )

    def __init__(self, use_spacy: bool = True):
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        self.engine_name = "spacy_with_rules" if self.use_spacy else "regex_rules"

    def clean_text_for_nlp(self, text: str) -> str:
        """Removes PDF watermark line artifacts and normalizes whitespace."""
        if not text:
            return ""
        # 1. Strip simulation banners
        lines = []
        for line in text.split("\n"):
            line_str = line.strip()
            # Skip watermark simulation banner lines
            if "*** SIMULATED" in line_str or "SIMULATED PROCUREMENT DATASET" in line_str:
                continue
            if "SIMULATED DATASET" in line_str or "LEGAL NOTICE: This tender document is a simulated" in line_str:
                continue
            # Skip isolated single uppercase letters from diagonal ReportLab watermarks ('S', 'I', 'M', etc.)
            if len(line_str) == 1 and line_str.isupper():
                continue
            lines.append(line)

        cleaned = "\n".join(lines)
        # Clean merged watermark characters: e.g. "PositiveA net" -> "Positive net"
        cleaned = re.sub(r"(?<=[a-z])([A-Z])(?=\s+[a-z])", "", cleaned)
        # 2. Collapse line-wrapped sentences: join lines unless the next line starts a numbered section/clause (e.g. "2.1 Minimum")
        cleaned = re.sub(r"(?<![\.:\?!])\n(?!\s*\d{1,2}\.\d{1,2}\s+[A-Za-z]|\s*Section|\s*[A-Z\s]{4,}:)", " ", cleaned)
        # 3. Collapse multiple spaces
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        return cleaned.strip()

    def split_into_clauses(self, text: str) -> List[Tuple[str, Optional[str]]]:
        """Splits page text into candidate requirement sentences with associated section headings."""
        cleaned = self.clean_text_for_nlp(text)
        if not cleaned:
            return []

        results = []
        current_section = "General / Scope"

        # Break text by line or paragraph first
        raw_blocks = [b.strip() for b in cleaned.split("\n") if b.strip()]

        for block in raw_blocks:
            # Check if this block is a section header (e.g. "2.0 Financial Qualification Criteria")
            sec_match = re.match(r"^(\d+\.\d+\s+[A-Za-z0-9&/,\s\-]+)$", block)
            if sec_match and len(block) < 80:
                current_section = sec_match.group(1).strip()
                continue

            # Check if block has embedded numbered clauses (e.g. "2.1 Turnover: ... \n 2.2 Solvency: ...")
            # Split clauses by numbered prefix e.g. "2.1 ", "3.2 ", or sentence endings
            clause_parts = re.split(r"(?<=\.)\s+(?=\d+\.\d+\s+[A-Z])", block)
            for part in clause_parts:
                part = part.strip()
                if not part:
                    continue

                # Further split into sentences if spaCy is available, or use sentence regex
                sentences = self._split_sentences(part)
                for sent in sentences:
                    sent = sent.strip()
                    if sent:
                        results.append((sent, current_section))

        return results

    def _split_sentences(self, text: str) -> List[str]:
        """Sentence splitter using spaCy sentencizer if available, or protected regex."""
        if self.use_spacy and _nlp is not None:
            doc = _nlp(text)
            return [s.text for s in doc.sents if s.text.strip()]

        # Pure-Python sentence boundary regex protecting abbreviations like "USD", "e.g.", "Dr.", "No.", "Ref."
        splits = re.split(r"(?<!\bUSD)(?<!\be\.g)(?<!\bi\.e)(?<!\bDr)(?<!\bNo)(?<!\bRef)(?<!\bFY\d{4})[\.\?!]\s+(?=[A-Z0-9])", text)
        sentences = []
        for s in splits:
            s_clean = s.strip()
            if s_clean:
                if not s_clean.endswith((".", "!", "?")):
                    s_clean += "."
                sentences.append(s_clean)
        return sentences

    def extract_from_text(
        self,
        text: str,
        source_document: str = "TENDER_TEXT",
        page_number: Optional[int] = 1,
        id_prefix: str = "REQ"
    ) -> List[ExtractedRequirement]:
        """Extracts structured requirements from a single page or string."""
        clauses = self.split_into_clauses(text)
        requirements: List[ExtractedRequirement] = []
        req_counter = 1

        for sent, section in clauses:
            # Skip pure document headers, disclaimers, or metadata tables
            if self._is_non_requirement_line(sent):
                continue

            # Process the clause
            req = self._parse_requirement_clause(
                text=sent,
                section=section,
                source_document=source_document,
                page_number=page_number,
                default_id=f"{id_prefix}-{req_counter:03d}"
            )
            if req:
                requirements.append(req)
                req_counter += 1

                # BUG 2 FIX: If the clause is a Project Manager certification requirement,
                # also emit a SEPARATE sub-requirement for the PM's years of experience.
                # This prevents corporate operating experience from being misused as
                # proof of the individual PM's relevant experience.
                if req.parameter == "Project Manager PMP/PRINCE2 certification":
                    pm_exp_req = self._build_pm_experience_requirement(
                        text=sent,
                        section=section,
                        source_document=source_document,
                        page_number=page_number,
                        base_id=f"{id_prefix}-{req_counter:03d}",
                        parent_clause_id=req.requirement_id,
                    )
                    if pm_exp_req:
                        requirements.append(pm_exp_req)
                        req_counter += 1

        return requirements


    def extract_from_extracted_document(self, doc: Any) -> RequirementExtractionReport:
        """Extracts requirements across all pages of a Phase 2 ExtractedDocument."""
        all_reqs: List[ExtractedRequirement] = []

        # Derive tender identifier for clean requirement IDs
        doc_name = getattr(doc, "file_name", "TENDER.pdf")
        tender_stem = os.path.splitext(doc_name)[0]
        id_prefix = f"REQ-{tender_stem}"

        total_pages = len(getattr(doc, "pages", []))

        for p in getattr(doc, "pages", []):
            page_num = p.page_number
            text = p.cleaned_text
            page_reqs = self.extract_from_text(
                text=text,
                source_document=doc_name,
                page_number=page_num,
                id_prefix=id_prefix
            )
            all_reqs.extend(page_reqs)

        # De-duplicate requirements by text similarity or identical clause text
        deduped = self._deduplicate_requirements(all_reqs)

        # Count metrics
        mand_count = sum(1 for r in deduped if r.is_mandatory is True)
        opt_count = sum(1 for r in deduped if r.is_mandatory is False)
        unc_count = sum(1 for r in deduped if r.uncertainty_flag)

        return RequirementExtractionReport(
            source_document=doc_name,
            total_pages=total_pages,
            total_requirements=len(deduped),
            mandatory_count=mand_count,
            optional_count=opt_count,
            uncertain_count=unc_count,
            nlp_engine=self.engine_name,
            requirements=deduped,
            notes=f"Extraction completed using {self.engine_name}. spaCy available: {SPACY_AVAILABLE}."
        )

    def _parse_requirement_clause(
        self,
        text: str,
        section: Optional[str],
        source_document: str,
        page_number: Optional[int],
        default_id: str
    ) -> Optional[ExtractedRequirement]:
        """Analyzes a candidate sentence and structures requirement attributes."""
        # 1. Check for clause numbering prefix (e.g. "2.1 Minimum Average Annual Turnover:")
        clause_id = default_id
        clause_num_match = re.match(r"^(\d+\.\d+)\s+([^:]+):\s*(.*)$", text)

        if clause_num_match:
            clause_number = clause_num_match.group(1)
            clause_id = f"{default_id}-{clause_number}"

        # 2. Check Modality (Mandatory vs Optional vs Threshold)
        is_mandatory, modality_notes, has_modality_uncertainty = self._determine_modality(text, section)

        # 3. Check for Incomplete Clause
        is_incomplete, incomplete_notes = self._check_incomplete(text)

        # 4. Determine Category
        category = self._determine_category(text, section)

        # 5. Extract Operator, Value, Unit, and Parameter
        operator, value, unit, parameter, extraction_notes, value_uncertainty = self._extract_value_and_operator(text, section)

        # 6. Synthesize Uncertainty and Review Notes
        uncertainty_flag = has_modality_uncertainty or is_incomplete or value_uncertainty
        all_notes = []
        if modality_notes:
            all_notes.append(modality_notes)
        if incomplete_notes:
            all_notes.append(incomplete_notes)
        if extraction_notes:
            all_notes.append(extraction_notes)

        review_notes = " | ".join(all_notes) if all_notes else "Clean requirement extraction."

        return ExtractedRequirement(
            requirement_id=clause_id,
            original_text=text.strip(),
            category=category,
            parameter=parameter,
            operator=operator,
            value=value,
            unit=unit,
            is_mandatory=is_mandatory,
            source_document=source_document,
            page_number=page_number,
            section_reference=section,
            uncertainty_flag=uncertainty_flag,
            review_notes=review_notes
        )

    def _build_pm_experience_requirement(
        self,
        text: str,
        section: Optional[str],
        source_document: str,
        page_number: Optional[int],
        base_id: str,
        parent_clause_id: str,
    ) -> Optional["ExtractedRequirement"]:
        """BUG 2 FIX: Builds the Project Manager *experience* sub-requirement.

        Clause 4.3 combines two criteria in one sentence:
          (a) PMP or PRINCE2 certification — handled by the main extraction path.
          (b) Minimum N years of *relevant* PM experience — emitted here as a
              separate ExtractedRequirement so it is matched against person-specific
              evidence only (not against the company's operating track record).

        If no explicit year count is found the requirement is still emitted with
        uncertainty_flag=True so it is evaluated as MISSING_EVIDENCE / UNCERTAIN
        rather than silently dropped.
        """
        exp_years: Optional[float] = None
        # Look for patterns such as "minimum of 8 years", "at least 8 years", "8 years of relevant"
        exp_match = re.search(
            r"(?:minimum\s+of|at\s+least|minimum|no\s+less\s+than)?\s*(\d+)\s+years?\s+of\s+(?:relevant|direct|related|applicable)\s+experience",
            text,
            re.IGNORECASE,
        )
        if not exp_match:
            # Fallback: any years pattern near the end of the clause
            exp_match = re.search(r"(\d+)\s+years?\s+of\s+(?:relevant|direct|related)", text, re.IGNORECASE)
        if exp_match:
            exp_years = float(exp_match.group(1))

        is_mandatory, _, _ = self._determine_modality(text, section)
        has_value_uncertainty = exp_years is None

        notes_parts = [
            "BUG-2-FIX: Project Manager experience sub-criterion emitted as a separate requirement.",
            "Only person-specific evidence naming the PM role is acceptable proof.",
            "Corporate company operating experience MUST NOT be used as a substitute.",
        ]
        if has_value_uncertainty:
            notes_parts.append("No explicit year count found in clause; numeric threshold could not be parsed.")

        return ExtractedRequirement(
            requirement_id=f"{base_id}-PM-EXP",
            original_text=text.strip(),
            category="Staffing & Experience",
            parameter="Project Manager relevant experience",
            operator=">=" if exp_years is not None else "NONE",
            value=exp_years,
            unit="years" if exp_years is not None else None,
            is_mandatory=is_mandatory if is_mandatory is not None else True,
            source_document=source_document,
            page_number=page_number,
            section_reference=section,
            uncertainty_flag=has_value_uncertainty,
            review_notes=" | ".join(notes_parts),
        )

    def _determine_modality(self, text: str, section: Optional[str] = None) -> Tuple[Optional[bool], str, bool]:
        """Evaluates whether clause is mandatory, optional/preferred, or ambiguous."""
        has_mandatory_modal = any(re.search(pat, text, re.IGNORECASE) for pat in self.MANDATORY_MODAL_PATTERNS)
        has_threshold = any(re.search(pat, text, re.IGNORECASE) for pat in self.MANDATORY_THRESHOLD_PATTERNS)
        has_optional = any(re.search(pat, text, re.IGNORECASE) for pat in self.OPTIONAL_PATTERNS)

        if (has_mandatory_modal or has_threshold) and has_optional:
            # Conflicting signals, e.g. "must submit X, though Y is preferred"
            return True, "Mandatory requirement with secondary preferred criteria noted. Marked for reviewer attention.", True

        if has_mandatory_modal:
            return True, "Mandatory clause (deontic trigger: 'must'/'shall'/'required').", False

        if has_threshold:
            return True, "Mandatory requirement: establishes strict floor threshold ('minimum'/'at least').", False

        if has_optional:
            return False, "Optional/evaluative clause (trigger: 'should'/'preferred'/'recommended'). Non-disqualifying.", False

        # In procurement, clauses with explicit SLA or standard specs in technical/financial sections
        if section and any(kw in section.lower() for kw in ["financial", "technical", "statutory", "sla"]):
            if any(re.search(pat, text, re.IGNORECASE) for pat in [r"\b\d+%\b", r"\b\d+\s*(?:years|months|gbps|mbps)\b"]):
                return True, "Mandatory specification clause established by technical/financial section criteria.", False

        # No explicit modal or threshold detected
        return None, "Uncertain modality: neither mandatory modal ('must'/'shall') nor optional keyword detected. Flagged for review.", True

    def _check_incomplete(self, text: str) -> Tuple[bool, str]:
        """Detects whether clause appears truncated or ends on a trailing preposition."""
        # 1. Trailing preposition/conjunction
        if self.INCOMPLETE_TAIL_REGEX.search(text.strip()):
            return True, "Incomplete statement: clause ends with trailing preposition/conjunction without completing object."

        # 2. Abnormally short fragment lacking verb
        words = text.strip().split()
        if len(words) < 5 and not text.strip().endswith((".", "!", "?")):
            return True, "Truncated clause fragment detected."

        return False, ""

    def _determine_category(self, text: str, section: Optional[str]) -> str:
        """Determines functional category using section context and clause keywords."""
        combined = f"{section or ''} {text}".lower()

        if any(w in combined for w in ["turnover", "solvency", "financial", "net worth", "equity", "audited balance sheet"]):
            return "Financial Capability"
        if any(w in combined for w in ["uptime", "sla", "throughput", "bandwidth", "latency", "aes-256", "tls 1.3", "encryption", "scada", "telemetry"]):
            return "Technical Specifications"
        if any(w in combined for w in ["experience", "track record", "project manager", "pmp", "prince2", "staffing", "engineers", "operating track record", "reference"]):
            return "Staffing & Experience"
        if any(w in combined for w in ["gstin", "pan", "itr", "make in india", "local content", "maf", "oem authorization", "gem"]):
            return "GeM & Statutory Procurement Mandates"
        if any(w in combined for w in ["warranty", "iso", "debarment", "blacklisted", "accreditation", "maintenance"]):
            return "Statutory Accreditations & Warranties"

        return "General / Technical Scope"

    def _extract_value_and_operator(
        self,
        text: str,
        section: Optional[str]
    ) -> Tuple[str, Optional[Union[float, str]], Optional[str], str, str, bool]:
        """Extracts operator, numeric/standard value, unit, and parameter name.

        BUG 1 FIX: Operator is now determined from the context window AROUND each
        extracted numeric value, not from a single global scan over the full sentence.
        This prevents a downstream SLA metric (e.g. 'inside 15 minutes') from
        overriding the operator for an earlier warranty metric ('at least 36 months')
        in the same compound clause.

        BUG 3 FIX: MAF/OEM Authorization check is now performed BEFORE duration
        extraction so that clauses such as 'Tender-specific MAF with back-to-back
        36 months warranty if bidder is reseller' are classified as MANDATORY_SUBMISSION
        rather than incorrectly extracted as a warranty-duration requirement.

        BUG 2 FIX: Project Manager clause is detected and decomposed separately;
        the PMP/PRINCE2 certification sub-criterion is returned with the correct
        parameter name, and the experience sub-criterion is explicitly labelled as
        'Project Manager relevant experience' so that it cannot be confused with
        the company's corporate operating track record.
        """

        # -----------------------------------------------------------------------
        # PRE-PASS: Statutory / special-form checks that must win over any numeric
        # extraction (they name the primary subject of the clause).
        # -----------------------------------------------------------------------

        # BUG 3 FIX: OEM Authorization / MAF check BEFORE duration extraction.
        # Clause 6.4 mentions '36 months' only as a warranty condition; the
        # primary obligation is the MAF submission itself.
        if "maf" in text.lower() or "oem authorization" in text.lower():
            return "MANDATORY_SUBMISSION", "OEM_MAF_LETTER", None, "tender-specific OEM authorization (MAF)", "Statutory OEM authorization mandate.", False

        if "gstin" in text.lower():
            return "MANDATORY_SUBMISSION", "VALID_GSTIN_FILING", None, "active GSTIN registration & filings", "Mandatory statutory registration requirement.", False
        if "pan" in text.lower() and "itr" in text.lower():
            return "MANDATORY_SUBMISSION", "PAN_AND_ITR_COPIES", None, "PAN card and 3-year ITR submissions", "Mandatory tax compliance requirement.", False
        if "blacklisted" in text.lower() or "debarred" in text.lower():
            return "MANDATORY_SUBMISSION", "NON_BLACKLISTING_AFFIDAVIT", None, "solemn non-blacklisting declaration", "Integrity and debarment compliance mandate.", False

        # BUG 2 FIX: Project Manager clause decomposition.
        # Clause 4.3 ('The proposed Project Manager must possess a valid PMP or
        # PRINCE2 certification and a minimum of 8 years of relevant experience')
        # has TWO distinct criteria.  We first return the certification criterion;
        # the experience sub-criterion is captured as a separate requirement by
        # _decompose_project_manager_clause() called from _parse_requirement_clause.
        if re.search(r"\bproject\s+manager\b", text, re.IGNORECASE) and (
            re.search(r"\bpmp\b", text, re.IGNORECASE) or
            re.search(r"\bprince2\b", text, re.IGNORECASE)
        ):
            # Return certification criterion (the experience sub-req is handled separately)
            return "CONTAINS", "PMP_OR_PRINCE2_CERTIFICATION", None, "Project Manager PMP/PRINCE2 certification", "Mandatory Project Manager certification requirement (PMP or PRINCE2). Experience sub-criterion extracted separately.", False

        # -----------------------------------------------------------------------
        # OPERATOR HELPER: determine operator from a local context window
        # -----------------------------------------------------------------------
        def _local_operator(ctx: str) -> str:
            """Returns the dominant operator found in a short context string.

            Evaluation order: EQ > GTE > LTE > NONE.
            GTE is evaluated before LTE so 'at least' beats 'inside'/'within'
            when both appear in the same multi-metric sentence.
            """
            if any(re.search(pat, ctx, re.IGNORECASE) for pat in self.OPERATOR_EQ_PATTERNS):
                return "=="
            if any(re.search(pat, ctx, re.IGNORECASE) for pat in self.OPERATOR_GTE_PATTERNS):
                return ">="
            if any(re.search(pat, ctx, re.IGNORECASE) for pat in self.OPERATOR_LTE_PATTERNS):
                return "<="
            return "NONE"

        # Context window size (characters before the matched value)
        CTX_WINDOW = 60

        # -----------------------------------------------------------------------
        # 1. Percentage values (e.g. 99.95%, 50.0%, 5%)
        # -----------------------------------------------------------------------
        pct_match = re.search(r"(\d+(?:\.\d+)?)\s*(%|percent)", text, re.IGNORECASE)
        if pct_match:
            val = float(pct_match.group(1))
            unit = "%"
            param = "parameter_percentage"
            if "uptime" in text.lower() or "availability" in text.lower():
                param = "monthly uptime SLA"
            elif "local" in text.lower() or "domestic" in text.lower():
                param = "local domestic content"
            elif "drop" in text.lower() or "loss" in text.lower() or "error" in text.lower():
                param = "maximum acceptable error/loss rate"
            ctx_start = max(0, pct_match.start() - CTX_WINDOW)
            local_ctx = text[ctx_start: pct_match.end()]
            operator = _local_operator(local_ctx)
            if operator == "NONE":
                operator = ">="
            return operator, val, unit, param, "Numeric percentage extracted.", False

        # -----------------------------------------------------------------------
        # 2. Financial amounts (e.g. USD 10.0 Million, USD 2.0 Million)
        # -----------------------------------------------------------------------
        fin_match = re.search(r"(?:USD|\$)\s*(\d+(?:\.\d+)?)\s*(Million|M|Billion|B|K)?", text, re.IGNORECASE)
        if fin_match:
            val = float(fin_match.group(1))
            mult = fin_match.group(2) or "Million"
            unit = f"USD {mult}" if mult else "USD"
            param = "audited financial turnover" if "turnover" in text.lower() else "net worth / solvency"
            ctx_start = max(0, fin_match.start() - CTX_WINDOW)
            local_ctx = text[ctx_start: fin_match.end()]
            operator = _local_operator(local_ctx)
            if operator == "NONE":
                operator = ">="
            return operator, val, unit, param, "Financial threshold extracted.", False

        # -----------------------------------------------------------------------
        # 3. Throughput / Bandwidth (e.g. 1 Gbps, 10 Gbps, 500 MB/s)
        # -----------------------------------------------------------------------
        bw_match = re.search(r"(\d+(?:\.\d+)?)\s*(Gbps|Mbps|MB\/s|GB\/s|kbps)", text, re.IGNORECASE)
        if bw_match:
            val = float(bw_match.group(1))
            unit = bw_match.group(2)
            param = "network data throughput capacity"
            ctx_start = max(0, bw_match.start() - CTX_WINDOW)
            local_ctx = text[ctx_start: bw_match.end()]
            operator = _local_operator(local_ctx)
            if operator == "NONE":
                operator = ">="
            return operator, val, unit, param, "Network capacity threshold extracted.", False

        # -----------------------------------------------------------------------
        # 4. Time durations / Warranties
        # BUG 1 FIX: The operator is derived from the local context around the
        # FIRST matched duration, not from a global scan.  This ensures that
        # 'at least 36 months' produces >= even when the same sentence also
        # contains 'inside 15 minutes' (which would otherwise yield <= globally).
        # -----------------------------------------------------------------------
        dur_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(years|year|months|month|days|day|hours|hour|minutes|minute|mins)",
            text,
            re.IGNORECASE
        )
        if dur_match:
            val = float(dur_match.group(1))
            unit = dur_match.group(2).lower()
            ctx_start = max(0, dur_match.start() - CTX_WINDOW)
            local_ctx = text[ctx_start: dur_match.end()]
            operator = _local_operator(local_ctx)

            if "warranty" in text.lower():
                param = "comprehensive warranty duration"
                if operator == "NONE":
                    operator = ">="
            elif "incident response" in text.lower() or "p1" in text.lower() or "response inside" in text.lower():
                param = "critical incident response time"
                if operator == "NONE":
                    operator = "<="
            elif "fix" in text.lower() or "resolution" in text.lower():
                param = "incident on-site resolution time"
                if operator == "NONE":
                    operator = "<="
            elif "experience" in text.lower() or "operating" in text.lower():
                param = "continuous commercial operating experience"
                if operator == "NONE":
                    operator = ">="
            else:
                param = f"duration in {unit}"
            return operator, val, unit, param, "Duration metric extracted (operator scoped to local context around matched value).", False

        # -----------------------------------------------------------------------
        # 5. Numerical counts
        # -----------------------------------------------------------------------
        count_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(units?|items?|nodes?|servers?|licenses?|instances?|devices?|sensors?|similar\s+reference\s+projects?|reference\s+hospital\s+deployments?|contracts?|certified\s+engineers?|dedicated\s+certified\s+engineers?|engineers?|projects?)",
            text,
            re.IGNORECASE
        )
        if count_match:
            val = float(count_match.group(1))
            unit = count_match.group(2).lower()
            param = f"quantity of {unit}"
            if "project" in unit or "contract" in unit or "deployment" in unit:
                param = "reference project scale"
            elif "engineer" in unit:
                param = "dedicated certified engineers"
            ctx_start = max(0, count_match.start() - CTX_WINDOW)
            local_ctx = text[ctx_start: count_match.end()]
            operator = _local_operator(local_ctx)
            if operator == "NONE":
                operator = ">="
            return operator, val, unit, param, "Count metric extracted.", False

        # -----------------------------------------------------------------------
        # 6. Known Accreditations, Encryption & Technical Standards
        # -----------------------------------------------------------------------
        for std_key, std_name in self.STANDARDS_MAP.items():
            if re.search(r"\b" + re.escape(std_key) + r"\b", text, re.IGNORECASE):
                return "CONTAINS", std_name, None, f"{std_name} compliance", "Specific technical standard extracted.", False

        # -----------------------------------------------------------------------
        # 7. Generic or Ambiguous clause without explicit metric
        # -----------------------------------------------------------------------
        return "NONE", None, None, "unspecified requirement criteria", "No explicit numeric value or recognized standard found. Flagged for review without inventing values.", True

    def _is_non_requirement_line(self, line: str) -> bool:
        """Filters out non-clause lines like document titles, table headers, tender references, and notices."""
        clean = line.strip().lower()
        if not clean or len(clean) < 10:
            return True
        if "tender reference:" in clean or "authority:" in clean or "procurement method" in clean:
            return True
        if clean.startswith("category ") or "simulation status" in clean or "legal notice" in clean:
            return True
        if "department of public" in clean or "academic/nlp benchmark" in clean:
            return True
        # BUG 4 FIX: Filter document-level disclaimer/metadata text that is not a bidder compliance criterion.
        # "It does not represent an actual public solicitation" is a legal notice, not a requirement.
        if "does not represent an actual public solicitation" in clean:
            return True
        if "simulated artifact" in clean or "strictly for testing" in clean:
            return True
        # Filter document title lines that appear at the beginning of tenders
        if clean.startswith("enterprise cloud hosting") or clean.startswith("hospital information management") or clean.startswith("smart industrial solar") or clean.startswith("real-time transaction fraud"):
            return True
        return False

    def _deduplicate_requirements(self, reqs: List[ExtractedRequirement]) -> List[ExtractedRequirement]:
        """Eliminates redundant duplicates while preserving ordering."""
        seen = set()
        deduped = []
        for r in reqs:
            # Key based on parameter, value, operator, and page number
            key = (r.parameter, str(r.value), r.operator, r.page_number)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(r)
        return deduped


# -------------------------------------------------------------
# Convenience Functions
# -------------------------------------------------------------
def extract_requirements_from_pdf(pdf_path: str) -> RequirementExtractionReport:
    """Convenience helper to extract requirements from a tender PDF file."""
    from backend.extraction.pdf_extractor import PDFExtractor
    extractor = PDFExtractor(strip_simulation_banners=True)
    doc = extractor.extract(pdf_path)
    req_extractor = RequirementExtractor()
    return req_extractor.extract_from_extracted_document(doc)


def extract_requirements_from_text(text: str, source_document: str = "TEXT_INPUT") -> List[ExtractedRequirement]:
    """Convenience helper to extract requirements directly from text."""
    extractor = RequirementExtractor()
    return extractor.extract_from_text(text, source_document=source_document)
