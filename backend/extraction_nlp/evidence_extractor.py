"""
evidence_extractor.py - Structured Bidder Evidence Extraction Pipeline
Part of Phase 3: Requirement & Evidence Extraction

Features:
1. Extracts structured evidence claims from bidder proposals, technical responses, and financial annexures.
2. For each evidence record, captures:
   - evidence_id (unique record identifier)
   - original_sentence (exact sentence text)
   - extracted_entities (organizations, standards, certificates, tax IDs, dates, personnel)
   - numeric_values (extracted numeric metrics, percentages, currencies, durations, units, and parameter hints)
   - source_document & page_number (preserves exact 1-indexed document origin)
   - section_reference (e.g. "Section 2.1 - Audited Financial Overview")
   - has_negation (flags negative polarity: 'not', 'never', 'cannot', 'without')
   - uncertainty_flag & review_notes (flags vague, promotional, or unverified claims for review)
3. Transparent rules, regular expressions, and optional spaCy integration.
   - Does NOT claim pre-trained model training.
   - Auditable, deterministic, and explainable.
4. Seamless integration with Phase 2 PDFExtractor and ExtractedDocument.
"""

import os
import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple

from backend.extraction_nlp.number_normalizer import NumberWordNormalizer

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
class ExtractedEvidence:
    """Represents a single extracted evidence statement from a bidder document."""
    evidence_id: str                              # e.g. "EVD-001", "EVD-ALPHA-01"
    original_sentence: str                        # Full unmodified sentence
    extracted_entities: List[Dict[str, Any]] = field(default_factory=list) # [{text, label, start, end}]
    numeric_values: List[Dict[str, Any]] = field(default_factory=list)     # [{raw, value, unit, parameter_hint}]
    source_document: str = "N/A"                  # Bidder PDF filename
    page_number: int = 1                          # 1-indexed page number
    section_reference: Optional[str] = None       # Section heading or title
    has_negation: bool = False                    # True if negative polarity detected
    uncertainty_flag: bool = False                # True if ambiguous, promotional, or unverified
    review_notes: str = ""                        # Rationale for reviewer guidance

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceExtractionReport:
    """Summary report of evidence extracted from a bidder document."""
    source_document: str
    bidder_name: str
    total_pages: int
    total_evidence_records: int
    records_with_numeric_values: int
    records_with_entities: int
    records_with_negation: int
    uncertain_records_count: int
    nlp_engine: str
    evidence_records: List[ExtractedEvidence] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evidence_records"] = [e.to_dict() for e in self.evidence_records]
        return d


class EvidenceExtractor:
    """Rule-based and regex-driven procurement bidder evidence extractor."""

    # 1. Negation Keywords (critical for polarity inversion)
    NEGATION_REGEX = re.compile(
        r"\b(not|never|no|does\s+not|do\s+not|did\s+not|cannot|can't|without|excluding|except)\b",
        re.IGNORECASE
    )

    # 2. Vague or Promotional Indicators (trigger uncertainty_flag)
    VAGUE_PHRASES = [
        "collective domain knowledge",
        "approved in principle",
        "in principle",
        "unexecuted pipeline estimates",
        "not recognized under gaap",
        "internal security reviews aligned with",
        "strong solvent standing",
        "vast experience",
        "highly skilled team",
        "decades of collective",
    ]

    # 3. Known Standards & Certifications
    STANDARDS_REGEX = re.compile(
        r"\b(ISO(?:\/IEC)?\s+\d{4,5}(?::\d{4})?|SOC\s+[12]\s+Type\s+(?:I|II)|CMMI\s+Level\s+[1-5]|PMI\s+PMP|PMP|PRINCE2|HIPAA|GDPR)\b",
        re.IGNORECASE
    )

    # 4. Cryptographic Standards
    CRYPTO_REGEX = re.compile(
        r"\b(AES-256|AES-128|TLS\s+1\.[23]|SHA-256|RSA-2048|RSA-4096)\b",
        re.IGNORECASE
    )

    # 5. Tax & Legal Identifiers
    GSTIN_REGEX = re.compile(r"\b(\d{2}[A-Z]{5}\d{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1})\b")
    PAN_REGEX = re.compile(r"\b([A-Z]{5}\d{4}[A-Z]{1})\b")
    CERT_ID_REGEX = re.compile(
        r"\b(?:Cert(?:\s+No)?|License|Ref|Registration)\s*[:#]\s*([A-Za-z0-9\-]+)\b",
        re.IGNORECASE
    )

    # 6. Fiscal Dates & Years
    DATE_REGEX = re.compile(
        r"\b(FY\s*\d{4}(?:[-/]\d{2,4})?|AY\s+\d{4}-\d{2}|[A-Z][a-z]+\s+\d{4}|(?:19|20)\d{2})\b"
    )

    # 7. Corporate Organization Suffixes
    ORG_REGEX = re.compile(
        r"\b([A-Z][A-Za-z0-9&.\s]{2,40}?(?:Ltd|Inc|Corp|Corporation|Pvt\s+Ltd|LLP|Systems|Solutions|Technologies|Services|Foundation|Bank|Authority|Commission))\b"
    )

    def __init__(self, use_spacy: bool = True):
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        self.engine_name = "spacy_with_rules" if self.use_spacy else "regex_rules"
        self.number_normalizer = NumberWordNormalizer()

    def clean_text_for_nlp(self, text: str) -> str:
        """Removes PDF watermark line artifacts and normalizes sentence continuity."""
        if not text:
            return ""
        lines = []
        for line in text.split("\n"):
            line_str = line.strip()
            # Skip simulation banners
            if "*** SIMULATED" in line_str or "SIMULATED PROCUREMENT DATASET" in line_str:
                continue
            if "SIMULATION NOTICE:" in line_str or "academic nlp" in line_str.lower():
                continue
            # Skip single isolated uppercase letters from diagonal watermark
            if len(line_str) == 1 and line_str.isupper():
                continue
            lines.append(line)

        cleaned = "\n".join(lines)
        # Clean merged watermark characters: e.g. "PositiveA net" -> "Positive net"
        cleaned = re.sub(r"(?<=[a-z])([A-Z])(?=\s+[a-z])", "", cleaned)
        # Collapse line-wrapped sentences: join lines unless next line starts section or capital clause
        cleaned = re.sub(r"(?<![\.:\?!])\n(?!\s*Section|\s*\d{1,2}\.\d{1,2}\s+[A-Za-z]|\s*[A-Z\s]{4,}:)", " ", cleaned)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        return cleaned.strip()

    def split_into_sentences(self, text: str) -> List[Tuple[str, Optional[str]]]:
        """Splits page text into candidate evidence statements with active section context."""
        cleaned = self.clean_text_for_nlp(text)
        if not cleaned:
            return []

        results = []
        current_section = "Bidder Proposal"

        # Split into blocks by newlines
        blocks = [b.strip() for b in cleaned.split("\n") if b.strip()]

        for block in blocks:
            # Check for section headers (e.g. "Section 2.1 - Audited Financial Overview:")
            sec_match = re.search(r"(Section\s+\d+\.\d+\s*-[^:]+):?", block)
            if sec_match:
                current_section = sec_match.group(1).strip()

            # Split block into individual sentences
            sents = self._split_sentences(block)
            for s in sents:
                s_clean = s.strip()
                if s_clean:
                    # Update local section if sentence itself has a section header prefix
                    local_sec = current_section
                    local_match = re.match(r"^(Section\s+\d+\.\d+\s*-[^:]+):\s*(.*)$", s_clean)
                    if local_match:
                        local_sec = local_match.group(1).strip()
                        s_clean = local_match.group(2).strip() or s_clean

                    results.append((s_clean, local_sec))

        return results

    def _split_sentences(self, text: str) -> List[str]:
        """Sentence splitter using spaCy sentencizer if available, or protected regex."""
        if self.use_spacy and _nlp is not None:
            doc = _nlp(text)
            return [s.text for s in doc.sents if s.text.strip()]

        # Pure-Python protected sentence splitter
        splits = re.split(r"(?<!\bUSD)(?<!\bINR)(?<!\bRs)(?<!\bPvt)(?<!\bLtd)(?<!\bInc)(?<!\bCorp)(?<!\bCo)(?<!\be\.g)(?<!\bi\.e)(?<!\bDr)(?<!\bNo)(?<!\bRef)(?<!\bAY)(?<!\bFY\d{4})[\.\?!]\s+(?=[A-Z0-9])", text)
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
        source_document: str = "BIDDER_PROPOSAL_TEXT",
        page_number: int = 1,
        id_prefix: str = "EVD"
    ) -> List[ExtractedEvidence]:
        """Extracts structured evidence records from text of a single page or excerpt."""
        pairs = self.split_into_sentences(text)
        evidence_records: List[ExtractedEvidence] = []
        counter = 1

        for sent, section in pairs:
            if self._is_non_evidence_line(sent):
                continue

            record = self._process_evidence_sentence(
                sentence=sent,
                section=section,
                source_document=source_document,
                page_number=page_number,
                evidence_id=f"{id_prefix}-{counter:03d}"
            )
            if record:
                evidence_records.append(record)
                counter += 1

        return evidence_records

    def extract_from_extracted_document(self, doc: Any) -> EvidenceExtractionReport:
        """Extracts evidence records across all pages of a Phase 2 ExtractedDocument."""
        all_records: List[ExtractedEvidence] = []

        doc_name = getattr(doc, "file_name", "BIDDER_PROPOSAL.pdf")
        stem = os.path.splitext(doc_name)[0]
        id_prefix = f"EVD-{stem}"

        total_pages = len(getattr(doc, "pages", []))
        bidder_name = self._detect_bidder_name(doc)

        for p in getattr(doc, "pages", []):
            page_num = p.page_number
            text = p.cleaned_text
            page_records = self.extract_from_text(
                text=text,
                source_document=doc_name,
                page_number=page_num,
                id_prefix=id_prefix
            )
            all_records.extend(page_records)

        # De-duplicate identical sentences
        deduped = self._deduplicate_evidence(all_records)

        # Compute summary metrics
        numeric_count = sum(1 for e in deduped if len(e.numeric_values) > 0)
        entity_count = sum(1 for e in deduped if len(e.extracted_entities) > 0)
        negation_count = sum(1 for e in deduped if e.has_negation)
        uncertain_count = sum(1 for e in deduped if e.uncertainty_flag)

        return EvidenceExtractionReport(
            source_document=doc_name,
            bidder_name=bidder_name,
            total_pages=total_pages,
            total_evidence_records=len(deduped),
            records_with_numeric_values=numeric_count,
            records_with_entities=entity_count,
            records_with_negation=negation_count,
            uncertain_records_count=uncertain_count,
            nlp_engine=self.engine_name,
            evidence_records=deduped,
            notes=f"Extraction completed using {self.engine_name}. spaCy available: {SPACY_AVAILABLE}."
        )

    def _process_evidence_sentence(
        self,
        sentence: str,
        section: Optional[str],
        source_document: str,
        page_number: int,
        evidence_id: str
    ) -> Optional[ExtractedEvidence]:
        """Analyzes a single evidence claim sentence to extract entities, metrics, and flags."""
        clean_sent = sentence.strip()
        if len(clean_sent) < 15:
            return None

        # 1. Detect Negation
        has_negation = bool(self.NEGATION_REGEX.search(clean_sent))

        # 2. Detect Entities
        entities = self._extract_entities(clean_sent)

        # 3. Detect Numeric Metrics
        numeric_metrics = self._extract_numeric_metrics(clean_sent)

        # 4. Detect Uncertainty / Promotional Language
        is_uncertain, uncertain_notes = self._detect_uncertainty(clean_sent)

        # 5. Build Review Notes
        notes_parts = []
        if has_negation:
            notes_parts.append("Negative polarity / negation keyword detected.")
        if is_uncertain:
            notes_parts.append(uncertain_notes)
        if numeric_metrics:
            metrics_summary = ", ".join(f"{m['raw']} ({m['parameter_hint']})" for m in numeric_metrics)
            notes_parts.append(f"Extracted metrics: {metrics_summary}.")
        if entities:
            ent_summary = ", ".join(f"{e['text']} [{e['label']}]" for e in entities)
            notes_parts.append(f"Detected entities: {ent_summary}.")

        review_notes = " | ".join(notes_parts) if notes_parts else "Standard factual proposal statement."

        return ExtractedEvidence(
            evidence_id=evidence_id,
            original_sentence=clean_sent,
            extracted_entities=entities,
            numeric_values=numeric_metrics,
            source_document=source_document,
            page_number=page_number,
            section_reference=section,
            has_negation=has_negation,
            uncertainty_flag=is_uncertain,
            review_notes=review_notes
        )

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extracts named entities using rule-based and regex matchers."""
        entities = []
        seen_spans = set()

        def add_entity(match_text: str, label: str, start: int, end: int):
            span_key = (start, end)
            if span_key not in seen_spans and match_text.strip():
                seen_spans.add(span_key)
                entities.append({
                    "text": match_text.strip(),
                    "label": label,
                    "start_char": start,
                    "end_char": end
                })

        # A. Cryptographic Standards
        for m in self.CRYPTO_REGEX.finditer(text):
            add_entity(m.group(1), "CRYPTO_STANDARD", m.start(1), m.end(1))

        # B. Quality & Security Standards
        for m in self.STANDARDS_REGEX.finditer(text):
            add_entity(m.group(1), "STANDARD_CERT", m.start(1), m.end(1))

        # C. Tax & Registration Identifiers (GSTIN & PAN)
        for m in self.GSTIN_REGEX.finditer(text):
            add_entity(m.group(1), "GSTIN", m.start(1), m.end(1))

        for m in self.PAN_REGEX.finditer(text):
            # Avoid false positives with common all-caps words
            if m.group(1) not in ["PROPOSAL", "NOTICE", "ANNEX"]:
                add_entity(m.group(1), "PAN", m.start(1), m.end(1))

        # D. Certificate & License Numbers
        for m in self.CERT_ID_REGEX.finditer(text):
            add_entity(m.group(1), "CERT_OR_LICENSE_ID", m.start(1), m.end(1))

        # E. Fiscal Dates & Years
        for m in self.DATE_REGEX.finditer(text):
            add_entity(m.group(1), "DATE_YEAR", m.start(1), m.end(1))

        # F. Corporate Organizations
        for m in self.ORG_REGEX.finditer(text):
            org_candidate = m.group(1).strip()
            # Filter non-org noisy capitalized phrases
            if not any(stop in org_candidate for stop in ["Section", "Tender Reference", "Technical", "Commercial", "Simulation"]):
                add_entity(org_candidate, "ORG", m.start(1), m.end(1))

        # G. Personnel Names
        person_match = re.search(r"\b(Dr\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+)\b", text)
        if person_match:
            add_entity(person_match.group(1), "PERSON", person_match.start(1), person_match.end(1))

        # Sort entities by start_char
        entities.sort(key=lambda x: x["start_char"])
        return entities

    def _extract_numeric_metrics(self, text: str) -> List[Dict[str, Any]]:
        """Extracts numeric metrics, percentages, currencies, durations, and units."""
        metrics = []

        # 1. Percentages (e.g. 99.99%, 99.90%, 50.0%, 62%)
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(%|percent)", text, re.IGNORECASE):
            val = float(m.group(1))
            raw = m.group(0).strip()
            hint = "uptime_or_availability" if ("uptime" in text.lower() or "availability" in text.lower()) else "percentage"
            if "domestic" in text.lower() or "local" in text.lower():
                hint = "local_domestic_content"
            metrics.append({
                "raw": raw,
                "value": val,
                "unit": "%",
                "parameter_hint": hint
            })

        # 2. Financial Amounts
        # 2a. Indian Financial Amounts (e.g. ₹10 crore, INR 15 Cr, Rs. 50 Lakh, 150 lakh rupees)
        inr_pat = re.compile(
            r"(?:(?:INR|Rs\.?|₹|Rupees?)\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(Crores?|Cr\.?|Lakhs?|Lacs?|Thousand|K)?\b|"
            r"(\d+(?:,\d+)*(?:\.\d+)?)\s*(Crores?|Cr\.?|Lakhs?|Lacs?)\b(?:\s*(?:INR|Rs\.?|₹|Rupees?))?|"
            r"(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:INR|Rs\.?|₹|Rupees?)\b)",
            re.IGNORECASE
        )
        for m in inr_pat.finditer(text):
            g = m.groups()
            if g[0] is not None:
                val = float(g[0].replace(",", ""))
                scale = g[1]
            elif g[2] is not None:
                val = float(g[2].replace(",", ""))
                scale = g[3]
            else:
                val = float(g[4].replace(",", ""))
                scale = None

            if scale:
                scale_lower = scale.lower()
                if scale_lower.startswith("cr"):
                    unit = "INR Crore"
                elif scale_lower.startswith("l"):
                    unit = "INR Lakh"
                elif scale_lower.startswith("t") or scale_lower == "k":
                    unit = "INR Thousand"
                else:
                    unit = f"INR {scale}"
            else:
                unit = "INR"

            raw = m.group(0).strip()
            hint = "annual_turnover" if ("turnover" in text.lower() or "revenue" in text.lower()) else "net_worth_solvency"
            if "project" in text.lower() or "commissioned" in text.lower():
                hint = "project_valuation"
            metrics.append({
                "raw": raw,
                "value": val,
                "unit": unit,
                "parameter_hint": hint
            })

        # 2b. USD / Foreign Financial Amounts (e.g. USD 14.5 Million, USD 4.5 Million, USD 6.2 Million)
        for m in re.finditer(r"(?:USD|\$)\s*(\d+(?:\.\d+)?)\s*(Million|M|Billion|B|K)?", text, re.IGNORECASE):
            val = float(m.group(1))
            mult = m.group(2) or "Million"
            unit = f"USD {mult}" if mult else "USD"
            raw = m.group(0).strip()
            hint = "annual_turnover" if ("turnover" in text.lower() or "revenue" in text.lower()) else "net_worth_solvency"
            if "project" in text.lower() or "commissioned" in text.lower():
                hint = "project_valuation"
            metrics.append({
                "raw": raw,
                "value": val,
                "unit": unit,
                "parameter_hint": hint
            })

        # 3. Throughput / Bandwidth (e.g. 1,000 Mbps, 1 Gbps, 10 Gbps, 500 Mbps, 4 Gbps)
        for m in re.finditer(r"(\d+(?:,\d+)?(?:\.\d+)?)\s*(Gbps|Mbps|MB\/s|GB\/s|kbps)", text, re.IGNORECASE):
            raw_val = m.group(1).replace(",", "")
            val = float(raw_val)
            unit = m.group(2)
            raw = m.group(0).strip()
            metrics.append({
                "raw": raw,
                "value": val,
                "unit": unit,
                "parameter_hint": "data_throughput"
            })

        # 4. Durations / Time (e.g. 8 years, 3 years, 36 months, 10 minutes, 2 hours, 18 months)
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(years|year|months|month|days|day|hours|hour|minutes|minute|mins)", text, re.IGNORECASE):
            val = float(m.group(1))
            unit = m.group(2).lower()
            raw = m.group(0).strip()
            hint = "duration"
            if "experience" in text.lower() or "operational" in text.lower() or "incorporated" in text.lower():
                hint = "corporate_experience"
            elif "warranty" in text.lower():
                hint = "warranty_period"
            elif "response" in text.lower() or "initial" in text.lower():
                hint = "incident_response_time"
            elif "dispatch" in text.lower() or "patch" in text.lower() or "fix" in text.lower() or "resolution" in text.lower():
                hint = "incident_fix_time"
            metrics.append({
                "raw": raw,
                "value": val,
                "unit": unit,
                "parameter_hint": hint
            })

        # 5. Counts (e.g. 4 certified client project completions, 5 full-time certified engineers, 6 full-time senior certified DBAs, 175 people)
        for m in re.finditer(r"(\d+)\s*(certified\s+client\s+project\s+completions?|(?:(?:full-time|dedicated|senior|certified)\s+)*(?:engineers?|dbas?|database\s+administrators?|architects?|specialists?|analysts?|consultants?|developers?)|on-site\s+engineers?|hospital\s+deployments?|medical\s+centers?|employees?|personnel|staff|people)", text, re.IGNORECASE):
            val = float(m.group(1))
            unit = m.group(2).lower()
            raw = m.group(0).strip()
            if any(k in unit for k in ["employee", "people", "staff", "personnel", "engineer", "dba", "administrator", "architect", "specialist"]):
                hint = "headcount"
            elif any(k in unit for k in ["project", "deployment", "center"]):
                hint = "project_count"
            else:
                hint = "count"
            metrics.append({
                "raw": raw,
                "value": val,
                "unit": unit,
                "parameter_hint": hint
            })

        # 6. Normalized Word-Numbers and Duration Idioms (Phase 8 Step 2A)
        # Recognizes spelled-out English numbers, duration idioms, and bounded expressions
        if hasattr(self, "number_normalizer") and self.number_normalizer:
            normalized_list = self.number_normalizer.extract_normalized_numbers(text)
            for norm in normalized_list:
                # Avoid duplicate metrics if already extracted by regex digit extractors
                is_duplicate = False
                for existing in metrics:
                    if existing.get("parameter_hint") == norm.parameter_hint and abs(existing.get("value", 0.0) - norm.value) < 1e-4:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    metrics.append({
                        "raw": norm.raw_text,
                        "value": norm.value,
                        "unit": norm.unit,
                        "parameter_hint": norm.parameter_hint,
                        "operator": norm.operator,
                        "is_normalized_word": True,
                        "start_char": norm.start_char,
                        "end_char": norm.end_char
                    })

        return metrics

    def _detect_uncertainty(self, text: str) -> Tuple[bool, str]:
        """Flags promotional, vague, or unverified claims for auditor review."""
        lower = text.lower()
        for phrase in self.VAGUE_PHRASES:
            if phrase in lower:
                return True, f"Ambiguous or promotional claim ('{phrase}') without verifiable third-party certification or specific empirical metric."

        # Check for claims asserting qualifications without empirical figures
        if "decades of collective" in lower:
            return True, "Asserts collective individual knowledge rather than verified corporate operating history."

        if "approved in principle" in lower:
            return True, "Credit lines approved in principle without formal certified solvency certificate or balance sheet equity."

        if "unexecuted pipeline estimates" in lower:
            return True, "Disclosed revenue includes unexecuted pipeline estimates rejected under GAAP accounting standards."

        return False, ""

    def _detect_bidder_name(self, doc: Any) -> str:
        """Extracts bidder organization name from document metadata or title lines."""
        for p in getattr(doc, "pages", []):
            text = p.cleaned_text
            m = re.search(r"Bidder:\s*([^\n|]+)", text)
            if m:
                return m.group(1).strip()
            m2 = re.search(r"([A-Z][A-Za-z0-9&.\s]{3,40}?(?:Ltd|Inc|Corp|Corporation|Pvt\s+Ltd|LLP))", text)
            if m2 and "SIMULATED" not in m2.group(1):
                return m2.group(1).strip()
        return "Unknown Bidder"

    def _is_non_evidence_line(self, line: str) -> bool:
        """Filters out title headers, running notices, and simulation disclaimers."""
        clean = line.strip().lower()
        if not clean or len(clean) < 15:
            return True
        if clean.startswith("technical & commercial tender proposal"):
            return True
        if clean.startswith("bidder:") and "referencing:" in clean:
            return True
        if clean.startswith("tender reference:") or clean.startswith("authority:"):
            return True
        if "academic nlp benchmark" in clean or "simulated procurement dataset" in clean:
            return True
        if "not an official legal submission" in clean or "simulation notice" in clean:
            return True
        # Filter standalone section titles like "2.0 Financial Standing." or "4.0 Past Performance & Qualifications."
        if re.match(r"^(\d+\.\d+\s+[A-Za-z\s&/,]+)\.?$", line.strip()):
            return True
        return False

    def _deduplicate_evidence(self, records: List[ExtractedEvidence]) -> List[ExtractedEvidence]:
        """Removes duplicates while preserving document order."""
        seen = set()
        deduped = []
        for r in records:
            key = (r.original_sentence.lower(), r.page_number)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(r)
        return deduped


# -------------------------------------------------------------
# Convenience Functions
# -------------------------------------------------------------
def extract_evidence_from_pdf(pdf_path: str) -> EvidenceExtractionReport:
    """Convenience helper to extract evidence records from a bidder PDF file."""
    from backend.extraction.pdf_extractor import PDFExtractor
    extractor = PDFExtractor(strip_simulation_banners=True)
    doc = extractor.extract(pdf_path)
    evd_extractor = EvidenceExtractor()
    return evd_extractor.extract_from_extracted_document(doc)


def extract_evidence_from_text(text: str, source_document: str = "TEXT_INPUT") -> List[ExtractedEvidence]:
    """Convenience helper to extract evidence records directly from text."""
    extractor = EvidenceExtractor()
    return extractor.extract_from_text(text, source_document=source_document)
