"""
number_normalizer.py - Number-Word & Duration Normalization for Procurement NLP
Part of Phase 8: Core NLP Reliability Improvements

Converts spelled-out English numbers, duration idioms, and compound phrases into
canonical floating-point metrics while preserving:
1. Exact character offsets (start_char, end_char) in the original sentence.
2. Comparison operators ('>=', '>', '<=', '<', '==') based on contextual modifiers
   (e.g., 'over', 'more than', 'at least', 'no less than').
3. Verbatim source text quotations (never modifies original sentences).
4. Physical and procurement units (Gbps, Mbps, years, months, %, USD Million, count).

Strict Policy:
- Ambiguous expressions ('several', 'many', 'a few', 'massive') are NOT converted
  to numbers and are flagged for human review.
- 'Over a decade' is parsed with value=10.0, unit='years', operator='>'.
- Exact numbers ('five years') are parsed with operator='=='.
- Does not independently determine compliance.
"""

import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class NormalizedNumber:
    """Represents a normalized numeric metric extracted from text."""
    raw_text: str                         # Verbatim substring from original text
    value: float                          # Canonical numeric value
    unit: Optional[str]                   # Canonical unit (e.g., 'Gbps', 'years', 'months')
    operator: str = "=="                  # '>=', '>', '<=', '<', '=='
    parameter_hint: str = "generic"       # Parameter type hint for downstream matching
    start_char: int = -1                  # Start offset in original text
    end_char: int = -1                    # End offset in original text
    is_exact: bool = True                 # True if precise number, False if bounded ('over', 'at least')
    confidence: float = 1.0               # Extraction confidence

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NumberWordNormalizer:
    """Deterministic, zero-dependency English number-word and duration normalizer."""

    # Cardinal number mapping
    SMALL_NUMBERS: Dict[str, int] = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    }

    TENS: Dict[str, int] = {
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
        "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    }

    SCALES: Dict[str, int] = {
        "hundred": 100,
        "thousand": 1_000,
        "lakh": 100_000,
        "lakhs": 100_000,
        "lac": 100_000,
        "lacs": 100_000,
        "crore": 10_000_000,
        "crores": 10_000_000,
        "cr": 10_000_000,
        "million": 1_000_000,
        "billion": 1_000_000_000,
    }

    FRACTIONS: Dict[str, float] = {
        "half": 0.5,
        "quarter": 0.25,
    }

    # Ambiguous phrases that MUST NOT be converted to definite numbers
    AMBIGUOUS_PATTERNS = [
        re.compile(r"\b(several|many|multiple|various|numerous|a\s+few|some|massive|extensive|substantial|significant)\b", re.IGNORECASE),
        re.compile(r"\b(collective\s+domain\s+knowledge|decades\s+of\s+collective)\b", re.IGNORECASE),
    ]

    # Comparison operator prefix patterns
    OPERATOR_PATTERNS: List[Tuple[re.Pattern, str]] = [
        (re.compile(r"\b(at\s+least|minimum\s+of|no\s+less\s+than|not\s+less\s+than)\b", re.IGNORECASE), ">="),
        (re.compile(r"\b(over|more\s+than|greater\s+than|exceeding|in\s+excess\s+of|strictly\s+more\s+than)\b", re.IGNORECASE), ">"),
        (re.compile(r"\b(at\s+most|maximum\s+of|no\s+more\s+than|not\s+more\s+than|up\s+to)\b", re.IGNORECASE), "<="),
        (re.compile(r"\b(less\s+than|under|below|strictly\s+less\s+than)\b", re.IGNORECASE), "<"),
        (re.compile(r"\b(exactly|precisely|strictly)\b", re.IGNORECASE), "=="),
    ]

    # Units patterns and their canonical names
    UNIT_MAPPINGS: List[Tuple[re.Pattern, str, str]] = [
        # (Pattern, Canonical Unit, Parameter Hint)
        (re.compile(r"\b(?:gigabits?\s+per\s+second|gbps|gb/s)\b", re.IGNORECASE), "Gbps", "data_throughput"),
        (re.compile(r"\b(?:megabits?\s+per\s+second|mbps|mb/s)\b", re.IGNORECASE), "Mbps", "data_throughput"),
        (re.compile(r"\b(?:kilobits?\s+per\s+second|kbps|kb/s)\b", re.IGNORECASE), "kbps", "data_throughput"),
        (re.compile(r"\b(?:years?|yrs?)\b", re.IGNORECASE), "years", "duration"),
        (re.compile(r"\b(?:months?|mos?)\b", re.IGNORECASE), "months", "duration"),
        (re.compile(r"\b(?:days?)\b", re.IGNORECASE), "days", "duration"),
        (re.compile(r"\b(?:hours?|hrs?)\b", re.IGNORECASE), "hours", "duration"),
        (re.compile(r"\b(?:minutes?|mins?)\b", re.IGNORECASE), "minutes", "duration"),
        (re.compile(r"\b(?:percent|percentage|%)\b", re.IGNORECASE), "%", "percentage"),
        (re.compile(r"\b(?:usd\s+millions?|million\s+usd|\$\s*millions?|usd\s*m)\b", re.IGNORECASE), "USD Million", "annual_turnover"),
        (re.compile(r"\b(?:usd\s+billions?|billion\s+usd|\$\s*billions?|usd\s*b)\b", re.IGNORECASE), "USD Billion", "annual_turnover"),
        (re.compile(r"\b(?:usd|\$|dollars?)\b", re.IGNORECASE), "USD", "financial"),
        (re.compile(r"\b(?:inr\s+crores?|crores?\s+inr|crores?\s+rupees?|rs\.?\s*crores?|₹\s*crores?|crores?|cr\.?|crs?)\b", re.IGNORECASE), "INR Crore", "annual_turnover"),
        (re.compile(r"\b(?:inr\s+lakhs?|lakhs?\s+inr|lakhs?\s+rupees?|rs\.?\s*lakhs?|₹\s*lakhs?|lakhs?|lacs?|lac)\b", re.IGNORECASE), "INR Lakh", "annual_turnover"),
        (re.compile(r"\b(?:inr\s+thousands?|thousands?\s+inr|thousands?\s+rupees?|rs\.?\s*thousands?|₹\s*thousands?)\b", re.IGNORECASE), "INR Thousand", "financial"),
        (re.compile(r"\b(?:inr|rupees?|rupee|rs\.?|₹)\b", re.IGNORECASE), "INR", "financial"),
        (re.compile(r"\b(?:engineers?|personnel|staff|employees?|members?|people)\b", re.IGNORECASE), "count", "headcount"),
        (re.compile(r"\b(?:projects?|contracts?|engagements?|implementations?|deployments?)\b", re.IGNORECASE), "count", "project_count"),
        (re.compile(r"\b(?:characters?|digits?)\b", re.IGNORECASE), "count", "format_length"),
    ]

    def parse_text_number(self, text: str) -> Optional[float]:
        """Convenience helper to parse a string of number words directly into a float."""
        if not text or not text.strip():
            return None
        words = [w.strip() for w in re.split(r"[\s-]+", text.strip()) if w.strip()]
        return self.parse_word_number(words)

    def is_ambiguous_phrase(self, text: str) -> bool:
        """Returns True if the text contains vague qualifiers that prevent numeric parsing."""
        return any(pat.search(text) for pat in self.AMBIGUOUS_PATTERNS)

    def parse_word_number(self, words: List[str]) -> Optional[float]:
        """Parses a list of English words into a float or integer value.

        Handles:
        - Single words: 'five' -> 5, 'twenty' -> 20, 'million' -> 1000000
        - Compound numbers: 'twenty five' -> 25, 'one hundred and fifty' -> 150
        - Fractions / Multipliers: 'half' -> 0.5, 'quarter' -> 0.25
        """
        if not words:
            return None

        # Filter out grammatical connectors like 'and', 'a', 'an'
        filtered = [w.lower().strip() for w in words if w.lower().strip() not in {"and", "a", "an"}]
        if not filtered:
            return None

        # Check for standalone fraction
        if len(filtered) == 1 and filtered[0] in self.FRACTIONS:
            return self.FRACTIONS[filtered[0]]

        # Check for 'half a million', 'half a billion'
        if len(filtered) >= 2 and filtered[0] == "half":
            rest_val = self.parse_word_number(filtered[1:])
            if rest_val is not None:
                return 0.5 * rest_val

        # Check for 'quarter of a million', etc.
        if len(filtered) >= 2 and filtered[0] == "quarter":
            # Strip 'of' if present
            sub = [w for w in filtered[1:] if w != "of"]
            rest_val = self.parse_word_number(sub)
            if rest_val is not None:
                return 0.25 * rest_val

        # Check for 'point' decimal representation e.g. 'one point five', 'two point five'
        if "point" in filtered:
            idx = filtered.index("point")
            int_part = self.parse_word_number(filtered[:idx]) or 0.0
            dec_words = filtered[idx + 1:]
            dec_str = ""
            scale_val = 1.0
            rem_words = []
            for i, w in enumerate(dec_words):
                if w in self.SMALL_NUMBERS:
                    dec_str += str(self.SMALL_NUMBERS[w])
                elif w.isdigit():
                    dec_str += w
                elif w in self.SCALES:
                    scale_val = float(self.SCALES[w])
                    rem_words = dec_words[i + 1:]
                    break
            dec_part = float("." + dec_str) if dec_str else 0.0
            base = (int_part + dec_part) * scale_val
            if rem_words:
                sub_rem = self.parse_word_number(rem_words)
                if sub_rem is not None:
                    base += sub_rem
            return base

        current_val = 0.0
        running_total = 0.0
        saw_valid_number = False

        for word in filtered:
            # Handle hyphenated words, e.g. 'twenty-five'
            if "-" in word:
                parts = word.split("-")
                sub_val = self.parse_word_number(parts)
                if sub_val is not None:
                    current_val += sub_val
                    saw_valid_number = True
                continue

            if word in self.SMALL_NUMBERS:
                current_val += self.SMALL_NUMBERS[word]
                saw_valid_number = True
            elif word in self.TENS:
                current_val += self.TENS[word]
                saw_valid_number = True
            elif word == "hundred":
                if current_val == 0.0:
                    current_val = 1.0
                current_val *= 100
                saw_valid_number = True
            elif word in {"thousand", "lakh", "lakhs", "lac", "lacs", "crore", "crores", "cr", "million", "billion"}:
                scale = self.SCALES[word]
                if current_val == 0.0:
                    current_val = 1.0
                running_total += current_val * scale
                current_val = 0.0
                saw_valid_number = True
            elif re.match(r"^\d+(?:\.\d+)?$", word):
                current_val += float(word)
                saw_valid_number = True
            else:
                # Word is not recognized as a number
                return None

        total = running_total + current_val
        return total if saw_valid_number else None

    def extract_normalized_numbers(self, text: str) -> List[NormalizedNumber]:
        """Extracts all recognized number-word and duration expressions from text.

        Preserves original character offsets, operators, and units without
        modifying the source text.
        """
        if not text or not text.strip():
            return []

        results: List[NormalizedNumber] = []
        seen_spans = set()

        def _add_result(norm: NormalizedNumber):
            span_key = (norm.start_char, norm.end_char)
            if span_key not in seen_spans:
                # Check for overlap with existing spans
                overlap = False
                for s, e in seen_spans:
                    if not (norm.end_char <= s or norm.start_char >= e):
                        overlap = True
                        break
                if not overlap:
                    seen_spans.add(span_key)
                    results.append(norm)

        # -------------------------------------------------------------------
        # 1. SPECIAL DURATION IDIOMS (decade, half a year, etc.)
        # -------------------------------------------------------------------
        # 1a. Decade idioms: 'over a decade', 'more than a decade', 'a decade', 'two decades'
        decade_pattern = re.compile(
            r"\b(?P<mod>over|more\s+than|at\s+least|exceeding|nearly|almost)?\s*"
            r"(?P<quant>a|an|one|two|three|four|five|\d+)?\s*decades?\b",
            re.IGNORECASE
        )
        for m in decade_pattern.finditer(text):
            raw_match = m.group(0).strip()
            # Guard against "decades of collective" (ambiguous)
            after_ctx = text[m.end(): min(len(text), m.end() + 25)].lower()
            if "of collective" in after_ctx or "of experience" in raw_match.lower() and not m.group("quant") and not m.group("mod"):
                if "decades" in raw_match.lower() and not m.group("quant"):
                    continue

            mod = (m.group("mod") or "").strip().lower()
            quant = (m.group("quant") or "one").strip().lower()

            multiplier = 1.0
            if quant.isdigit():
                multiplier = float(quant)
            elif quant in self.SMALL_NUMBERS:
                multiplier = float(self.SMALL_NUMBERS[quant])
            elif quant in {"a", "an", "one"}:
                multiplier = 1.0

            val = multiplier * 10.0
            operator = "=="
            is_exact = True

            if mod in {"over", "more than", "exceeding"}:
                operator = ">"
                is_exact = False
            elif mod in {"at least"}:
                operator = ">="
                is_exact = False
            elif mod in {"nearly", "almost"}:
                operator = "<="
                is_exact = False

            hint = "duration"
            lower_full = text.lower()
            if any(w in lower_full for w in ["experience", "operation", "operational", "operating", "incorporated", "engineering"]):
                hint = "corporate_experience"

            _add_result(NormalizedNumber(
                raw_text=raw_match,
                value=val,
                unit="years",
                operator=operator,
                parameter_hint=hint,
                start_char=m.start(),
                end_char=m.end(),
                is_exact=is_exact,
                confidence=0.95
            ))

        # 1b. Fractional year/hour idioms: 'half a year', 'half an hour', 'quarter of a year'
        half_year_pat = re.compile(
            r"\b(?P<mod>at\s+least|over|more\s+than)?\s*(?:half\s+a\s+year|six\s+months)\b",
            re.IGNORECASE
        )
        for m in half_year_pat.finditer(text):
            raw = m.group(0).strip()
            mod = (m.group("mod") or "").strip().lower()
            op = ">" if mod in {"over", "more than"} else (">=" if mod == "at least" else "==")
            _add_result(NormalizedNumber(
                raw_text=raw,
                value=6.0,
                unit="months",
                operator=op,
                parameter_hint="duration",
                start_char=m.start(),
                end_char=m.end(),
                is_exact=(op == "=="),
                confidence=0.95
            ))

        # -------------------------------------------------------------------
        # 2. WORD-NUMBERS BOUND TO UNITS
        # e.g., 'one gigabit per second', 'five years', 'half a million USD'
        # -------------------------------------------------------------------
        # Pattern to capture: [Optional Operator] + [Number Words] + [Unit]
        number_tokens = (
            r"(?:half\s+a|quarter\s+of\s+a|one|two|three|four|five|six|seven|eight|nine|ten|"
            r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
            r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
            r"hundred|thousand|lakh|lakhs|lac|lacs|crore|crores|cr|million|billion|"
            r"\d+(?:\.\d+)?|"
            r"point\s+\w+|"
            r"and|\s|-)+"
        )

        modifier_prefix = (
            r"(?P<op_prefix>at\s+least|minimum\s+of|no\s+less\s+than|not\s+less\s+than|"
            r"over|more\s+than|greater\s+than|exceeding|in\s+excess\s+of|"
            r"at\s+most|maximum\s+of|no\s+more\s+than|not\s+more\s+than|up\s+to|"
            r"less\s+than|under|below|"
            r"exactly|precisely|strictly)?"
        )

        for unit_pat, canon_unit, param_hint in self.UNIT_MAPPINGS:
            # Build regex matching optional operator + number words + unit
            combined_re = re.compile(
                rf"\b{modifier_prefix}\s*(?P<num_words>{number_tokens})\s*(?P<unit_str>{unit_pat.pattern})\b",
                re.IGNORECASE
            )

            for m in combined_re.finditer(text):
                full_raw = m.group(0).strip()
                op_raw = (m.group("op_prefix") or "").strip()
                num_raw = m.group("num_words").strip()
                unit_raw = m.group("unit_str").strip()

                # Clean leading 'and' from num_raw if present
                clean_num_raw = re.sub(r"^(?:and\s+)+", "", num_raw, flags=re.IGNORECASE).strip()
                if clean_num_raw != num_raw:
                    # Adjust raw text
                    if op_raw:
                        full_raw = f"{op_raw} {clean_num_raw} {unit_raw}"
                    else:
                        full_raw = f"{clean_num_raw} {unit_raw}"
                    num_raw = clean_num_raw

                # Determine operator
                operator = "=="
                is_exact = True
                if op_raw:
                    for pat, op_sym in self.OPERATOR_PATTERNS:
                        if pat.search(op_raw):
                            operator = op_sym
                            is_exact = (op_sym == "==")
                            break

                # Parse the numeric words
                words = [w.strip() for w in re.split(r"[\s-]+", num_raw) if w.strip()]
                parsed_val = self.parse_word_number(words)
                if parsed_val is None:
                    continue

                # Adjust for currency magnitude if unit already has 'million'
                # e.g., 'five million' + 'USD' -> if parsed_val is 5,000,000 and canon_unit is 'USD Million'
                final_val = parsed_val
                final_unit = canon_unit
                if canon_unit == "USD Million":
                    if final_val >= 1_000_000:
                        final_val = final_val / 1_000_000.0
                elif canon_unit == "USD Billion":
                    if final_val >= 1_000_000_000:
                        final_val = final_val / 1_000_000_000.0
                    elif final_val >= 1_000_000:
                        final_val = final_val / 1_000.0
                elif canon_unit == "INR Crore":
                    if final_val >= 10_000_000:
                        final_val = final_val / 10_000_000.0
                elif canon_unit == "INR Lakh":
                    if final_val >= 100_000:
                        final_val = final_val / 100_000.0
                elif canon_unit == "INR Thousand":
                    if final_val >= 1_000:
                        final_val = final_val / 1_000.0

                # Specific parameter hints based on surrounding context
                local_hint = param_hint
                lower_text = text.lower()
                if final_unit in {"years", "months"}:
                    if any(w in lower_text for w in ["experience", "operation", "operational", "operating", "incorporated", "engineering", "commercial"]):
                        local_hint = "corporate_experience"
                    elif "warranty" in lower_text:
                        local_hint = "warranty_period"
                    elif any(w in lower_text for w in ["response", "initial"]):
                        local_hint = "incident_response_time"
                elif final_unit in {"Gbps", "Mbps"}:
                    local_hint = "data_throughput"
                elif final_unit in {"INR Crore", "INR Lakh", "INR Thousand", "INR", "USD Million", "USD"}:
                    if "turnover" in lower_text or "revenue" in lower_text:
                        local_hint = "annual_turnover"
                    else:
                        local_hint = "financial_capability"

                _add_result(NormalizedNumber(
                    raw_text=full_raw,
                    value=final_val,
                    unit=final_unit,
                    operator=operator,
                    parameter_hint=local_hint,
                    start_char=m.start(),
                    end_char=m.end(),
                    is_exact=is_exact,
                    confidence=0.92
                ))

        # -------------------------------------------------------------------
        # 3. STANDALONE SCALE / COMPOUND / OPERATOR-BOUND NUMBERS
        # e.g., 'two hundred', 'one thousand', 'five million', 'one billion',
        #       'one hundred and fifty', 'at least five', 'more than ten'
        # -------------------------------------------------------------------
        stripped = text.strip()
        stripped_lower = stripped.lower()
        if stripped_lower in self.SMALL_NUMBERS:
            _add_result(NormalizedNumber(
                raw_text=stripped,
                value=float(self.SMALL_NUMBERS[stripped_lower]),
                unit=None,
                operator="==",
                parameter_hint="numeric_threshold",
                start_char=text.find(stripped),
                end_char=text.find(stripped) + len(stripped),
                is_exact=True,
                confidence=0.95
            ))
        elif stripped_lower in self.TENS:
            _add_result(NormalizedNumber(
                raw_text=stripped,
                value=float(self.TENS[stripped_lower]),
                unit=None,
                operator="==",
                parameter_hint="numeric_threshold",
                start_char=text.find(stripped),
                end_char=text.find(stripped) + len(stripped),
                is_exact=True,
                confidence=0.95
            ))

        standalone_re = re.compile(
            rf"\b{modifier_prefix}\s*(?P<num_words>{number_tokens})\b",
            re.IGNORECASE
        )
        for m in standalone_re.finditer(text):
            full_raw = m.group(0).strip()
            op_raw = (m.group("op_prefix") or "").strip()
            num_raw = m.group("num_words").strip()

            clean_num_raw = re.sub(r"^(?:and\s+)+", "", num_raw, flags=re.IGNORECASE).strip()
            if not clean_num_raw:
                continue

            words = [w.strip() for w in re.split(r"[\s-]+", clean_num_raw) if w.strip()]
            if not words:
                continue

            has_scale = any(w.lower() in self.SCALES for w in words)
            has_operator = bool(op_raw)
            is_compound = len(words) >= 2 and any(w.lower() in self.TENS for w in words)
            is_full_text = clean_num_raw.lower() == text.strip().lower()

            if not (has_scale or has_operator or is_compound or is_full_text):
                continue

            surrounding = text[max(0, m.start() - 20): min(len(text), m.end() + 20)]
            if self.is_ambiguous_phrase(surrounding):
                continue

            parsed_val = self.parse_word_number(words)
            if parsed_val is None:
                continue

            operator = "=="
            is_exact = True
            if op_raw:
                for pat, op_sym in self.OPERATOR_PATTERNS:
                    if pat.search(op_raw):
                        operator = op_sym
                        is_exact = (op_sym == "==")
                        break

            _add_result(NormalizedNumber(
                raw_text=full_raw,
                value=parsed_val,
                unit=None,
                operator=operator,
                parameter_hint="numeric_threshold",
                start_char=m.start(),
                end_char=m.end(),
                is_exact=is_exact,
                confidence=0.90
            ))

        # Sort by start_char
        results.sort(key=lambda x: x.start_char)
        return results
