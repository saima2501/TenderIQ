"""
numeric_comparator.py - Deterministic Numeric & Unit Compliance Evaluation
Part of Phase 4: Baseline Demo Pipeline

Performs auditable mathematical threshold comparisons between tender requirements
and bidder evidence metrics, including automated unit conversions (e.g. Gbps vs Mbps,
years vs months, hours vs minutes).
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple


@dataclass
class NumericComparisonResult:
    """Result of a deterministic numeric comparison."""
    is_applicable: bool                  # True if both requirement and evidence have numeric metrics
    status: str                          # 'SATISFIED', 'VIOLATED', 'INCOMPATIBLE_UNITS', 'NO_NUMERIC_EVIDENCE'
    required_raw: Optional[float] = None # Unconverted requirement threshold
    required_unit: Optional[str] = None
    offered_raw: Optional[float] = None  # Unconverted offered metric
    offered_unit: Optional[str] = None
    required_converted: Optional[float] = None # Converted to base unit
    offered_converted: Optional[float] = None  # Converted to base unit
    base_unit: Optional[str] = None
    operator: str = ">="
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NumericComparator:
    """Deterministic, transparent numeric comparator with unit normalization."""

    # Unit Normalization Tables: Maps variations to (canonical_unit, conversion_multiplier)
    UNIT_FACTORS = {
        # Bandwidth / Data Rate (Canonical base unit: Mbps)
        "gbps": ("mbps", 1000.0),
        "mbps": ("mbps", 1.0),
        "mb/s": ("mb/s", 1.0),
        "gb/s": ("mb/s", 1000.0),

        # Duration / Time (Canonical base unit: months or minutes)
        "years": ("months", 12.0),
        "year": ("months", 12.0),
        "months": ("months", 1.0),
        "month": ("months", 1.0),
        "hours": ("minutes", 60.0),
        "hour": ("minutes", 60.0),
        "minutes": ("minutes", 1.0),
        "minute": ("minutes", 1.0),
        "mins": ("minutes", 1.0),

        # Percentages (Canonical base unit: %)
        "%": ("%", 1.0),
        "percent": ("%", 1.0),

        # Currency / Financials - USD (Canonical base unit: USD Million)
        "usd million": ("usd million", 1.0),
        "usd m": ("usd million", 1.0),
        "million usd": ("usd million", 1.0),
        "usd billion": ("usd million", 1000.0),
        "billion usd": ("usd million", 1000.0),
        "million": ("usd million", 1.0),
        "usd": ("usd", 1.0),

        # Currency / Financials - Indian INR (Canonical base unit: INR)
        "inr crore": ("inr", 10000000.0),
        "inr crores": ("inr", 10000000.0),
        "crore inr": ("inr", 10000000.0),
        "crores inr": ("inr", 10000000.0),
        "crore rupees": ("inr", 10000000.0),
        "crores rupees": ("inr", 10000000.0),
        "rs crore": ("inr", 10000000.0),
        "rs. crore": ("inr", 10000000.0),
        "rs crores": ("inr", 10000000.0),
        "₹ crore": ("inr", 10000000.0),
        "₹ crores": ("inr", 10000000.0),
        "inr cr": ("inr", 10000000.0),
        "rs cr": ("inr", 10000000.0),
        "rs. cr": ("inr", 10000000.0),
        "₹ cr": ("inr", 10000000.0),
        "crore": ("inr", 10000000.0),
        "crores": ("inr", 10000000.0),
        "cr": ("inr", 10000000.0),
        "cr.": ("inr", 10000000.0),

        "inr lakh": ("inr", 100000.0),
        "inr lakhs": ("inr", 100000.0),
        "lakh inr": ("inr", 100000.0),
        "lakhs inr": ("inr", 100000.0),
        "lakh rupees": ("inr", 100000.0),
        "lakhs rupees": ("inr", 100000.0),
        "rs lakh": ("inr", 100000.0),
        "rs. lakh": ("inr", 100000.0),
        "rs lakhs": ("inr", 100000.0),
        "₹ lakh": ("inr", 100000.0),
        "₹ lakhs": ("inr", 100000.0),
        "inr lac": ("inr", 100000.0),
        "inr lacs": ("inr", 100000.0),
        "lac": ("inr", 100000.0),
        "lacs": ("inr", 100000.0),
        "lakh": ("inr", 100000.0),
        "lakhs": ("inr", 100000.0),

        "inr thousand": ("inr", 1000.0),
        "thousand inr": ("inr", 1000.0),
        "rs thousand": ("inr", 1000.0),
        "₹ thousand": ("inr", 1000.0),
        "k inr": ("inr", 1000.0),

        "inr": ("inr", 1.0),
        "rupees": ("inr", 1.0),
        "rupee": ("inr", 1.0),
        "rs": ("inr", 1.0),
        "rs.": ("inr", 1.0),
        "₹": ("inr", 1.0),

        # Counts
        "projects": ("count", 1.0),
        "contracts": ("count", 1.0),
        "engineers": ("count", 1.0),
        "engineer": ("count", 1.0),
        "employees": ("count", 1.0),
        "employee": ("count", 1.0),
        "people": ("count", 1.0),
        "staff": ("count", 1.0),
        "personnel": ("count", 1.0),
        "units": ("units", 1.0),
        "count": ("count", 1.0),
    }

    def normalize_unit(self, unit_str: Optional[str]) -> Tuple[str, float]:
        """Returns the canonical base unit and multiplication factor."""
        if not unit_str:
            return "generic", 1.0
        clean_unit = unit_str.strip().lower()
        if clean_unit in self.UNIT_FACTORS:
            return self.UNIT_FACTORS[clean_unit]
        # Partial matches: evaluate longer/more-specific unit phrases first
        for key in sorted(self.UNIT_FACTORS.keys(), key=len, reverse=True):
            if key in clean_unit or (len(clean_unit) > 2 and clean_unit in key):
                return self.UNIT_FACTORS[key]
        return clean_unit, 1.0

    def compare(
        self,
        required_val: Optional[float],
        required_unit: Optional[str],
        operator: str,
        offered_val: Optional[float],
        offered_unit: Optional[str],
        tolerance: float = 1e-5,
        offered_operator: str = "=="
    ) -> NumericComparisonResult:
        """Evaluates whether offered value satisfies the required threshold."""
        # Check if numeric comparison is possible
        if required_val is None:
            return NumericComparisonResult(
                is_applicable=False,
                status="NO_NUMERIC_REQUIREMENT",
                reasoning="Tender requirement does not define an explicit numeric threshold."
            )

        if offered_val is None:
            return NumericComparisonResult(
                is_applicable=False,
                status="NO_NUMERIC_EVIDENCE",
                required_raw=required_val,
                required_unit=required_unit,
                operator=operator,
                reasoning="No numeric metric was found in the evaluated bidder evidence statement."
            )

        # Normalize units
        req_base_unit, req_factor = self.normalize_unit(required_unit)
        off_base_unit, off_factor = self.normalize_unit(offered_unit)

        # Check unit compatibility
        if req_base_unit != off_base_unit and req_base_unit != "generic" and off_base_unit != "generic":
            # Attempt cross-scale compatibility (e.g. both are time, duration, or speed)
            return NumericComparisonResult(
                is_applicable=True,
                status="INCOMPATIBLE_UNITS",
                required_raw=required_val,
                required_unit=required_unit,
                offered_raw=offered_val,
                offered_unit=offered_unit,
                operator=operator,
                reasoning=f"Unit mismatch: required '{required_unit}' (base: {req_base_unit}) cannot be converted to offered '{offered_unit}' (base: {off_base_unit})."
            )

        # Convert values to common base
        conv_req = float(required_val) * req_factor
        conv_off = float(offered_val) * off_factor
        base_unit = req_base_unit if req_base_unit != "generic" else off_base_unit

        clean_op = operator.strip()
        clean_off_op = (offered_operator or "==").strip()

        # Check evidence-side bounds when offered_operator != "=="
        if clean_off_op != "==":
            # 1. Requirement is a minimum (>= or >)
            if clean_op in (">=", ">"):
                if clean_off_op in (">=", ">"):
                    # Bidder claims a lower bound
                    if conv_off > conv_req + tolerance:
                        is_satisfied = True
                    elif abs(conv_off - conv_req) <= tolerance:
                        if clean_op == ">=":
                            is_satisfied = True
                        elif clean_op == ">" and clean_off_op == ">":
                            is_satisfied = True
                        else:
                            return NumericComparisonResult(
                                is_applicable=True,
                                status="UNCERTAIN",
                                required_raw=required_val,
                                required_unit=required_unit,
                                offered_raw=offered_val,
                                offered_unit=offered_unit,
                                required_converted=conv_req,
                                offered_converted=conv_off,
                                base_unit=base_unit,
                                operator=operator,
                                reasoning=(
                                    f"Bidder evidence claims at least {offered_val} {offered_unit or ''}, "
                                    f"but requirement strictly requires greater than {required_val} {required_unit or ''}. "
                                    f"Bound does not guarantee compliance."
                                )
                            )
                    else:
                        is_satisfied = False
                elif clean_off_op in ("<=", "<"):
                    # Bidder claims an upper bound for a minimum requirement
                    if conv_off < conv_req - tolerance:
                        is_satisfied = False
                    else:
                        return NumericComparisonResult(
                            is_applicable=True,
                            status="UNCERTAIN",
                            required_raw=required_val,
                            required_unit=required_unit,
                            offered_raw=offered_val,
                            offered_unit=offered_unit,
                            required_converted=conv_req,
                            offered_converted=conv_off,
                            base_unit=base_unit,
                            operator=operator,
                            reasoning=(
                                f"Bidder evidence states an upper bound ({clean_off_op} {offered_val} {offered_unit or ''}), "
                                f"which cannot contractually guarantee the required minimum threshold "
                                f"({clean_op} {required_val} {required_unit or ''}). Flagged for review."
                            )
                        )
            # 2. Requirement is an upper bound (<= or <)
            elif clean_op in ("<=", "<"):
                if clean_off_op in ("<=", "<"):
                    if conv_off < conv_req - tolerance:
                        is_satisfied = True
                    elif abs(conv_off - conv_req) <= tolerance:
                        if clean_op == "<=":
                            is_satisfied = True
                        else:
                            is_satisfied = (clean_off_op == "<")
                    else:
                        is_satisfied = False
                elif clean_off_op in (">=", ">"):
                    if conv_off > conv_req + tolerance:
                        is_satisfied = False
                    else:
                        return NumericComparisonResult(
                            is_applicable=True,
                            status="UNCERTAIN",
                            required_raw=required_val,
                            required_unit=required_unit,
                            offered_raw=offered_val,
                            offered_unit=offered_unit,
                            required_converted=conv_req,
                            offered_converted=conv_off,
                            base_unit=base_unit,
                            operator=operator,
                            reasoning=(
                                f"Bidder evidence claims a lower bound ({clean_off_op} {offered_val} {offered_unit or ''}), "
                                f"which cannot contractually guarantee the required maximum ceiling "
                                f"({clean_op} {required_val} {required_unit or ''}). Flagged for review."
                            )
                        )
            elif clean_op == "==":
                return NumericComparisonResult(
                    is_applicable=True,
                    status="UNCERTAIN",
                    required_raw=required_val,
                    required_unit=required_unit,
                    offered_raw=offered_val,
                    offered_unit=offered_unit,
                    required_converted=conv_req,
                    offered_converted=conv_off,
                    base_unit=base_unit,
                    operator=operator,
                    reasoning=(
                        f"Bidder evidence gives a bounded claim ({clean_off_op} {offered_val} {offered_unit or ''}), "
                        f"which cannot satisfy an exact equality requirement (== {required_val} {required_unit or ''})."
                    )
                )
        else:
            # Exact evidence comparison according to operator
            if clean_op == ">=":
                is_satisfied = (conv_off >= conv_req - tolerance)
            elif clean_op == "<=":
                is_satisfied = (conv_off <= conv_req + tolerance)
            elif clean_op == "==":
                is_satisfied = (abs(conv_off - conv_req) <= tolerance)
            elif clean_op == ">":
                is_satisfied = (conv_off > conv_req + tolerance)
            elif clean_op == "<":
                is_satisfied = (conv_off < conv_req - tolerance)
            else:
                is_satisfied = (conv_off >= conv_req - tolerance)

        status = "SATISFIED" if is_satisfied else "VIOLATED"

        # Format human-readable explanation
        if is_satisfied:
            if req_factor != off_factor and req_factor != 1.0:
                reasoning = (
                    f"Offered {offered_val} {offered_unit} converts to {conv_off:.2f} {base_unit}, "
                    f"which satisfies required {required_val} {required_unit} ({conv_req:.2f} {base_unit}) "
                    f"under condition '{clean_op}'."
                )
            else:
                reasoning = (
                    f"Offered {offered_val} {offered_unit or ''} satisfies required threshold "
                    f"of {clean_op} {required_val} {required_unit or ''}."
                )
        else:
            diff = abs(conv_req - conv_off)
            reasoning = (
                f"Offered {offered_val} {offered_unit or ''} ({conv_off:.2f} {base_unit}) "
                f"fails threshold requirement of {clean_op} {required_val} {required_unit or ''} "
                f"({conv_req:.2f} {base_unit}). Discrepancy shortfall: {diff:.2f} {base_unit}."
            )

        return NumericComparisonResult(
            is_applicable=True,
            status=status,
            required_raw=required_val,
            required_unit=required_unit,
            offered_raw=offered_val,
            offered_unit=offered_unit,
            required_converted=round(conv_req, 4),
            offered_converted=round(conv_off, 4),
            base_unit=base_unit,
            operator=clean_op,
            reasoning=reasoning
        )
