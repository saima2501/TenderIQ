"""
backend.classification package
Part of Phase 4: Baseline Demo Pipeline
"""

from .baseline_classifier import (
    BaselineClassifier,
    ClauseComplianceEvaluation,
    DocumentComplianceReport
)
from .nli_classifier import (
    NLIClassifier,
    NLIPrediction
)

__all__ = [
    "BaselineClassifier",
    "ClauseComplianceEvaluation",
    "DocumentComplianceReport",
    "NLIClassifier",
    "NLIPrediction"
]
