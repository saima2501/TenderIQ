"""
extraction_nlp - Natural Language Processing & Information Extraction Pipeline
Part of Phase 3: Requirement & Evidence Extraction

Modules:
- requirement_extractor: Extracts structured tender requirements with modality, operators, and parameters
- evidence_extractor: Extracts structured bidder claims, named entities, numeric metrics, and negation signals
"""

from .requirement_extractor import (
    RequirementExtractor,
    ExtractedRequirement,
    RequirementExtractionReport,
    extract_requirements_from_pdf,
    extract_requirements_from_text,
)

from .evidence_extractor import (
    EvidenceExtractor,
    ExtractedEvidence,
    EvidenceExtractionReport,
    extract_evidence_from_pdf,
    extract_evidence_from_text,
)

__all__ = [
    "RequirementExtractor",
    "ExtractedRequirement",
    "RequirementExtractionReport",
    "extract_requirements_from_pdf",
    "extract_requirements_from_text",
    "EvidenceExtractor",
    "ExtractedEvidence",
    "EvidenceExtractionReport",
    "extract_evidence_from_pdf",
    "extract_evidence_from_text",
]
