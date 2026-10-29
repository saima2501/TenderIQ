"""
reconciliation/ - Phase 7: Multi-Document Evidence Reconciliation

Detects conflicting evidence claims across a bidder's main proposal
and supporting annexures, financial statements, and certificates.
"""
from backend.reconciliation.cross_document_reconciler import (
    CrossDocumentReconciler,
    ConflictingClaimPair,
    ReconciliationResult,
    DocumentEvidenceGroup,
)

__all__ = [
    "CrossDocumentReconciler",
    "ConflictingClaimPair",
    "ReconciliationResult",
    "DocumentEvidenceGroup",
]
