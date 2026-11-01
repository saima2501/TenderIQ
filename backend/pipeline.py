"""
pipeline.py - End-to-End Baseline Compliance Analysis Pipeline
Part of Phase 4: Baseline Demo Pipeline

Connects:
1. PDF Document Text Extraction (Phase 2)
2. Structured Tender Requirement Extraction (Phase 3)
3. Structured Bidder Evidence & Entity Extraction (Phase 3)
4. TF-IDF Top-3 Candidate Evidence Retrieval (Phase 4)
5. Deterministic Numeric & Unit Comparison (Phase 4)
6. Transparent Rule-Based Baseline Classification (Phase 4)
7. SQLite Database Persistence (Phase 4)
8. Phase 7: Cross-Document Evidence Reconciliation
"""

import os
from typing import Optional, Dict, Any, List

from backend.extraction.pdf_extractor import PDFExtractor
from backend.extraction_nlp.requirement_extractor import RequirementExtractor
from backend.extraction_nlp.evidence_extractor import EvidenceExtractor
from backend.matching.tfidf_retriever import TFIDFRetriever
from backend.matching.semantic_retriever import SemanticRetriever
from backend.classification.baseline_classifier import BaselineClassifier, DocumentComplianceReport
from backend.database.db_manager import DatabaseManager

# Deterministic package mapping: associates bidders with their certified annexures
DEFAULT_BIDDER_ANNEXURES: Dict[str, List[str]] = {
    "BIDDER-ALPHA-TECH-PROPOSAL.pdf": ["ANNEXURE-FINANCIAL-STATEMENTS.pdf"],
}


class CompliancePipeline:
    """End-to-end pipeline orchestrator for tender compliance analysis.
    
    Supports both TF-IDF keyword baseline retrieval and Sentence-BERT semantic
    embedding retrieval (all-MiniLM-L6-v2) with automatic fallback.
    """

    def __init__(
        self,
        strip_simulation_banners: bool = True,
        min_retrieval_threshold: float = 0.08,
        min_semantic_threshold: float = 0.25,
        default_retrieval_method: str = "semantic",
        semantic_model_name: str = "all-MiniLM-L6-v2",
        enable_nli: bool = True,
        nli_model_name: str = "cross-encoder/nli-distilroberta-base",
        enable_reconciliation: bool = True,
        db_path: Optional[str] = None
    ):
        self.strip_simulation_banners = strip_simulation_banners
        self.default_retrieval_method = default_retrieval_method
        self.enable_nli = enable_nli
        self.nli_model_name = nli_model_name
        self.enable_reconciliation = enable_reconciliation
        self.pdf_extractor = PDFExtractor(strip_simulation_banners=strip_simulation_banners)
        self.requirement_extractor = RequirementExtractor()
        self.evidence_extractor = EvidenceExtractor()
        
        # Dual retrievers: TF-IDF baseline and Sentence-BERT semantic
        self.tfidf_retriever = TFIDFRetriever(min_similarity_threshold=min_retrieval_threshold)
        self.semantic_retriever = SemanticRetriever(
            model_name=semantic_model_name,
            min_similarity_threshold=min_semantic_threshold,
            fallback_to_tfidf=True
        )
        # Default retriever pointer for backward compatibility
        self.retriever = (
            self.semantic_retriever
            if "semantic" in default_retrieval_method.lower()
            else self.tfidf_retriever
        )
        self.classifier = BaselineClassifier(
            min_retrieval_threshold=min_retrieval_threshold,
            min_semantic_threshold=min_semantic_threshold,
            enable_nli=enable_nli,
            nli_model_name=nli_model_name,
            enable_reconciliation=enable_reconciliation
        )
        self.db = DatabaseManager(db_path=db_path) if db_path else DatabaseManager()

    def run(
        self,
        tender_pdf_path: str,
        bidder_pdf_path: str,
        annexure_pdf_paths: Optional[List[str]] = None,
        retrieval_method: Optional[str] = None,
        enable_nli: Optional[bool] = None,
        enable_reconciliation: Optional[bool] = None,
        tender_submission_date: Optional[Any] = None,
        save_to_db: bool = True
    ) -> DocumentComplianceReport:
        """Executes full end-to-end extraction, matching, and multi-document classification.
        
        Args:
            tender_pdf_path: Path to Tender RFP PDF.
            bidder_pdf_path: Path to Bidder Proposal PDF.
            annexure_pdf_paths: Optional list of paths to secondary bidder annexures
                                (e.g. audited balance sheet, staffing schedule).
                                If None, auto-associates known annexures from package mapping.
                                If empty list [], analyzes proposal only (backward compatible).
            retrieval_method: Optional override: 'semantic' (Sentence-BERT) or 'tfidf'.
            enable_nli: Optional override to enable or disable NLI classification.
            enable_reconciliation: Optional override for cross-document reconciliation.
            save_to_db: Whether to persist analysis to SQLite.
        """
        # Save previous classifier NLI and reconciliation state if temporary override requested
        orig_nli_state = self.classifier.enable_nli
        orig_reconciliation_state = getattr(self.classifier, "enable_reconciliation", True)
        if enable_nli is not None:
            self.classifier.enable_nli = enable_nli
        if enable_reconciliation is not None:
            self.classifier.enable_reconciliation = enable_reconciliation

        try:
            tender_abs = os.path.abspath(tender_pdf_path)
            bidder_abs = os.path.abspath(bidder_pdf_path)

            tender_filename = os.path.basename(tender_abs)
            bidder_filename = os.path.basename(bidder_abs)

            # Determine active annexure PDF paths
            active_annexure_paths: List[str] = []
            if annexure_pdf_paths is None:
                default_annexes = DEFAULT_BIDDER_ANNEXURES.get(bidder_filename, [])
                bidder_dir = os.path.dirname(bidder_abs)
                for a_name in default_annexes:
                    candidate_path = os.path.join(bidder_dir, a_name)
                    if os.path.exists(candidate_path):
                        active_annexure_paths.append(candidate_path)
            else:
                active_annexure_paths = [os.path.abspath(p) for p in annexure_pdf_paths if os.path.exists(p)]

            # Cross-bidder isolation check: ensure annexure is not misallocated from another bidder
            for a_path in active_annexure_paths:
                a_name = os.path.basename(a_path)
                for other_bidder, allowed_annexes in DEFAULT_BIDDER_ANNEXURES.items():
                    if other_bidder != bidder_filename and a_name in allowed_annexes:
                        raise ValueError(
                            f"Cross-bidder isolation error: Annexure '{a_name}' is associated with "
                            f"'{other_bidder}', not '{bidder_filename}'. Cannot mix documents across different bidders."
                        )

            # 1. Extract Tender Requirements
            tender_doc = self.pdf_extractor.extract(tender_abs)
            req_report = self.requirement_extractor.extract_from_extracted_document(tender_doc)
            requirements = req_report.requirements

            # 2. Extract Main Bidder Proposal Evidence
            bidder_doc = self.pdf_extractor.extract(bidder_abs)
            evd_report = self.evidence_extractor.extract_from_extracted_document(bidder_doc)
            evidence_records = list(evd_report.evidence_records)
            bidder_name = evd_report.bidder_name

            # 3. Extract Secondary Annexure Evidence (Multi-Document Package)
            annexure_filenames: List[str] = []
            for a_path in active_annexure_paths:
                annex_doc = self.pdf_extractor.extract(a_path)
                annex_report = self.evidence_extractor.extract_from_extracted_document(annex_doc)
                evidence_records.extend(annex_report.evidence_records)
                annexure_filenames.append(os.path.basename(a_path))

            # 4. Determine Active Retrieval Engine
            method = (retrieval_method or self.default_retrieval_method).lower()
            if "semantic" in method:
                active_retriever = self.semantic_retriever
            else:
                active_retriever = self.tfidf_retriever

            # 5. Retrieve Top-3 Candidate Evidence Sentences per Requirement across all ingested documents
            retrieval_results = active_retriever.retrieve_top_k(
                requirements=requirements,
                evidence_records=evidence_records,
                top_k=3
            )

            # 6. Classify Compliance Posture & Reconcile Cross-Document Conflicts
            report = self.classifier.evaluate_document(
                requirements=requirements,
                retrieval_results=retrieval_results,
                tender_filename=tender_filename,
                bidder_filename=bidder_filename,
                bidder_name=bidder_name,
                annexure_documents=annexure_filenames,
                tender_submission_date=tender_submission_date
            )

            # 7. Save Analysis & Results to SQLite Database
            if save_to_db:
                analysis_id = self.db.save_analysis(report)
                setattr(report, "analysis_id", analysis_id)

            return report
        finally:
            self.classifier.enable_nli = orig_nli_state
            if hasattr(self.classifier, "enable_reconciliation"):
                self.classifier.enable_reconciliation = orig_reconciliation_state


# Alias for backward compatibility and semantic clarity
ComplianceVerificationPipeline = CompliancePipeline
