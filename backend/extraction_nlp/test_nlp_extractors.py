"""
test_nlp_extractors.py - Comprehensive Unit & Integration Test Suite for Phase 3
Part of Phase 3: Requirement & Evidence Extraction

Test Coverage:
1. Extraction of structured requirements from 4 synthetic tender PDFs using Phase 2 PDFExtractor
2. Extraction of structured evidence records from 6 synthetic bidder PDFs and financial annexures
3. Extraction of numeric values and units across formats (%, USD Million, Gbps, years, months, minutes, hours, counts)
4. Accurate discrimination between mandatory clauses ('must', 'shall') and optional/preferred clauses ('should', 'preferred')
5. Identification and marking of incomplete statements and vague promotional claims for review (zero hallucination)
6. Exact preservation of source document filenames and 1-indexed page numbers across multi-page documents
7. Explicit detection of negation markers ('not', 'never', 'does not', 'without') in bidder claims
8. Extraction of domain-specific named entities (GSTIN, PAN, ISO standards, SOC 2, AES-256, TLS 1.3, corporate organizations)
9. Graceful handling of empty, whitespace, and edge-case inputs
"""

import os
import sys
import unittest

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.extraction.pdf_extractor import PDFExtractor
from backend.extraction_nlp.requirement_extractor import (
    RequirementExtractor,
    ExtractedRequirement,
    RequirementExtractionReport,
    extract_requirements_from_pdf,
    extract_requirements_from_text,
)
from backend.extraction_nlp.evidence_extractor import (
    EvidenceExtractor,
    ExtractedEvidence,
    EvidenceExtractionReport,
    extract_evidence_from_pdf,
    extract_evidence_from_text,
)


class TestNLPExtractionPipeline(unittest.TestCase):
    """Test suite for Phase 3 NLP requirement and evidence extractors."""

    @classmethod
    def setUpClass(cls):
        cls.pdf_extractor = PDFExtractor(strip_simulation_banners=True)
        cls.req_extractor = RequirementExtractor()
        cls.evd_extractor = EvidenceExtractor()
        cls.tenders_dir = os.path.join(PROJECT_ROOT, "dataset", "synthetic_pdfs", "tenders")
        cls.bidders_dir = os.path.join(PROJECT_ROOT, "dataset", "synthetic_pdfs", "bidders")

    # -------------------------------------------------------------
    # 1. Test Requirement Extraction on Synthetic Tender PDFs
    # -------------------------------------------------------------
    def test_requirement_extraction_from_tender_pdfs(self):
        """Verify requirement extraction across all 4 synthetic tender PDFs."""
        expected_tenders = [
            ("TENDER-2024-IT-001.pdf", "Enterprise Cloud Hosting & Cybersecurity"),
            ("TENDER-2024-MED-002.pdf", "Hospital Information Management System"),
            ("TENDER-2024-SOLAR-003.pdf", "Solar Microgrid & SCADA Automation"),
            ("TENDER-2024-FIN-004.pdf", "Real-Time Transaction Fraud Detection")
        ]

        for filename, desc in expected_tenders:
            pdf_path = os.path.join(self.tenders_dir, filename)
            self.assertTrue(os.path.exists(pdf_path), f"Missing tender PDF: {pdf_path}")

            report = extract_requirements_from_pdf(pdf_path)
            self.assertIsInstance(report, RequirementExtractionReport)
            self.assertEqual(report.source_document, filename)
            self.assertGreaterEqual(report.total_pages, 1)
            self.assertGreater(report.total_requirements, 5, f"Too few requirements extracted from {filename}")
            self.assertGreater(report.mandatory_count, 0, f"No mandatory requirements found in {filename}")

            # Verify that every extracted requirement has required Phase 3 fields
            for req in report.requirements:
                self.assertTrue(req.requirement_id, "requirement_id must not be empty")
                self.assertTrue(req.original_text, "original_text must not be empty")
                self.assertTrue(req.category, "category must not be empty")
                self.assertTrue(req.parameter, "parameter must not be empty")
                self.assertIn(req.operator, [">=", "<=", "==", "CONTAINS", "MANDATORY_SUBMISSION", "NONE"])
                self.assertEqual(req.source_document, filename)
                self.assertIsNotNone(req.page_number)
                self.assertGreaterEqual(req.page_number, 1)

    # -------------------------------------------------------------
    # 2. Test Evidence Extraction on Synthetic Bidder PDFs
    # -------------------------------------------------------------
    def test_evidence_extraction_from_bidder_pdfs(self):
        """Verify evidence extraction across synthetic bidder proposals and annexures."""
        expected_bidders = [
            "BIDDER-ALPHA-TECH-PROPOSAL.pdf",
            "ANNEXURE-FINANCIAL-STATEMENTS.pdf",
            "BIDDER-BETA-HEALTH-PROPOSAL.pdf",
            "BIDDER-GAMMA-SOLAR-PROPOSAL.pdf",
            "BIDDER-DELTA-FINAI-PROPOSAL.pdf",
            "BIDDER-EPSILON-INFRA-PROPOSAL.pdf"
        ]

        for filename in expected_bidders:
            pdf_path = os.path.join(self.bidders_dir, filename)
            self.assertTrue(os.path.exists(pdf_path), f"Missing bidder PDF: {pdf_path}")

            report = extract_evidence_from_pdf(pdf_path)
            self.assertIsInstance(report, EvidenceExtractionReport)
            self.assertEqual(report.source_document, filename)
            self.assertGreater(report.total_evidence_records, 2, f"Too few evidence records from {filename}")

            # Verify that every extracted evidence record has required fields
            for evd in report.evidence_records:
                self.assertTrue(evd.evidence_id, "evidence_id must not be empty")
                self.assertTrue(evd.original_sentence, "original_sentence must not be empty")
                self.assertIsInstance(evd.extracted_entities, list)
                self.assertIsInstance(evd.numeric_values, list)
                self.assertEqual(evd.source_document, filename)
                self.assertGreaterEqual(evd.page_number, 1)
                self.assertIsInstance(evd.has_negation, bool)
                self.assertIsInstance(evd.uncertainty_flag, bool)

    # -------------------------------------------------------------
    # 3. Test Numeric Values and Units Extraction
    # -------------------------------------------------------------
    def test_numeric_values_and_units_extraction(self):
        """Verify precision of numeric metrics and unit parsing."""
        test_cases = [
            {
                "clause": "The hosting environment must guarantee a minimum monthly uptime SLA of 99.95%.",
                "expected_val": 99.95,
                "expected_unit": "%",
                "expected_op": ">=",
                "expected_param": "monthly uptime SLA"
            },
            {
                "clause": "The bidder must demonstrate an average annual audited turnover of at least USD 10.0 Million.",
                "expected_val": 10.0,
                "expected_unit": "USD Million",
                "expected_op": ">=",
                "expected_param": "audited financial turnover"
            },
            {
                "clause": "The data pipeline must support an ingress throughput capacity of at least 1 Gbps.",
                "expected_val": 1.0,
                "expected_unit": "Gbps",
                "expected_op": ">=",
                "expected_param": "network data throughput capacity"
            },
            {
                "clause": "The contractor must provide a comprehensive warranty of at least 36 months.",
                "expected_val": 36.0,
                "expected_unit": "months",
                "expected_op": ">=",
                "expected_param": "comprehensive warranty duration"
            },
            {
                "clause": "The lead engineer must have at least 8 years of relevant experience.",
                "expected_val": 8.0,
                "expected_unit": "years",
                "expected_op": ">=",
                "expected_param": "continuous commercial operating experience"
            },
            {
                "clause": "Critical incident response inside 15 minutes and on-site fix within 4 hours.",
                "expected_val": 15.0,
                "expected_unit": "minutes",
                "expected_op": "<=",
                "expected_param": "critical incident response time"
            },
            {
                "clause": "The bidder must furnish client completion certificates for at least 3 similar reference projects.",
                "expected_val": 3.0,
                "expected_unit": "similar reference projects",
                "expected_op": ">=",
                "expected_param": "reference project scale"
            },
            {
                "clause": "Packet drop rate must not exceed 5%.",
                "expected_val": 5.0,
                "expected_unit": "%",
                "expected_op": "<=",
                "expected_param": "maximum acceptable error/loss rate"
            },
            {
                "clause": "The contractor must supply exactly 10 units.",
                "expected_val": 10.0,
                "expected_unit": "units",
                "expected_op": "==",
                "expected_param": "quantity of units"
            }
        ]

        for tc in test_cases:
            reqs = extract_requirements_from_text(tc["clause"])
            self.assertEqual(len(reqs), 1, f"Failed to extract requirement from: {tc['clause']}")
            req = reqs[0]
            self.assertEqual(req.value, tc["expected_val"], f"Value mismatch for: {tc['clause']}")
            self.assertEqual(req.unit, tc["expected_unit"], f"Unit mismatch for: {tc['clause']}")
            self.assertEqual(req.operator, tc["expected_op"], f"Operator mismatch for: {tc['clause']}")
            self.assertEqual(req.parameter, tc["expected_param"], f"Parameter mismatch for: {tc['clause']}")

    # -------------------------------------------------------------
    # 4. Test Mandatory versus Optional Clauses
    # -------------------------------------------------------------
    def test_mandatory_versus_optional_clauses(self):
        """Verify modal auxiliary classification: 'must'/'shall' vs 'should'/'preferred'."""
        mandatory_clauses = [
            "The bidder must submit a valid Bank Solvency Certificate.",
            "The contractor shall provide 24/7 dedicated telephone support.",
            "Furnishing a 15-digit GSTIN is mandatory under statutory procurement rules.",
            "Suppliers are required to possess an active ISO 9001 certification."
        ]

        optional_clauses = [
            "Vendors should preferably hold CMMI Level 5 certification.",
            "Prior experience with public sector health institutions is preferred.",
            "The prospective contractor is advised to provide sample integration schematics.",
            "Bidders may optionally submit client letters of commendation."
        ]

        for cl in mandatory_clauses:
            reqs = extract_requirements_from_text(cl)
            self.assertTrue(len(reqs) >= 1, f"Failed on mandatory clause: {cl}")
            self.assertTrue(reqs[0].is_mandatory, f"Clause should be mandatory: {cl}")
            self.assertIn("mandatory", reqs[0].review_notes.lower())

        for cl in optional_clauses:
            reqs = extract_requirements_from_text(cl)
            self.assertTrue(len(reqs) >= 1, f"Failed on optional clause: {cl}")
            self.assertFalse(reqs[0].is_mandatory, f"Clause should be optional: {cl}")
            self.assertIn("optional", reqs[0].review_notes.lower())

    # -------------------------------------------------------------
    # 5. Test Incomplete Statements & Vague Claims (Zero Hallucination)
    # -------------------------------------------------------------
    def test_incomplete_and_ambiguous_statements(self):
        """Verify that incomplete or vague claims are marked for review rather than inventing values."""
        # A. Incomplete tender statement ending with dangling preposition
        incomplete_clause = "The contractor must submit documentation regarding."
        reqs = extract_requirements_from_text(incomplete_clause)
        self.assertEqual(len(reqs), 1)
        self.assertTrue(reqs[0].uncertainty_flag, "Incomplete statement must be flagged for review")
        self.assertIsNone(reqs[0].value, "Must not hallucinate missing value for incomplete statement")
        self.assertIn("incomplete", reqs[0].review_notes.lower())

        # B. Ambiguous modality without deontic modal verbs
        ambiguous_clause = "System performance benchmarks 100 transactions per second."
        reqs2 = extract_requirements_from_text(ambiguous_clause)
        self.assertEqual(len(reqs2), 1)
        self.assertTrue(reqs2[0].uncertainty_flag)
        self.assertIsNone(reqs2[0].is_mandatory)
        self.assertIn("uncertain modality", reqs2[0].review_notes.lower())

        # C. Vague / promotional bidder claim lacking verifiable empirical metric
        vague_evidence = "Our founders and senior engineers hold extensive decades of collective domain knowledge in cloud."
        evds = extract_evidence_from_text(vague_evidence)
        self.assertEqual(len(evds), 1)
        self.assertTrue(evds[0].uncertainty_flag, "Promotional claim must be flagged as uncertain")
        self.assertIn("collective", evds[0].review_notes.lower())

        # D. Credit line in principle without solvency certificate
        vague_solvency = "The company maintains a strong solvent standing with lines of credit approved in principle."
        evds2 = extract_evidence_from_text(vague_solvency)
        self.assertEqual(len(evds2), 1)
        self.assertTrue(evds2[0].uncertainty_flag)
        self.assertIn("in principle", evds2[0].review_notes.lower())

    # -------------------------------------------------------------
    # 6. Test Source Document and 1-Indexed Page Preservation
    # -------------------------------------------------------------
    def test_source_and_page_reference_preservation(self):
        """Verify that source PDF filenames and 1-indexed page numbers are preserved across multi-page PDFs."""
        pdf_path = os.path.join(self.tenders_dir, "TENDER-2024-IT-001.pdf")
        report = extract_requirements_from_pdf(pdf_path)

        self.assertEqual(report.source_document, "TENDER-2024-IT-001.pdf")
        self.assertEqual(report.total_pages, 2)

        # Page 1 contains sections 1 to 4; Page 2 contains sections 5 and 6
        page_1_reqs = [r for r in report.requirements if r.page_number == 1]
        page_2_reqs = [r for r in report.requirements if r.page_number == 2]

        self.assertGreater(len(page_1_reqs), 0, "Page 1 must contain requirements")
        self.assertGreater(len(page_2_reqs), 0, "Page 2 must contain requirements")

        # Verify page numbers are strictly 1-indexed (1, 2)
        for r in report.requirements:
            self.assertIn(r.page_number, [1, 2])
            self.assertEqual(r.source_document, "TENDER-2024-IT-001.pdf")

    # -------------------------------------------------------------
    # 7. Test Negation Detection in Bidder Evidence
    # -------------------------------------------------------------
    def test_negation_detection_in_bidder_evidence(self):
        """Verify detection of negative polarity markers ('not', 'never', 'cannot')."""
        negation_statements = [
            "The production system hosting environment does not guarantee a minimum monthly uptime SLA of 99.95%.",
            "Our standard remote tier does NOT include on-site engineer dispatch within 4 hours.",
            "Affidavit confirms our directors have never been debarred or blacklisted by any government authority.",
            "System operations cannot proceed without dual administrator authorization."
        ]

        positive_statements = [
            "Our enterprise cloud hosting contractually commits to a minimum monthly uptime SLA of 99.99%.",
            "Service uptime availability is guaranteed at exactly 99.90% as measured by synthetic monitoring."
        ]

        for stmt in negation_statements:
            evds = extract_evidence_from_text(stmt)
            self.assertEqual(len(evds), 1)
            self.assertTrue(evds[0].has_negation, f"Negation missed in: {stmt}")

        for stmt in positive_statements:
            evds = extract_evidence_from_text(stmt)
            self.assertEqual(len(evds), 1)
            self.assertFalse(evds[0].has_negation, f"False positive negation in: {stmt}")

    # -------------------------------------------------------------
    # 8. Test Named Entity and Identifier Extraction
    # -------------------------------------------------------------
    def test_named_entity_and_identifier_extraction(self):
        """Verify extraction of GSTIN, PAN, ISO standards, and encryption algorithms."""
        evidence_text = (
            "Alpha Cloud & Cyber Systems Ltd. holds active GSTIN Certificate (GSTIN: 27AAACA1234F1Z5) "
            "and PAN: AAACA1234F. Enclosed is our ISO/IEC 27001:2022 accreditation certificate "
            "implementing AES-256 at rest and TLS 1.3 in transit."
        )

        evds = extract_evidence_from_text(evidence_text)
        self.assertGreaterEqual(len(evds), 1)

        # Aggregate entities across sentences in the paragraph
        all_entities = []
        for evd in evds:
            all_entities.extend(evd.extracted_entities)

        entity_texts = [e["text"] for e in all_entities]
        entity_labels = {e["text"]: e["label"] for e in all_entities}

        self.assertIn("27AAACA1234F1Z5", entity_texts)
        self.assertEqual(entity_labels["27AAACA1234F1Z5"], "GSTIN")

        self.assertIn("AAACA1234F", entity_texts)
        self.assertEqual(entity_labels["AAACA1234F"], "PAN")

        self.assertIn("ISO/IEC 27001:2022", entity_texts)
        self.assertEqual(entity_labels["ISO/IEC 27001:2022"], "STANDARD_CERT")

        self.assertIn("AES-256", entity_texts)
        self.assertEqual(entity_labels["AES-256"], "CRYPTO_STANDARD")

        self.assertIn("TLS 1.3", entity_texts)
        self.assertEqual(entity_labels["TLS 1.3"], "CRYPTO_STANDARD")

    # -------------------------------------------------------------
    # 9. Test Edge Cases & Input Robustness
    # -------------------------------------------------------------
    def test_edge_cases_and_input_robustness(self):
        """Verify extractors handle empty text, whitespace, and noisy input gracefully."""
        self.assertEqual(extract_requirements_from_text(""), [])
        self.assertEqual(extract_requirements_from_text("   \n\t  "), [])
        self.assertEqual(extract_evidence_from_text(""), [])
        self.assertEqual(extract_evidence_from_text("   \n\t  "), [])


if __name__ == "__main__":
    unittest.main()
