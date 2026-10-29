"""
test_extractor.py - Comprehensive Test Suite for Phase 2 PDF Document Extraction
Part of Phase 2: PDF Document Extraction

Test Coverage:
1. Normal synthetic tender PDFs extraction & text preservation
2. Normal synthetic bidder proposal PDFs extraction
3. Empty file handling (0-byte PDF) -> graceful EMPTY_FILE status
4. Corrupted/unreadable file handling (garbage bytes) -> graceful CORRUPTED_FILE status
5. Scanned image-only PDF detection -> flags is_scanned=True, ocr_applied=False, status=SCANNED_NEEDS_OCR
6. Exact 1-indexed page-number preservation
7. PyMuPDF fallback engine consistency
8. TextCleaner whitespace, quote, and de-hyphenation normalization
"""

import os
import sys
import tempfile
import unittest
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.extraction.pdf_extractor import PDFExtractor, extract_pdf
from backend.extraction.text_cleaner import clean_page_text, TextCleaner


class TestPDFExtractionPipeline(unittest.TestCase):
    """Automated unit and integration test suite for PDF extraction."""

    @classmethod
    def setUpClass(cls):
        cls.extractor = PDFExtractor()
        cls.tenders_dir = os.path.join(PROJECT_ROOT, "dataset", "synthetic_pdfs", "tenders")
        cls.bidders_dir = os.path.join(PROJECT_ROOT, "dataset", "synthetic_pdfs", "bidders")

    # -------------------------------------------------------------
    # 1. Test Normal Synthetic Tender PDFs
    # -------------------------------------------------------------
    def test_extract_normal_tender_pdfs(self):
        """Verify extraction on all 4 generated synthetic tender PDFs."""
        expected_tenders = [
            "TENDER-2024-IT-001.pdf",
            "TENDER-2024-MED-002.pdf",
            "TENDER-2024-SOLAR-003.pdf",
            "TENDER-2024-FIN-004.pdf"
        ]
        for t_file in expected_tenders:
            pdf_path = os.path.join(self.tenders_dir, t_file)
            self.assertTrue(os.path.exists(pdf_path), f"Tender PDF missing: {pdf_path}")

            doc = self.extractor.extract(pdf_path)
            self.assertEqual(doc.status, "SUCCESS")
            self.assertFalse(doc.is_scanned)
            self.assertFalse(doc.ocr_applied)
            self.assertGreater(doc.total_pages, 0)
            self.assertGreater(len(doc.pages), 0)

            # Check that each page has text and valid 1-indexed numbering
            for p in doc.pages:
                self.assertGreater(p.page_number, 0)
                self.assertGreater(p.char_count, 100)
                self.assertGreater(p.word_count, 15)

    # -------------------------------------------------------------
    # 2. Test Normal Synthetic Bidder Proposal PDFs
    # -------------------------------------------------------------
    def test_extract_normal_bidder_pdfs(self):
        """Verify extraction on synthetic bidder proposals and annexures."""
        expected_bidders = [
            "BIDDER-ALPHA-TECH-PROPOSAL.pdf",
            "ANNEXURE-FINANCIAL-STATEMENTS.pdf",
            "BIDDER-BETA-HEALTH-PROPOSAL.pdf",
            "BIDDER-GAMMA-SOLAR-PROPOSAL.pdf",
            "BIDDER-DELTA-FINAI-PROPOSAL.pdf",
            "BIDDER-EPSILON-INFRA-PROPOSAL.pdf"
        ]
        for b_file in expected_bidders:
            pdf_path = os.path.join(self.bidders_dir, b_file)
            self.assertTrue(os.path.exists(pdf_path), f"Bidder PDF missing: {pdf_path}")

            doc = self.extractor.extract(pdf_path)
            self.assertEqual(doc.status, "SUCCESS")
            self.assertFalse(doc.is_scanned)
            self.assertGreater(doc.total_pages, 0)

            # Specific text content verification for Alpha Tech
            if b_file == "BIDDER-ALPHA-TECH-PROPOSAL.pdf":
                full_text = doc.get_full_text()
                self.assertIn("Alpha Cloud & Cyber Systems Ltd.", full_text)
                self.assertIn("27AAACA1234F1Z5", full_text)  # GSTIN
                self.assertIn("AAACA1234F", full_text)        # PAN

    # -------------------------------------------------------------
    # 3. Test 1-Indexed Page Number Preservation
    # -------------------------------------------------------------
    def test_page_number_preservation(self):
        """Verify that page numbers are strictly continuous, 1-indexed integers."""
        sample_pdf = os.path.join(self.tenders_dir, "TENDER-2024-IT-001.pdf")
        doc = self.extractor.extract(sample_pdf)

        for expected_page, page_obj in enumerate(doc.pages, start=1):
            self.assertEqual(page_obj.page_number, expected_page,
                             f"Page number mismatch: expected {expected_page}, got {page_obj.page_number}")

    # -------------------------------------------------------------
    # 4. Test Empty File Handling (0 bytes)
    # -------------------------------------------------------------
    def test_empty_file_handling(self):
        """Verify graceful handling when a 0-byte PDF is supplied."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            empty_path = tmp.name

        try:
            doc = self.extractor.extract(empty_path)
            self.assertEqual(doc.status, "EMPTY_FILE")
            self.assertEqual(doc.total_pages, 0)
            self.assertEqual(len(doc.pages), 0)
            self.assertIn("empty", doc.errors[0].lower())
        finally:
            if os.path.exists(empty_path):
                os.remove(empty_path)

    # -------------------------------------------------------------
    # 5. Test Corrupted / Unreadable File Handling
    # -------------------------------------------------------------
    def test_corrupted_file_handling(self):
        """Verify graceful handling when unreadable/garbage bytes are supplied."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(b"%PDF-1.4 GARBAGE INVALID BINARY DATA 1234567890 NOT A VALID TRAILER")
            corrupt_path = tmp.name

        try:
            doc = self.extractor.extract(corrupt_path)
            self.assertEqual(doc.status, "CORRUPTED_FILE")
            self.assertEqual(doc.total_pages, 0)
            self.assertGreater(len(doc.errors), 0)
        finally:
            if os.path.exists(corrupt_path):
                os.remove(corrupt_path)

    # -------------------------------------------------------------
    # 6. Test Scanned Image-Only PDF Detection (No Faked OCR)
    # -------------------------------------------------------------
    def test_scanned_image_only_pdf_detection(self):
        """Verify that image-only PDFs are flagged as SCANNED_NEEDS_OCR without pretending OCR was performed."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            scanned_path = tmp.name

        # Create a genuine image-only PDF using ReportLab
        try:
            c = canvas.Canvas(scanned_path, pagesize=letter)
            # Create a dummy image rectangle to simulate a scanned document page without any text layer
            from reportlab.graphics.shapes import Drawing, Rect
            d = Drawing(letter[0], letter[1])
            d.add(Rect(0, 0, letter[0], letter[1], fillColor=None, strokeColor=None))
            # Draw empty page with an image placeholder
            from PIL import Image
            img = Image.new("RGB", (300, 400), color=(240, 240, 240))
            img_path = scanned_path.replace(".pdf", ".png")
            img.save(img_path)

            c.drawImage(img_path, 50, 50, width=500, height=700)
            c.showPage()
            c.save()

            if os.path.exists(img_path):
                os.remove(img_path)

            doc = self.extractor.extract(scanned_path)
            self.assertEqual(doc.status, "SCANNED_NEEDS_OCR")
            self.assertTrue(doc.is_scanned)
            self.assertFalse(doc.ocr_applied)  # Strict rule: Do not pretend OCR was run
            self.assertIn("Scanned image-only PDF detected", doc.notes)
        finally:
            if os.path.exists(scanned_path):
                os.remove(scanned_path)

    # -------------------------------------------------------------
    # 7. Test Fallback Engine (PyMuPDF) Directly
    # -------------------------------------------------------------
    def test_pymupdf_fallback_direct(self):
        """Verify that the PyMuPDF fallback engine extracts identical page counts and valid text."""
        sample_pdf = os.path.join(self.tenders_dir, "TENDER-2024-IT-001.pdf")
        doc_plumber = self.extractor.extract(sample_pdf)
        doc_pymupdf = self.extractor._extract_with_pymupdf(sample_pdf, "TENDER-2024-IT-001.pdf")

        self.assertEqual(doc_pymupdf.status, "SUCCESS")
        self.assertEqual(doc_plumber.total_pages, doc_pymupdf.total_pages)
        self.assertIn("Enterprise Cloud Hosting", doc_pymupdf.pages[0].cleaned_text)

    # -------------------------------------------------------------
    # 8. Test TextCleaner Normalization
    # -------------------------------------------------------------
    def test_text_cleaner_features(self):
        """Verify unicode quotes, bullets, de-hyphenation, and whitespace normalization."""
        raw_input = (
            "• Section 1: “Enterprise  Cloud”\n"
            "The infra-\n"
            "structure guarantees 99.95%.\r\n\r\n\r\n"
            "• Feature 2: High—availability.\u00a0\u00a0"
        )
        cleaned = clean_page_text(raw_input)

        # Quotes normalized to straight quotes
        self.assertIn('"Enterprise Cloud"', cleaned)
        # Bullet normalized to standard markdown hyphen
        self.assertTrue(cleaned.startswith("- Section 1:"))
        # De-hyphenation fixed across line break
        self.assertIn("infrastructure", cleaned)
        # Excessive newlines compressed
        self.assertNotIn("\n\n\n", cleaned)
        # Non-breaking space removed
        self.assertNotIn("\u00a0", cleaned)


def run_extraction_showcase():
    """Extracts and prints representative text from real synthetic PDFs for demonstration."""
    print("\n" + "=" * 75)
    print("PHASE 2: REPRESENTATIVE PDF EXTRACTION SAMPLES")
    print("=" * 75)

    extractor = PDFExtractor(strip_simulation_banners=False)
    sample_targets = [
        os.path.join(PROJECT_ROOT, "dataset", "synthetic_pdfs", "tenders", "TENDER-2024-IT-001.pdf"),
        os.path.join(PROJECT_ROOT, "dataset", "synthetic_pdfs", "bidders", "BIDDER-ALPHA-TECH-PROPOSAL.pdf"),
        os.path.join(PROJECT_ROOT, "dataset", "synthetic_pdfs", "bidders", "ANNEXURE-FINANCIAL-STATEMENTS.pdf")
    ]

    for pdf_path in sample_targets:
        doc = extractor.extract(pdf_path)
        print(f"\n[FILE]: {doc.file_name}")
        print(f"  * Status:            {doc.status}")
        print(f"  * Engine Used:       {doc.extraction_engine}")
        print(f"  * Total Pages:       {doc.total_pages}")
        print(f"  * Is Scanned:        {doc.is_scanned} (OCR Applied: {doc.ocr_applied})")

        # Show Page 1 snippet
        if doc.pages:
            p1 = doc.pages[0]
            print(f"  * Page {p1.page_number} Stats:     {p1.char_count} chars | {p1.word_count} words | Has Images: {p1.has_images}")
            print(f"  * Page {p1.page_number} Excerpt:")
            lines = p1.cleaned_text.split("\n")
            excerpt = "\n".join(f"      | {line}" for line in lines[:8])
            print(excerpt)
            if len(lines) > 8:
                print(f"      | ... [{len(lines)-8} more lines omitted]")
        print("-" * 75)


def main():
    print("\n" + "=" * 75)
    print("RUNNING PHASE 2 AUTOMATED PDF EXTRACTION TEST SUITE")
    print("=" * 75 + "\n")

    suite = unittest.TestLoader().loadTestsFromTestCase(TestPDFExtractionPipeline)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\nAll 8 automated extraction tests PASSED successfully!")
        run_extraction_showcase()
    else:
        print("\nSome tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
