"""
pdf_extractor.py - Robust PDF Text Extraction Pipeline with Fallback & Scanned Document Detection
Part of Phase 2: PDF Document Extraction

Features:
1. Primary engine: pdfplumber (layout-aware, table-friendly)
2. Fallback engine: PyMuPDF / pymupdf (high speed, handles damaged PDF streams)
3. Preserves exact 1-indexed page numbers
4. Graceful handling of:
   - Normal text PDFs
   - Empty files (0 bytes)
   - Corrupted or unreadable files
   - Scanned image-only PDFs (flags 'is_scanned=True', 'ocr_applied=False' without faking OCR)
   - Password-protected files
5. Integrates with TextCleaner for consistent whitespace and encoding normalization
"""

import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

import pdfplumber

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None

from .text_cleaner import clean_page_text


@dataclass
class ExtractedPage:
    """Represents a single page extracted from a PDF document."""
    page_number: int                 # 1-indexed (Page 1, 2, ...)
    raw_text: str                    # Raw text extracted by the engine
    cleaned_text: str                # Normalized text post-cleaning
    char_count: int                  # Number of characters in cleaned text
    word_count: int                  # Number of words in cleaned text
    has_images: bool = False         # True if page contains raster images
    is_blank: bool = False           # True if no text is present
    engine_used: str = "pdfplumber"  # 'pdfplumber' or 'pymupdf'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractedDocument:
    """Represents the complete extracted result of a PDF file."""
    file_path: str
    file_name: str
    total_pages: int = 0
    extraction_engine: str = "pdfplumber"
    status: str = "SUCCESS"          # SUCCESS, EMPTY_FILE, CORRUPTED_FILE, SCANNED_NEEDS_OCR, PASSWORD_PROTECTED
    is_scanned: bool = False         # True if document consists of scanned images without text layer
    ocr_applied: bool = False        # Explicitly False: We do not fake OCR
    pages: List[ExtractedPage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    notes: str = ""

    def get_full_text(self, use_cleaned: bool = True) -> str:
        """Concatenates all pages with clear page demarcations."""
        parts = []
        for p in self.pages:
            t = p.cleaned_text if use_cleaned else p.raw_text
            parts.append(f"--- [Page {p.page_number}] ---\n{t}")
        return "\n\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["pages"] = [p.to_dict() for p in self.pages]
        return d


class PDFExtractor:
    """Production-grade PDF extractor designed for procurement tender evaluation."""

    def __init__(self, strip_simulation_banners: bool = False, min_text_chars_for_scanned: int = 40):
        self.strip_simulation_banners = strip_simulation_banners
        self.min_text_chars_for_scanned = min_text_chars_for_scanned

    def extract(self, file_path: str) -> ExtractedDocument:
        """Extracts text page by page from a given PDF path with automatic fallback."""
        abs_path = os.path.abspath(file_path)
        file_name = os.path.basename(abs_path)

        # 1. Check File Existence
        if not os.path.exists(abs_path):
            return ExtractedDocument(
                file_path=abs_path,
                file_name=file_name,
                status="FILE_NOT_FOUND",
                errors=[f"File does not exist: {abs_path}"]
            )

        # 2. Check for Empty File (0 bytes)
        file_size = os.path.getsize(abs_path)
        if file_size == 0:
            return ExtractedDocument(
                file_path=abs_path,
                file_name=file_name,
                total_pages=0,
                status="EMPTY_FILE",
                errors=["The PDF file is completely empty (0 bytes)."]
            )

        # 3. Attempt Primary Extraction via pdfplumber
        try:
            return self._extract_with_pdfplumber(abs_path, file_name)
        except Exception as plumber_err:
            # 4. Fallback Extraction via PyMuPDF if pdfplumber fails
            if fitz is not None:
                try:
                    fallback_doc = self._extract_with_pymupdf(abs_path, file_name)
                    fallback_doc.notes = f"Recovered using PyMuPDF fallback after pdfplumber exception: {str(plumber_err)[:120]}"
                    fallback_doc.errors.append(f"pdfplumber error: {str(plumber_err)}")
                    return fallback_doc
                except Exception as fitz_err:
                    return ExtractedDocument(
                        file_path=abs_path,
                        file_name=file_name,
                        status="CORRUPTED_FILE",
                        errors=[
                            f"pdfplumber failed: {str(plumber_err)}",
                            f"PyMuPDF fallback failed: {str(fitz_err)}"
                        ]
                    )
            else:
                return ExtractedDocument(
                    file_path=abs_path,
                    file_name=file_name,
                    status="CORRUPTED_FILE",
                    errors=[f"pdfplumber failed: {str(plumber_err)} (PyMuPDF fallback not available)"]
                )

    def _extract_with_pdfplumber(self, abs_path: str, file_name: str) -> ExtractedDocument:
        """Internal extractor using pdfplumber."""
        extracted_pages: List[ExtractedPage] = []
        total_text_chars = 0
        has_any_images = False
        meta = {}

        with pdfplumber.open(abs_path) as pdf:
            meta = {k: str(v) for k, v in (pdf.metadata or {}).items()}
            total_pages = len(pdf.pages)

            if total_pages == 0:
                return ExtractedDocument(
                    file_path=abs_path,
                    file_name=file_name,
                    total_pages=0,
                    status="EMPTY_DOCUMENT",
                    metadata=meta,
                    errors=["PDF has 0 pages."]
                )

            for page_idx, page in enumerate(pdf.pages, start=1):
                raw_text = page.extract_text() or ""
                cleaned = clean_page_text(raw_text, strip_simulation_banners=self.strip_simulation_banners)
                has_images = bool(page.images and len(page.images) > 0)

                if has_images:
                    has_any_images = True

                c_len = len(cleaned)
                w_len = len(cleaned.split())
                is_blank = (c_len == 0)
                total_text_chars += c_len

                extracted_pages.append(ExtractedPage(
                    page_number=page_idx,
                    raw_text=raw_text,
                    cleaned_text=cleaned,
                    char_count=c_len,
                    word_count=w_len,
                    has_images=has_images,
                    is_blank=is_blank,
                    engine_used="pdfplumber"
                ))

        # Check for scanned image-only PDF (images exist, but zero or near-zero extractable text)
        is_scanned = False
        status = "SUCCESS"
        notes = "Successfully extracted text layers with pdfplumber."

        if total_text_chars < self.min_text_chars_for_scanned and has_any_images:
            is_scanned = True
            status = "SCANNED_NEEDS_OCR"
            notes = (
                "Scanned image-only PDF detected. No embedded text stream found. "
                "OCR is not implemented in this baseline; text could not be extracted."
            )
        elif all(p.is_blank for p in extracted_pages):
            status = "BLANK_DOCUMENT"
            notes = "Document contains pages, but all pages are blank text."

        return ExtractedDocument(
            file_path=abs_path,
            file_name=file_name,
            total_pages=total_pages,
            extraction_engine="pdfplumber",
            status=status,
            is_scanned=is_scanned,
            ocr_applied=False,
            pages=extracted_pages,
            metadata=meta,
            notes=notes
        )

    def _extract_with_pymupdf(self, abs_path: str, file_name: str) -> ExtractedDocument:
        """Internal fallback extractor using PyMuPDF."""
        extracted_pages: List[ExtractedPage] = []
        total_text_chars = 0
        has_any_images = False
        meta = {}

        doc = fitz.open(abs_path)
        try:
            meta = {k: str(v) for k, v in (doc.metadata or {}).items()}
            total_pages = len(doc)

            if total_pages == 0:
                return ExtractedDocument(
                    file_path=abs_path,
                    file_name=file_name,
                    total_pages=0,
                    status="EMPTY_DOCUMENT",
                    metadata=meta,
                    errors=["PDF has 0 pages."]
                )

            for page_idx, page in enumerate(doc, start=1):
                raw_text = page.get_text("text") or ""
                cleaned = clean_page_text(raw_text, strip_simulation_banners=self.strip_simulation_banners)
                image_list = page.get_images()
                has_images = bool(image_list and len(image_list) > 0)

                if has_images:
                    has_any_images = True

                c_len = len(cleaned)
                w_len = len(cleaned.split())
                is_blank = (c_len == 0)
                total_text_chars += c_len

                extracted_pages.append(ExtractedPage(
                    page_number=page_idx,
                    raw_text=raw_text,
                    cleaned_text=cleaned,
                    char_count=c_len,
                    word_count=w_len,
                    has_images=has_images,
                    is_blank=is_blank,
                    engine_used="pymupdf"
                ))

            is_scanned = False
            status = "SUCCESS"
            notes = "Extracted text layers using PyMuPDF fallback engine."

            if total_text_chars < self.min_text_chars_for_scanned and has_any_images:
                is_scanned = True
                status = "SCANNED_NEEDS_OCR"
                notes = (
                    "Scanned image-only PDF detected via PyMuPDF. No text stream found. "
                    "OCR is not implemented in this baseline."
                )

            return ExtractedDocument(
                file_path=abs_path,
                file_name=file_name,
                total_pages=total_pages,
                extraction_engine="pymupdf_fallback",
                status=status,
                is_scanned=is_scanned,
                ocr_applied=False,
                pages=extracted_pages,
                metadata=meta,
                notes=notes
            )
        finally:
            doc.close()


def extract_pdf(file_path: str, strip_simulation_banners: bool = False) -> ExtractedDocument:
    """Convenience helper to extract a single PDF with default settings."""
    extractor = PDFExtractor(strip_simulation_banners=strip_simulation_banners)
    return extractor.extract(file_path)
