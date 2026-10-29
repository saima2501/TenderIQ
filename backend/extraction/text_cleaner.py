"""
text_cleaner.py - Text Cleaning & Normalization for Extracted Procurement Documents
Part of Phase 2: PDF Document Extraction

Functions:
- Unicode normalization (NFKC)
- Whitespace and line-break consolidation
- De-hyphenation across line endings (e.g. 'infra-\\nstructure' -> 'infrastructure')
- Bullet and quotation standardization
- Removal of non-printable control characters
- Preservation of section numbering, tables, and clause structure
"""

import re
import unicodedata
from typing import Dict, Any, List, Optional


class TextCleaner:
    """Comprehensive text cleaner designed for procurement tenders and bidder proposals."""

    def __init__(self, strip_simulation_banners: bool = False):
        self.strip_simulation_banners = strip_simulation_banners

    def clean_text(self, text: Optional[str]) -> str:
        """Cleans and standardizes raw text extracted from a PDF page."""
        if not text:
            return ""

        # 1. Unicode Normalization (converts ligatures like 'fi', 'fl', special spaces)
        cleaned = unicodedata.normalize("NFKC", text)

        # 2. Convert Windows \r\n and old Mac \r to standard \n
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

        # 3. Standardize Quotation Marks & Apostrophes
        quote_map = {
            "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
            "‘": "'", "’": "'", "‚": "'", "‛": "'"
        }
        for curly, straight in quote_map.items():
            cleaned = cleaned.replace(curly, straight)

        # 4. Standardize Bullets & Dashes
        cleaned = re.sub(r"^[ \t]*[•▪►●*][ \t]+", "- ", cleaned, flags=re.MULTILINE)
        cleaned = cleaned.replace("–", "-").replace("—", "-")

        # 5. Fix hyphenated line breaks (e.g., 'infra-\nstructure' -> 'infrastructure')
        # Only when the first part is letters and the second part continues letters
        cleaned = re.sub(r"([A-Za-z]+)-\n([a-z]+)", r"\1\2", cleaned)

        # 6. Normalize non-breaking spaces and irregular whitespace
        cleaned = re.sub(r"[\u00a0\u2000-\u200b\u202f\u205f\u3000]", " ", cleaned)

        # 7. Remove non-printable control characters (keep \n and \t)
        cleaned = "".join(ch for ch in cleaned if ch in ("\n", "\t") or ch >= " ")

        # 8. Clean up whitespace per line: collapse multiple spaces and trim edges
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in cleaned.split("\n")]

        # 9. Optional: strip simulated watermark banners if requested
        if self.strip_simulation_banners:
            filtered_lines = []
            for l in lines:
                if "*** SIMULATED" in l or "SIMULATED PROCUREMENT DATASET" in l:
                    continue
                filtered_lines.append(l)
            lines = filtered_lines

        # 10. Consolidate excessive blank lines (more than 2 consecutive newlines -> 2)
        reconstructed = "\n".join(lines)
        reconstructed = re.sub(r"\n{3,}", "\n\n", reconstructed)

        return reconstructed.strip()


# Convenience global instance
_default_cleaner = TextCleaner()


def clean_page_text(raw_text: Optional[str], strip_simulation_banners: bool = False) -> str:
    """Convenience helper to clean a single page of text."""
    if strip_simulation_banners:
        cleaner = TextCleaner(strip_simulation_banners=True)
        return cleaner.clean_text(raw_text)
    return _default_cleaner.clean_text(raw_text)


def clean_extracted_document(pages: List[Dict[str, Any]], strip_simulation_banners: bool = False) -> List[Dict[str, Any]]:
    """Clean all pages in an extracted document dictionary list."""
    cleaner = TextCleaner(strip_simulation_banners=strip_simulation_banners)
    for p in pages:
        p["cleaned_text"] = cleaner.clean_text(p.get("raw_text", ""))
        p["word_count"] = len(p["cleaned_text"].split())
        p["char_count"] = len(p["cleaned_text"])
    return pages
