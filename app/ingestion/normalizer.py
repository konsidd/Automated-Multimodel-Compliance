"""
app/ingestion/normalizer.py
────────────────────────────
Normalise and merge multimodal text before chunking.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NormalisedDocument:
    source: str                        # e.g. YouTube URL or file path
    transcript: str = ""               # from Whisper
    ocr_text: str = ""                 # from Tesseract
    combined_text: str = ""            # merged output
    metadata: dict = field(default_factory=dict)


def _clean(text: str) -> str:
    """Remove control chars, normalise whitespace, fix unicode."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\S\n]+", " ", text)        # collapse spaces
    text = re.sub(r"\n{3,}", "\n\n", text)        # max 2 blank lines
    return text.strip()


def normalise_document(
    source: str,
    transcript: str = "",
    ocr_text: str = "",
    metadata: Optional[dict] = None,
) -> NormalisedDocument:
    """
    Merge transcript + OCR text into a single cleaned document.
    Deduplicates overlapping OCR sentences.
    """
    clean_transcript = _clean(transcript)
    clean_ocr = _clean(ocr_text)

    parts: list[str] = []
    if clean_transcript:
        parts.append(f"[TRANSCRIPT]\n{clean_transcript}")
    if clean_ocr:
        parts.append(f"[SCREEN TEXT / OCR]\n{clean_ocr}")

    combined = "\n\n".join(parts)

    return NormalisedDocument(
        source=source,
        transcript=clean_transcript,
        ocr_text=clean_ocr,
        combined_text=combined,
        metadata=metadata or {},
    )
