"""
app/retrieval/chunking.py
──────────────────────────
Split documents into overlapping chunks for embedding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.utils.config import get_config


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def chunk_text(
    text: str,
    source: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    metadata: Optional[dict] = None,
) -> list[Chunk]:
    """
    Split text into overlapping chunks.
    Uses sentence-aware splitting to avoid cutting mid-sentence.
    """
    cfg = get_config().ingestion
    size = chunk_size or cfg.chunk_size
    overlap = chunk_overlap or cfg.chunk_overlap
    meta = metadata or {}

    words = text.split()
    chunks: list[Chunk] = []
    start = 0
    idx = 0

    while start < len(words):
        end = min(start + size, len(words))
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)
        chunks.append(Chunk(
            text=chunk_text_str,
            source=source,
            chunk_index=idx,
            metadata=meta,
        ))
        idx += 1
        start += size - overlap

    return chunks


def chunk_document(doc_text: str, source: str, metadata: Optional[dict] = None) -> list[Chunk]:
    return chunk_text(doc_text, source, metadata=metadata)
