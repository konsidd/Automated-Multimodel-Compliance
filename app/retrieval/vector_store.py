"""
app/retrieval/vector_store.py
──────────────────────────────
FAISS-based vector store with JSON metadata sidecar.
Replaces Azure AI Search with a fully local solution.
"""

from __future__ import annotations
import json
import faiss
import numpy as np

from typing import Optional, List
from dataclasses import dataclass
from pathlib import Path
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from app.retrieval.chunking import Chunk
from app.retrieval.embeddings import embed_texts, embed_query
from app.telemetry.logging_config import get_logger
from app.utils.config import get_config

logger = get_logger(__name__)


class FAISSVectorStore:
    """
    In-memory / on-disk FAISS vector store.

    Layout on disk:
      <index_path>        – FAISS binary index
      <metadata_path>     – JSON array of chunk metadata dicts
    """

    def __init__(self, index_path: Optional[str] = None, metadata_path: Optional[str] = None):
        cfg = get_config().vector_store
        root = get_config().root_dir

        self.index_path = Path(index_path or (root / cfg.index_path))
        self.metadata_path = Path(metadata_path or (root / cfg.metadata_path))
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        self._index: Optional[faiss.Index] = None
        self._metadata: list[dict] = []
        self._dim: Optional[int] = None

        if self.index_path.exists() and self.metadata_path.exists():
            self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        self._index = faiss.read_index(str(self.index_path))
        self._metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        # Pylance doesn't know that `read_index` never returns None; assert to
        # narrow the type so we can access attributes without complaints.
        assert self._index is not None
        self._dim = self._index.d
        logger.info("vector_store_loaded", vectors=self._index.ntotal, dim=self._dim)

    def save(self) -> None:
        if self._index is None:
            return
        faiss.write_index(self._index, str(self.index_path))
        self.metadata_path.write_text(
            json.dumps(self._metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("vector_store_saved", vectors=self._index.ntotal)

    # ── Indexing ─────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Embed chunks and add to the FAISS index."""
        if not chunks:
            return

        texts = [c.text for c in chunks]
        vectors = embed_texts(texts)  # (N, dim)

        if self._index is None:
            dim = vectors.shape[1]
            self._dim = dim
            self._index = faiss.IndexFlatL2(dim)
            logger.info("faiss_index_created", dim=dim)

        # after the check above, index is guaranteed
        assert self._index is not None
        self._index.add(vectors)  # type: ignore[call-arg]

        for chunk in chunks:
            self._metadata.append({
                "text": chunk.text,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                **chunk.metadata,
            })

        logger.info("chunks_added", count=len(chunks), total=self._index.ntotal)

    # ── Search ───────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """
        Search for the most relevant chunks.
        Returns list of metadata dicts with added 'score' (L2 distance).
        """
        if self._index is None or self._index.ntotal == 0:
            logger.warning("vector_store_empty_search")
            return []

        # narrow for pylance and avoid repeated Optional checks
        index = self._index  # type: faiss.Index
        assert index is not None

        k = top_k or get_config().vector_store.top_k
        q_vec = embed_query(query).reshape(1, -1)
        # signature stubs sometimes don't match; ignore param errors here
        distances, indices = index.search(q_vec, k)  # type: ignore[call-arg]

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            meta = self._metadata[idx].copy()
            meta["score"] = float(dist)
            results.append(meta)

        return results

    # ── Utilities ─────────────────────────────────────────────────────────────

    def clear(self) -> None:
        self._index = None
        self._metadata = []
        self._dim = None
        logger.info("vector_store_cleared")

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._index else 0
