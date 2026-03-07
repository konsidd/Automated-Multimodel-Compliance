"""
app/retrieval/embeddings.py
────────────────────────────
Local embeddings via sentence-transformers (no cloud required).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Union

import numpy as np

from app.telemetry.logging_config import get_logger
from app.utils.config import get_config

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _get_model():
    """Load and cache the sentence-transformer model."""
    from sentence_transformers import SentenceTransformer  # type: ignore
    cfg = get_config().embeddings
    logger.info("loading_embedding_model", model=cfg.model, device=cfg.device)
    return SentenceTransformer(cfg.model, device=cfg.device)


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of strings.
    Returns float32 numpy array of shape (N, dim).
    """
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns shape (dim,)."""
    return embed_texts([query])[0]
