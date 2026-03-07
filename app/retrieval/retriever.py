"""
app/retrieval/retriever.py
───────────────────────────
RAG retriever: rewrites query, searches FAISS, formats context.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from app.llm.client import invoke_llm
from app.llm.prompts import QUERY_REWRITER_SYSTEM, QUERY_REWRITER_USER
from app.retrieval.vector_store import FAISSVectorStore
from app.telemetry.logging_config import get_logger
from app.utils.config import get_config

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_vector_store() -> FAISSVectorStore:
    return FAISSVectorStore()


def rewrite_query(raw_query: str) -> str:
    """Use the LLM to optimise the search query for the vector store."""
    rewritten = invoke_llm(
        system_prompt=QUERY_REWRITER_SYSTEM,
        user_prompt=QUERY_REWRITER_USER.format(input_text=raw_query),
    )
    logger.info("query_rewritten", original=raw_query[:80], rewritten=rewritten[:80])
    return rewritten


def retrieve_context(query: str, top_k: Optional[int] = None, rewrite: bool = True) -> str:
    """
    Retrieve relevant policy/document chunks for the given query.
    Returns formatted context string ready for the LLM prompt.
    """
    search_query = rewrite_query(query) if rewrite else query
    store = get_vector_store()
    results = store.search(search_query, top_k=top_k)

    if not results:
        return "No relevant policy context found."

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        source = r.get("source", "unknown")
        text = r.get("text", "")
        lines.append(f"[{i}] Source: {source}\n{text}")

    return "\n\n---\n\n".join(lines)
