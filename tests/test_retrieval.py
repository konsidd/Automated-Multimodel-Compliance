"""
tests/test_retrieval.py
────────────────────────
Unit tests for the retrieval / vector store layer.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.retrieval.chunking import chunk_text, Chunk
from app.retrieval.embeddings import embed_texts, embed_query
from app.retrieval.vector_store import FAISSVectorStore


# ── Chunking ──────────────────────────────────────────────────────────────────

def test_chunk_text_basic():
    text = " ".join(["word"] * 200)
    chunks = chunk_text(text, source="test", chunk_size=50, chunk_overlap=10)
    assert len(chunks) > 1
    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.source == "test"


def test_chunk_text_small_input():
    chunks = chunk_text("Hello world", source="tiny", chunk_size=50, chunk_overlap=5)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world"


# ── Embeddings ────────────────────────────────────────────────────────────────

def test_embed_texts_shape():
    vecs = embed_texts(["Hello world", "Compliance matters"])
    assert vecs.shape[0] == 2
    assert vecs.shape[1] > 0


def test_embed_query_shape():
    vec = embed_query("What is GDPR?")
    assert vec.ndim == 1
    assert len(vec) > 0


# ── FAISS Vector Store ────────────────────────────────────────────────────────

@pytest.fixture
def tmp_store(tmp_path):
    index_path = str(tmp_path / "test.index")
    meta_path = str(tmp_path / "test_meta.json")
    return FAISSVectorStore(index_path=index_path, metadata_path=meta_path)


def test_add_and_search(tmp_store):
    chunks = [
        Chunk(text="GDPR requires explicit user consent for data collection.", source="doc1", chunk_index=0),
        Chunk(text="Cardholder data must be encrypted.", source="doc2", chunk_index=0),
        Chunk(text="Access logs must be maintained.", source="doc3", chunk_index=0),
    ]
    tmp_store.add_chunks(chunks)
    assert tmp_store.total_vectors == 3

    results = tmp_store.search("data privacy consent", top_k=2)
    assert len(results) <= 2
    assert "text" in results[0]
    assert "score" in results[0]


def test_save_and_reload(tmp_store, tmp_path):
    chunks = [Chunk(text="Test chunk for save/load.", source="test", chunk_index=0)]
    tmp_store.add_chunks(chunks)
    tmp_store.save()

    # Reload
    index_path = str(tmp_path / "test.index")
    meta_path = str(tmp_path / "test_meta.json")
    store2 = FAISSVectorStore(index_path=index_path, metadata_path=meta_path)
    assert store2.total_vectors == 1


def test_empty_search_returns_empty(tmp_store):
    results = tmp_store.search("anything")
    assert results == []
