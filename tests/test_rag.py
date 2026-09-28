"""
Unit test suite for RAG chunker, hybrid store, and policy retrieval.
"""

from __future__ import annotations

from src.rag.chunker import load_all_policy_chunks
from src.rag.hybrid_store import HybridPolicyStore


def test_policy_chunking():
    chunks = load_all_policy_chunks("data/policies")
    assert len(chunks) > 0
    first = chunks[0]
    assert "chunk_id" in first
    assert "source_doc" in first
    assert "section_title" in first
    assert "text" in first
    assert len(first["text"]) > 20


def test_hybrid_policy_search():
    store = HybridPolicyStore.load("models/rag_store.joblib")
    results = store.search("What is Rule VEL-01 regarding hourly velocity limits?", top_k=3)
    assert len(results) > 0
    assert "rrf_score" in results[0]
    assert "bm25_score" in results[0]
    assert "dense_score" in results[0]

    # Verify relevant keywords in top result
    top_text = results[0]["text"].lower()
    assert "velocity" in top_text or "vel-01" in top_text


def test_zero_liability_policy_search():
    store = HybridPolicyStore.load("models/rag_store.joblib")
    results = store.search("Under what conditions does a customer have zero liability?", top_k=2)
    assert len(results) > 0
    top_text = results[0]["text"].lower()
    assert "liability" in top_text
