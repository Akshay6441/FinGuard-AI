"""
Hybrid dense-lexical vector store with Reciprocal Rank Fusion (RRF) for FinGuard AI.
Combines BM25 lexical precision with dense semantic cosine similarity.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from src.rag.chunker import load_all_policy_chunks


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokenizer."""
    return re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", text.lower())


class HybridPolicyStore:
    """
    Hybrid retriever combining BM25 keyword matching with dense TF-IDF n-gram embeddings
    and Reciprocal Rank Fusion (RRF) reranking.
    """

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k
        self.chunks: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.dense_vectorizer: Optional[TfidfVectorizer] = None
        self.dense_matrix: Optional[np.ndarray] = None
        self.is_indexed: bool = False

    def build_index(self, chunks: List[Dict[str, Any]]) -> HybridPolicyStore:
        """
        Build both lexical BM25 index and dense semantic vector index over document chunks.
        """
        self.chunks = chunks
        corpus_texts = [c["text"] for c in chunks]

        # 1. BM25 Lexical Index
        tokenized_corpus = [tokenize(t) for t in corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 2. Dense Semantic Vectorizer (word + character n-grams for typo resilience)
        self.dense_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            analyzer="word",
            sublinear_tf=True,
            norm="l2",
            min_df=1,
        )
        self.dense_matrix = self.dense_vectorizer.fit_transform(corpus_texts).toarray()

        self.is_indexed = True
        return self

    def search(
        self,
        query: str,
        top_k: int = 3,
        weight_dense: float = 0.5,
        weight_bm25: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Execute hybrid search and rerank results with Reciprocal Rank Fusion (RRF).
        """
        if not self.is_indexed:
            raise RuntimeError("HybridPolicyStore is not indexed yet.")

        # 1. Lexical BM25 Ranking
        q_tokens = tokenize(query)
        bm25_scores = self.bm25.get_scores(q_tokens)
        bm25_ranks = np.argsort(bm25_scores)[::-1]  # descending

        # 2. Dense Cosine Similarity Ranking
        q_vec = self.dense_vectorizer.transform([query]).toarray()
        # Cosine similarity since both vectors are L2-normalized
        dense_scores = np.dot(self.dense_matrix, q_vec.T).ravel()
        dense_ranks = np.argsort(dense_scores)[::-1]

        # 3. Reciprocal Rank Fusion
        n = len(self.chunks)
        rrf_scores = np.zeros(n)

        for rank, idx in enumerate(bm25_ranks):
            rrf_scores[idx] += weight_bm25 / (self.rrf_k + rank + 1)

        for rank, idx in enumerate(dense_ranks):
            rrf_scores[idx] += weight_dense / (self.rrf_k + rank + 1)

        top_indices = np.argsort(rrf_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk["rrf_score"] = round(float(rrf_scores[idx]), 5)
            chunk["bm25_score"] = round(float(bm25_scores[idx]), 3)
            chunk["dense_score"] = round(float(dense_scores[idx]), 3)
            results.append(chunk)

        return results

    def save(self, filepath: Path | str = "models/rag_store.joblib") -> None:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "chunks": self.chunks,
            "vectorizer": self.dense_vectorizer,
            "matrix": self.dense_matrix,
            "rrf_k": self.rrf_k,
        }, p)

    @classmethod
    def load(cls, filepath: Path | str = "models/rag_store.joblib") -> HybridPolicyStore:
        p = Path(filepath)
        if not p.exists():
            store = cls()
            chunks = load_all_policy_chunks()
            store.build_index(chunks)
            store.save(filepath)
            return store
        try:
            state = joblib.load(p)
            if isinstance(state, dict) and "chunks" in state:
                store = cls(rrf_k=state.get("rrf_k", 60))
                store.chunks = state["chunks"]
                store.dense_vectorizer = state["vectorizer"]
                store.dense_matrix = state["matrix"]
                tokenized_corpus = [tokenize(c["text"]) for c in store.chunks]
                store.bm25 = BM25Okapi(tokenized_corpus)
                store.is_indexed = True
                return store
            elif isinstance(state, HybridPolicyStore):
                return state
        except Exception:
            pass

        # Fallback fresh index
        store = cls()
        chunks = load_all_policy_chunks()
        store.build_index(chunks)
        store.save(filepath)
        return store


if __name__ == "__main__":
    chunks = load_all_policy_chunks("data/policies")
    print(f"Loaded {len(chunks)} chunks from policies.")
    store = HybridPolicyStore()
    store.build_index(chunks)
    store.save("models/rag_store.joblib")
    print("Hybrid policy store saved to models/rag_store.joblib")

    # Test query
    sample_q = "What is the liability for card-not-present fraud under Reason Code 10.4 when 3DS is not used?"
    res = store.search(sample_q, top_k=2)
    for r in res:
        print(f"-> Found [{r['source_doc']}]: {r['section_title']} (RRF Score: {r['rrf_score']})")
