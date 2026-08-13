"""
retrieval.py
------------
Loads the persisted FAISS + BM25 indexes and the chunk store, and
performs hybrid retrieval: dense vector search + sparse BM25 search,
fused with Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np

_TOKEN_RE = re.compile(r"[A-Za-z\u0900-\u0D7F]{2,}")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class HybridIndex:
    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.faiss_index = faiss.read_index(str(self.index_dir / "faiss.index"))
        with open(self.index_dir / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        with open(self.index_dir / "embedder.pkl", "rb") as f:
            self.embedder = pickle.load(f)
        with open(self.index_dir / "chunks.json", "r", encoding="utf-8") as f:
            self.chunks: list[dict[str, Any]] = json.load(f)

    # ------------------------------------------------------------------
    def dense_search(self, query: str, k: int = 20) -> list[tuple]:
        qvec = self.embedder.encode([query])
        scores, idxs = self.faiss_index.search(qvec, k)
        out = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            out.append((int(idx), float(score)))
        return out

    def bm25_search(self, query: str, k: int = 20) -> list[tuple]:
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]

    # ------------------------------------------------------------------
    def hybrid_search(self, query: str, k_each: int = 20, rrf_k: int = 60,
                       language: str | None = None, top_n: int = 8) -> list[dict[str, Any]]:
        """Dense + BM25 retrieval fused with Reciprocal Rank Fusion.

        RRF score for a chunk = sum over result lists it appears in of
        1 / (rank_in_that_list + rrf_k). A chunk ranked highly by BOTH
        dense and sparse search gets a boosted combined score.
        """
        dense = self.dense_search(query, k_each)
        sparse = self.bm25_search(query, k_each)

        rrf_scores: dict[int, float] = {}
        for rank, (idx, _score) in enumerate(dense):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rank + rrf_k)
        for rank, (idx, _score) in enumerate(sparse):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rank + rrf_k)

        ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])

        results = []
        for idx, score in ranked:
            chunk = self.chunks[idx]
            if language and chunk["language"] != language:
                continue  # metadata-aware filtering
            results.append({**chunk, "rrf_score": score})
            if len(results) >= top_n:
                break
        return results
