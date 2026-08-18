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

from collections import defaultdict
import math

class FastBM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.N = 0
        self.avgdl = 0.0
        self.idf = {}
        self.postings = defaultdict(list)
        self.doc_lengths = []

    def get_scores(self, query_tokens: list[str]) -> dict[int, float]:
        import numpy as np
        doc_lists = []
        weight_lists = []
        for token in set(query_tokens):
            if token in self.postings:
                doc_lists.append(self.postings[token][0])
                weight_lists.append(self.postings[token][1])
        
        if not doc_lists:
            return {}
            
        all_docs = np.concatenate(doc_lists)
        all_weights = np.concatenate(weight_lists)
        scores = np.bincount(all_docs, weights=all_weights, minlength=self.N)
        
        # We still need to return a dictionary or something hybrid_search can iterate over
        # Wait, returning a dict here might be slow. The current code does:
        # scores_dict = self.bm25.get_scores(tokens)
        # heapq.nlargest(k, scores_dict.items(), key=lambda x: x[1])
        # If we return top-k directly from FastBM25 it's better. 
        # But for now let's just make it compatible. Or actually, FastBM25 is only used in bm25_search.
        return scores


class HybridIndex:
    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.faiss_index = faiss.read_index(str(self.index_dir / "dense.index"))
        
        # Load fast inverted BM25 if it exists, else load old rank_bm25
        fast_bm25_path = self.index_dir / "inverted_bm25.pkl"
        if fast_bm25_path.exists():
            import sys
            sys.modules['__main__'].FastBM25 = FastBM25
            with open(fast_bm25_path, "rb") as f:
                self.bm25 = pickle.load(f)
            self.use_fast_bm25 = True
        else:
            with open(self.index_dir / "bm25.pkl", "rb") as f:
                self.bm25 = pickle.load(f)
            self.use_fast_bm25 = False
            
        import sys
        import types
        import backend.rag.retrieval.embeddings as hhg_rag
        if 'rag' not in sys.modules:
            sys.modules['rag'] = types.ModuleType('rag')
        if 'rag.embeddings' not in sys.modules:
            sys.modules['rag.embeddings'] = types.ModuleType('rag.embeddings')
        sys.modules['rag.embeddings.encoder'] = hhg_rag
        
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
        if self.use_fast_bm25:
            scores = self.bm25.get_scores(tokens)
            if isinstance(scores, dict) and not scores:
                return []
            import numpy as np
            if isinstance(scores, np.ndarray):
                if len(scores) < k:
                    k = len(scores)
                if k == 0:
                    return []
                top_indices = np.argpartition(scores, -k)[-k:]
                top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
                return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]
            else:
                import heapq
                top_items = heapq.nlargest(k, scores.items(), key=lambda x: x[1])
                return [(int(idx), float(score)) for idx, score in top_items]
        else:
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

        dense_dict = {idx: rank for rank, (idx, _score) in enumerate(dense)}
        sparse_dict = {idx: rank for rank, (idx, _score) in enumerate(sparse)}
        
        rrf_scores: dict[int, float] = {}
        for idx, rank in dense_dict.items():
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rank + rrf_k)
        for idx, rank in sparse_dict.items():
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rank + rrf_k)

        ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])

        results = []
        for idx, score in ranked:
            chunk = self.chunks[idx]
            if language and chunk["language"] != language:
                continue  # metadata-aware filtering
            
            d_rank = dense_dict.get(idx, -1)
            s_rank = sparse_dict.get(idx, -1)
            
            results.append({
                **chunk, 
                "rrf_score": score,
                "dense_rank": d_rank,
                "sparse_rank": s_rank
            })
            if len(results) >= top_n:
                break
        return results


