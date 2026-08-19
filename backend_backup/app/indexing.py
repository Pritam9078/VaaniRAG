"""
indexing.py
-----------
Offline indexing job: loads the corpus, runs every chunking strategy,
embeds all chunks, and builds:
  - a FAISS HNSW index for dense vector search
  - a BM25 (rank_bm25) index for sparse lexical search

Run once, ahead of query time (see scripts/build_index.py). The
resulting index + chunk store are persisted to disk and simply loaded
at server start -- no chunking/embedding happens on the request path.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

import faiss
from rank_bm25 import BM25Okapi

from .chunking import Chunk, chunk_document
from .embeddings import BaseEmbedder, get_embedder

_TOKEN_RE = re.compile(r"[A-Za-z\u0900-\u097F]{2,}")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def build_index(corpus_path: str, out_dir: str,
                 strategies: list[str] = None,
                 embedding_backend: str = "tfidf_svd") -> dict[str, Any]:
    corpus_path = Path(corpus_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(corpus_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    all_chunks: list[Chunk] = []
    for doc in docs:
        doc_id = doc["doc_id"]
        for passage in doc["passages"]:
            lang = passage["lang"]
            text = passage["text"]
            all_chunks.extend(chunk_document(f"{doc_id}:{lang}", text, lang, strategies))

    if not all_chunks:
        raise ValueError("No chunks produced from corpus -- check corpus_path")

    texts = [c.text for c in all_chunks]

    # --- Dense embeddings + FAISS ---
    embedder: BaseEmbedder = get_embedder(embedding_backend)
    embedder.fit(texts)
    vectors = embedder.encode(texts)

    index = faiss.IndexHNSWFlat(vectors.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 80
    index.hnsw.efSearch = 64
    index.add(vectors)

    faiss.write_index(index, str(out_dir / "faiss.index"))

    # --- BM25 ---
    tokenized = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    with open(out_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    # --- Embedder + chunk metadata store ---
    with open(out_dir / "embedder.pkl", "wb") as f:
        pickle.dump(embedder, f)

    chunk_records = [c.to_dict() for c in all_chunks]
    with open(out_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunk_records, f, ensure_ascii=False)

    stats = {
        "num_docs": len(docs),
        "num_chunks": len(all_chunks),
        "chunk_type_counts": {
            t: sum(1 for c in all_chunks if c.chunk_type == t)
            for t in sorted(set(c.chunk_type for c in all_chunks))
        },
        "embedding_dim": int(vectors.shape[1]),
    }
    with open(out_dir / "index_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


if __name__ == "__main__":
    import sys
    corpus = sys.argv[1] if len(sys.argv) > 1 else "backend/data/sample_corpus.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "backend/data/index"
    print(build_index(corpus, out))
