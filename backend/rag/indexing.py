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

from backend.rag.chunking.chunking import Chunk, chunk_document
from backend.rag.retrieval.embeddings import BaseEmbedder, get_embedder

_TOKEN_RE = re.compile(r"[A-Za-z\u0900-\u097F]{2,}")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def build_index(corpus_path: str | None, out_dir: str,
                 strategies: list[str] | None = None,
                 embedding_backend: str = "tfidf_svd",
                 sample_size: int = 1000) -> dict[str, Any]:
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    all_chunks: list[Chunk] = []
    queries = []
    docs_processed = 0  # Initialize docs_processed outside
    
    if corpus_path:
        corpus_path_obj = Path(corpus_path)
        with open(corpus_path_obj, "r", encoding="utf-8") as f:
            docs = json.load(f)
        for doc in docs:
            doc_id = doc["doc_id"]
            for passage in doc["passages"]:
                lang = passage["lang"]
                text = passage["text"]
                all_chunks.extend(chunk_document(f"{doc_id}:{lang}", text, lang, strategies))
            docs_processed += 1
    else:
        from backend.scripts.fetch_msmarco import stream_msmarco
        for doc in stream_msmarco(sample_size=sample_size):
            doc_id = doc["doc_id"]
            expected_chunks = []
            for passage in doc["passages"]:
                lang = passage["lang"]
                text = passage["text"]
                is_sel = passage.get("is_selected", 0)
                chunks = chunk_document(f"{doc_id}:{lang}", text, lang, strategies, is_selected=is_sel)
                all_chunks.extend(chunks)
                if is_sel:
                    expected_chunks.extend([c.chunk_id for c in chunks])
            
            if expected_chunks and doc.get("query_tgt"):
                queries.append({
                    "query_id": doc_id,
                    "query": doc["query_tgt"],
                    "expected_chunk_ids": expected_chunks
                })
            docs_processed += 1
            if docs_processed >= sample_size:
                break
                
        # Save queries for evaluation
        eval_path = out_dir_path.parent.parent.parent / "evaluation" / "queries.json"
        eval_path.parent.mkdir(exist_ok=True)
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(queries, f, ensure_ascii=False, indent=2)

    if not all_chunks:
        raise ValueError("No chunks produced from corpus")

    texts = [c.text for c in all_chunks]

    # --- Dense embeddings + FAISS ---
    embedder: BaseEmbedder = get_embedder(embedding_backend)
    embedder.fit(texts)
    vectors = embedder.encode(texts)

    index = faiss.IndexHNSWFlat(vectors.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 80
    index.hnsw.efSearch = 64
    index.add(vectors)

    faiss.write_index(index, str(out_dir_path / "faiss.index"))

    # --- BM25 ---
    tokenized = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    with open(out_dir_path / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    # --- Embedder + chunk metadata store ---
    with open(out_dir_path / "embedder.pkl", "wb") as f:
        pickle.dump(embedder, f)

    chunk_records = [c.to_dict() for c in all_chunks]
    with open(out_dir_path / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunk_records, f, ensure_ascii=False)

    stats = {
        "num_docs": docs_processed,
        "num_chunks": len(all_chunks),
        "chunk_type_counts": {
            t: sum(1 for c in all_chunks if c.chunk_type == t)
            for t in sorted(set(c.chunk_type for c in all_chunks))
        },
        "embedding_dim": int(vectors.shape[1]),
    }
    with open(out_dir_path / "index_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


if __name__ == "__main__":
    import sys
    corpus = sys.argv[1] if len(sys.argv) > 1 else "backend/data/sample_corpus.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "backend/data/index"
    print(build_index(corpus, out))
