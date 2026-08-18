import json
import pickle
import faiss
from pathlib import Path
import sys

def main():
    root = Path(__file__).resolve().parent.parent.parent
    index_dir = root / "backend" / "artifacts" / "msmarco_xi" / "v001"
    
    # 1. Load manifest
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        sys.exit(1)
        
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    expected_chunks = manifest.get("chunks")
    print(f"Manifest expects {expected_chunks} chunks.")
    
    # 2. Check metadata
    chunks_path = index_dir / "chunks.json"
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    metadata_count = len(chunks)
    print(f"Metadata (chunks.json) count: {metadata_count}")
    
    # 3. Check FAISS
    faiss_path = index_dir / "dense.index"
    index = faiss.read_index(str(faiss_path))
    dense_count = index.ntotal
    print(f"FAISS (dense.index) ntotal: {dense_count}")
    
    # 4. Check Sparse
    bm25_path = index_dir / "bm25.pkl"
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)
    
    # For rank_bm25, doc_len stores the length of each document.
    sparse_count = len(bm25.doc_len)
    print(f"Sparse (bm25.pkl) count: {sparse_count}")
    
    # 5. Assertions
    assert expected_chunks == metadata_count, "Metadata count mismatch!"
    assert expected_chunks == dense_count, "FAISS count mismatch!"
    assert expected_chunks == sparse_count, "Sparse BM25 count mismatch!"
    
    # 6. Verify Embedding compatibility with our online settings
    # For VaaniRAG's current setup, the embedder handles TF-IDF SVD
    embedder_path = index_dir / "embedder.pkl"
    with open(embedder_path, "rb") as f:
        import backend.rag.retrieval.embeddings as hhg_rag
        import types
        if 'rag' not in sys.modules:
            sys.modules['rag'] = types.ModuleType('rag')
        if 'rag.embeddings' not in sys.modules:
            sys.modules['rag.embeddings'] = types.ModuleType('rag.embeddings')
        sys.modules['rag.embeddings.encoder'] = hhg_rag
        
        embedder = pickle.load(f)
    
    # In TfidfSvdEmbedder, the svd component holds the n_components which is the dimension
    dim = embedder.svd.n_components
    expected_dim = manifest.get("embedding_dimension")
    print(f"Embedding model dimension: {dim} (Expected: {expected_dim})")
    assert dim == expected_dim, f"Embedding dimension mismatch: {dim} vs {expected_dim}"
    
    # Validate missing texts
    missing_texts = sum(1 for c in chunks if not c.get("text"))
    print(f"Chunks with missing text: {missing_texts}")
    assert missing_texts == 0, "Found chunks with empty text!"
    
    print("✅ Index validation passed!")
    
if __name__ == "__main__":
    main()
