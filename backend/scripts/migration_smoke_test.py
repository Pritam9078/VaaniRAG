import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from backend.rag.retrieval.retrieval import HybridIndex

INDEX_DIR = "backend/artifacts/msmarco_xi/v001"

def run_smoke_test():
    print("--- Starting Migration Smoke Test ---")
    
    t0 = time.perf_counter()
    index = HybridIndex(INDEX_DIR)
    t1 = time.perf_counter()
    
    print(f"Index loaded in {t1 - t0:.2f} seconds.")
    
    # Check chunks length
    num_chunks = len(index.chunks)
    print(f"Total chunks: {num_chunks}")
    assert num_chunks == 223883, f"Expected 223883 chunks, got {num_chunks}"
    
    # Test queries
    queries = [
        "What is the capital of India?",
        "How is cheese made?",
        "What is relativity?",
        "Tell me about the Eiffel Tower."
    ]
    
    for q in queries:
        print(f"\nQuery: {q}")
        t_start = time.perf_counter()
        results = index.hybrid_search(q, top_n=3)
        t_end = time.perf_counter()
        print(f"Latency: {(t_end - t_start)*1000:.2f} ms")
        
        for i, res in enumerate(results):
            print(f"  [{i+1}] Score: {res.get('rrf_score', 0):.4f} | DocID: {res.get('doc_id')} | {res['text'][:100]}...")

if __name__ == "__main__":
    run_smoke_test()
