import time
import json
import random
import numpy as np
from pathlib import Path
from backend.rag.retrieval.embeddings import get_embedder

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "backend" / "artifacts" / "msmarco_xi" / "v001"

def benchmark_embeddings():
    print("Loading queries...")
    with open("evaluation/queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)[:100]  # First 100 queries
        
    query_texts = [q["query"] for q in queries]
    
    backends = [
        {"name": "model2vec (Current)", "id": "model2vec"},
        {"name": "sentence_transformers (Deep Neural)", "id": "sentence_transformer"}
    ]
    
    results = []
    
    for b in backends:
        print(f"\nEvaluating {b['name']}...")
        try:
            embedder = get_embedder(b['id'])
            
            # Warmup
            embedder.encode(["warmup"])
            
            latencies = []
            for q in query_texts:
                t0 = time.perf_counter()
                embedder.encode([q])
                latencies.append((time.perf_counter() - t0) * 1000)
                
            results.append({
                "Backend": b['name'],
                "Dim": embedder.dim,
                "P50 (ms)": np.percentile(latencies, 50),
                "P100 (ms)": np.max(latencies)
            })
        except Exception as e:
            print(f"Failed to load {b['name']}: {e}")
            
    print("\n" + "="*60)
    print(f"{'Backend':<30} | {'Dim':<5} | {'P50':<8} | {'P100':<8}")
    print("-" * 60)
    for r in results:
        print(f"{r['Backend']:<30} | {r['Dim']:<5} | {r['P50 (ms)']:<8.2f} | {r['P100 (ms)']:<8.2f}")

if __name__ == "__main__":
    benchmark_embeddings()
