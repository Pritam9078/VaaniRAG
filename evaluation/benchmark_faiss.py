import time
import argparse
import numpy as np
import faiss
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "backend" / "artifacts" / "msmarco_xi" / "v001"

def benchmark_faiss(k: int = 10, num_queries: int = 1000):
    print("Loading vectors from existing HNSW index...")
    base_index = faiss.read_index(str(INDEX_DIR / "dense.index"))
    vectors = base_index.reconstruct_n(0, base_index.ntotal)
    
    dim = vectors.shape[1]
    
    # Generate random queries for latency benchmark
    np.random.seed(42)
    query_vectors = np.random.randn(num_queries, dim).astype(np.float32)
    faiss.normalize_L2(query_vectors)

    configs = {
        "IndexFlatIP": faiss.IndexFlatIP(dim),
        "IndexHNSWFlat": faiss.IndexHNSWFlat(dim, 32),
        "IndexIVFFlat": faiss.index_factory(dim, "IVF1024,Flat"),
        "IndexIVFPQ": faiss.index_factory(dim, "IVF1024,PQ32x4fs")
    }

    results = []

    for name, index in configs.items():
        print(f"\nEvaluating {name}...")
        
        # Training (if needed)
        t0 = time.time()
        if not index.is_trained:
            print("  Training...")
            index.train(vectors)
        train_time = time.time() - t0
        
        # Building
        t0 = time.time()
        index.add(vectors)
        build_time = time.time() - t0
        
        # Latency benchmark
        latencies = []
        for i in range(num_queries):
            q = query_vectors[i:i+1]
            t_start = time.perf_counter()
            index.search(q, k)
            latencies.append((time.perf_counter() - t_start) * 1000)
            
        p50 = np.percentile(latencies, 50)
        p70 = np.percentile(latencies, 70)
        p100 = np.max(latencies)
        
        results.append({
            "Name": name,
            "Build (s)": round(build_time, 2),
            "P50 (ms)": round(p50, 2),
            "P70 (ms)": round(p70, 2),
            "P100 (ms)": round(p100, 2)
        })
        
    print("\n" + "="*60)
    print(f"{'Configuration':<15} | {'Build(s)':<8} | {'P50':<6} | {'P70':<6} | {'P100':<6}")
    print("-" * 60)
    for r in results:
        print(f"{r['Name']:<15} | {r['Build (s)']:<8} | {r['P50 (ms)']:<6} | {r['P70 (ms)']:<6} | {r['P100 (ms)']:<6}")

if __name__ == "__main__":
    benchmark_faiss()
