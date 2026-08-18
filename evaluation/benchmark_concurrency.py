import argparse
import json
import time
import statistics as stats
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from backend.rag.retrieval.retrieval import HybridIndex
from backend.rag.reranking.rerank import rerank
from backend.rag.generation.generation import get_generator

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "backend" / "artifacts" / "msmarco_xi" / "v001"
EVAL_DIR = ROOT / "evaluation"

def build_query_set(n: int) -> list[str]:
    with open(EVAL_DIR / "queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
    return [q["query"] for q in queries[:n]]

def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)

def run_load_test(concurrency: int = 5, total_requests: int = 20):
    index = HybridIndex(str(INDEX_DIR))
    generator = get_generator()
    queries = build_query_set(total_requests)

    latencies = []
    
    def process_query(q: str):
        t0 = time.perf_counter()
        
        # 1. Retrieval
        candidates = index.hybrid_search(q, top_n=5)
        
        # 2. Rerank
        top_chunks = rerank(q, candidates, top_n=3)
        
        # 3. Generation
        # (Mock generation since Groq will hit rate limits under concurrency)
        # generator.generate(q, top_chunks)
        time.sleep(0.5) # Simulate generation
        
        total_ms = (time.perf_counter() - t0) * 1000
        return total_ms

    print(f"Running Concurrency Benchmark ({concurrency} parallel users, {total_requests} total requests)...")
    
    start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(process_query, queries))
    
    total_time_s = time.perf_counter() - start_time
    
    for r in results:
        latencies.append(r)

    qps = total_requests / total_time_s

    report = {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "total_time_s": round(total_time_s, 2),
        "QPS": round(qps, 2),
        "total_ms": {
            "P50": round(percentile(latencies, 50), 3),
            "P95": round(percentile(latencies, 95), 3),
            "P100": round(percentile(latencies, 100), 3),
            "mean": round(stats.mean(latencies), 3)
        }
    }
    return report

if __name__ == "__main__":
    report = run_load_test(concurrency=10, total_requests=50)
    print(json.dumps(report, indent=2))
    
    out_dir = EVAL_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "concurrency_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
