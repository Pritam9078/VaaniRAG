import json
import time
import statistics as stats
from pathlib import Path

from backend.rag.retrieval.retrieval import HybridIndex
from backend.rag.reranking.rerank import rerank

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "backend" / "artifacts" / "msmarco_xi" / "v001"
EVAL_DIR = ROOT / "evaluation"

def load_queries():
    with open(EVAL_DIR / "queries.json", "r", encoding="utf-8") as f:
        return json.load(f)

def load_qrels():
    with open(EVAL_DIR / "qrels.json", "r", encoding="utf-8") as f:
        return json.load(f)

def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)

def evaluate(index, queries, qrels, config: str):
    latencies = []
    recalls = []
    mrrs = []
    
    # Warmup
    warmup_q = "warmup query"
    _ = index.hybrid_search(warmup_q, top_n=5)
    rerank(warmup_q, [{"text": "dummy", "chunk_id": "dummy"}], top_n=1)
    
    for q_obj in queries[:100]:  # Limit to 100 for speed
        q_id = str(q_obj["query_id"])
        q_text = q_obj["query"]
        expected_docs = set(qrels.get(q_id, []))
        
        t0 = time.perf_counter()
        
        candidates = []
        if config == "Dense only":
            raw = index.dense_search(q_text, k=10)
            candidates = [{"chunk_id": index.chunks[idx]["chunk_id"], "text": index.chunks[idx]["text"]} for idx, _ in raw]
        elif config == "BM25 only":
            raw = index.bm25_search(q_text, k=10)
            candidates = [{"chunk_id": index.chunks[idx]["chunk_id"], "text": index.chunks[idx]["text"]} for idx, _ in raw]
        elif config == "Dense + BM25 + RRF":
            candidates = index.hybrid_search(q_text, top_n=10)
        elif config == "Hybrid + Reranker Top 10":
            candidates = index.hybrid_search(q_text, top_n=10)
            candidates = rerank(q_text, candidates, top_n=10)
        elif config == "Hybrid + Reranker Top 5":
            candidates = index.hybrid_search(q_text, top_n=5)
            candidates = rerank(q_text, candidates, top_n=5)
        elif config == "Hybrid + Reranker Top 3":
            candidates = index.hybrid_search(q_text, top_n=3)
            candidates = rerank(q_text, candidates, top_n=3)
            
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)
        
        retrieved_ids = [c["chunk_id"] for c in candidates]
        
        # Recall@10
        hits = len(expected_docs.intersection(set(retrieved_ids[:10])))
        recall = hits / len(expected_docs) if expected_docs else 0.0
        recalls.append(recall)
        
        # MRR@10
        mrr = 0.0
        for i, r_id in enumerate(retrieved_ids[:10]):
            if r_id in expected_docs:
                mrr = 1.0 / (i + 1)
                break
        mrrs.append(mrr)
        
    return {
        "Config": config,
        "Recall@10": round(stats.mean(recalls), 3),
        "MRR": round(stats.mean(mrrs), 3),
        "P50": round(percentile(latencies, 50), 2),
        "P70": round(percentile(latencies, 70), 2),
        "P100": round(percentile(latencies, 100), 2)
    }

if __name__ == "__main__":
    index = HybridIndex(str(INDEX_DIR))
    queries = load_queries()
    qrels = load_qrels()
    
    configs = [
        "Dense only",
        "BM25 only",
        "Dense + BM25 + RRF",
        "Hybrid + Reranker Top 10",
        "Hybrid + Reranker Top 5",
        "Hybrid + Reranker Top 3"
    ]
    
    results = []
    print(f"{'Configuration':<25} | {'Recall@10':<9} | {'MRR':<5} | {'P50':<6} | {'P70':<6} | {'P100':<6}")
    print("-" * 70)
    for c in configs:
        res = evaluate(index, queries, qrels, c)
        results.append(res)
        print(f"{res['Config']:<25} | {res['Recall@10']:<9.3f} | {res['MRR']:<5.3f} | {res['P50']:<6.2f} | {res['P70']:<6.2f} | {res['P100']:<6.2f}")
    
    out_dir = EVAL_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "quality_vs_latency_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
