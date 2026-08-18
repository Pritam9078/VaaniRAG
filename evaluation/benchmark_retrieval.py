import argparse
import json
import time
import statistics as stats
from pathlib import Path
from backend.rag.retrieval.retrieval import HybridIndex

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "backend" / "artifacts" / "msmarco_xi" / "v001"
EVAL_DIR = ROOT / "evaluation"

def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)

def calculate_metrics(results, top_k):
    # results is a list of dicts: {"query": q, "retrieved_ids": [...], "expected_ids": [...]}
    recalls = []
    mrrs = []
    for r in results:
        expected = set(r["expected_ids"])
        retrieved = r["retrieved_ids"][:top_k]
        
        # Recall
        hits = sum(1 for cid in retrieved if cid in expected)
        recalls.append(hits / len(expected) if expected else 0.0)
        
        # MRR
        mrr = 0.0
        for i, cid in enumerate(retrieved):
            if cid in expected:
                mrr = 1.0 / (i + 1)
                break
        mrrs.append(mrr)
        
    return {
        f"Recall@{top_k}": round(stats.mean(recalls), 4) if recalls else 0.0,
        "MRR": round(stats.mean(mrrs), 4) if mrrs else 0.0
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["dense", "bm25", "hybrid"], required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    # Load queries and qrels
    with open(EVAL_DIR / "queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open(EVAL_DIR / "qrels.json", "r", encoding="utf-8") as f:
        qrels = json.load(f)

    print("Loading Index...")
    index = HybridIndex(str(INDEX_DIR))

    latencies = []
    results = []
    
    print(f"Running {args.mode} benchmark for {len(queries)} queries...")
    for q_obj in queries:
        q = q_obj["query"]
        qid = str(q_obj["query_id"])
        expected = qrels.get(qid, [])
        
        t0 = time.perf_counter()
        if args.mode == "dense":
            results_tuples = index.dense_search(q, k=args.top_k)
            retrieved_ids = [index.chunks[idx]["chunk_id"] for idx, _ in results_tuples]
        elif args.mode == "bm25":
            results_tuples = index.bm25_search(q, k=args.top_k)
            retrieved_ids = [index.chunks[idx]["chunk_id"] for idx, _ in results_tuples]
        elif args.mode == "hybrid":
            # Hybrid search natively uses RRF and returns dict chunks
            chunks = index.hybrid_search(q, top_n=args.top_k)
            retrieved_ids = [c["chunk_id"] for c in chunks]
        t1 = time.perf_counter()
        
        latencies.append((t1 - t0) * 1000)
        results.append({
            "query_id": qid,
            "retrieved_ids": retrieved_ids,
            "expected_ids": expected
        })

    metrics = calculate_metrics(results, args.top_k)
    
    report = {
        "mode": args.mode,
        "top_k": args.top_k,
        "n_queries": len(queries),
        "metrics": metrics,
        "latency_ms": {
            "P50": round(percentile(latencies, 50), 2),
            "P70": round(percentile(latencies, 70), 2),
            "P100": round(percentile(latencies, 100), 2),
            "mean": round(stats.mean(latencies), 2)
        }
    }
    
    out_dir = EVAL_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.mode}_baseline.json"
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(json.dumps(report, indent=2))
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
