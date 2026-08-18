import json
import time
from collections import defaultdict
from pathlib import Path

from backend.rag.generation.generation import get_generator
from backend.rag.reranking.rerank import rerank as rerank_chunks
from backend.rag.retrieval.retrieval import HybridIndex
from evaluation.benchmark import build_query_set, percentile, ROOT, INDEX_DIR

def calc_metrics(results):
    metrics = {"recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0, "ndcg@5": 0.0, "ndcg@10": 0.0}
    if not results: return metrics
    
    import math
    def dcg(rels, k):
        return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(rels[:k]))
    
    for r in results:
        expected = set(r["expected_chunk_ids"])
        retrieved = r["retrieved_ids"]
        if not expected: continue
            
        rels = [1 if c in expected else 0 for c in retrieved]
        metrics["recall@1"] += 1 if sum(rels[:1]) > 0 else 0
        metrics["recall@5"] += 1 if sum(rels[:5]) > 0 else 0
        metrics["recall@10"] += 1 if sum(rels[:10]) > 0 else 0
        
        for i, rel in enumerate(rels):
            if rel == 1:
                metrics["mrr"] += 1.0 / (i + 1)
                break
                
        idcg5 = dcg([1] * min(len(expected), 5), 5)
        if idcg5 > 0: metrics["ndcg@5"] += dcg(rels, 5) / idcg5
        idcg10 = dcg([1] * min(len(expected), 10), 10)
        if idcg10 > 0: metrics["ndcg@10"] += dcg(rels, 10) / idcg10

    n = len([r for r in results if r["expected_chunk_ids"]])
    if n > 0:
        for k in metrics:
            metrics[k] /= n
    return metrics

def run_comprehensive(n=10):
    index = HybridIndex(str(INDEX_DIR))
    generator = get_generator("groq")
    queries = build_query_set(n + 1) # +1 for cold start
    
    timings = defaultdict(list)
    retrieval_results = []
    
    jina_invocations = 0
    total_queries = 0

    print("Running COLD START query...")
    cold_q = queries[0]["query"]
    t0_cold = time.perf_counter()
    c_cold = index.hybrid_search(cold_q, top_n=5)
    r_cold = time.perf_counter()
    rerank_chunks(cold_q, c_cold, top_n=3, adaptive=True)
    rr_cold = time.perf_counter()
    generator.generate(cold_q, c_cold)
    gen_cold = time.perf_counter()
    
    cold_metrics = {
        "retrieval_ms": (r_cold - t0_cold) * 1000,
        "rerank_ms": (rr_cold - r_cold) * 1000,
        "generation_ms": (gen_cold - rr_cold) * 1000,
        "total_ms": (gen_cold - t0_cold) * 1000
    }
    
    print(f"Benchmarking {n} WARM queries...")
    for q_obj in queries[1:]:
        q = q_obj["query"]
        expected = q_obj.get("expected_chunk_ids", [])
        total_queries += 1
        
        t_start = time.perf_counter()
        
        r0 = time.perf_counter()
        candidates = index.hybrid_search(q, top_n=5)
        timings["retrieval_ms"].append((time.perf_counter() - r0) * 1000)
        
        is_confident = False
        if candidates and candidates[0].get("dense_rank") == 0 and candidates[0].get("sparse_rank") == 0:
            is_confident = True
        if not is_confident:
            jina_invocations += 1
            
        rr0 = time.perf_counter()
        top_chunks = rerank_chunks(q, candidates, top_n=3, adaptive=True)
        timings["rerank_ms"].append((time.perf_counter() - rr0) * 1000)
        
        retrieval_results.append({
            "query": q,
            "expected_chunk_ids": expected,
            "retrieved_ids": [c["chunk_id"] for c in top_chunks]
        })
        
        answer = generator.generate(q, top_chunks)
        
        tel = generator.last_telemetry
        for k,v in tel.items():
            timings[k].append(v)
            
        timings["total_ms"].append((time.perf_counter() - t_start) * 1000)
        time.sleep(1.5) # rate limit mitigation

    metrics = calc_metrics(retrieval_results)
    
    report_latency = {}
    for stage, vals in timings.items():
        if vals:
            report_latency[stage] = {
                "P50": round(percentile(vals, 50), 3),
                "P70": round(percentile(vals, 70), 3),
                "P95": round(percentile(vals, 95), 3),
                "P100": round(percentile(vals, 100), 3),
            }
            
    Path("evaluation/results").mkdir(parents=True, exist_ok=True)
    
    # Save Report
    with open("evaluation/results/optimization_report.md", "w") as f:
        f.write("# VaaniRAG Optimization Report\n\n")
        f.write("## 1. BEFORE vs AFTER Latency\n\n")
        f.write("| Metric | Before | After | Improvement |\n")
        f.write("|---|---:|---:|---:|\n")
        
        r_p50 = report_latency['retrieval_ms']['P50']
        rr_p50 = report_latency['rerank_ms']['P50']
        g_p50 = report_latency['generation_total_ms']['P50']
        t_p50 = report_latency['total_ms']['P50']
        t_p95 = report_latency['total_ms']['P95']
        
        f.write(f"| Retrieval P50 | 16.12 ms | {r_p50} ms | {round((16.12-r_p50)/16.12*100, 1)}% |\n")
        f.write(f"| Rerank P50 | 110.88 ms | {rr_p50} ms | {round((110.88-rr_p50)/110.88*100, 1)}% |\n")
        f.write(f"| Generation P50 | 1212.43 ms | {g_p50} ms | {round((1212.43-g_p50)/1212.43*100, 1)}% |\n")
        f.write(f"| Total P50 | 1342.39 ms | {t_p50} ms | {round((1342.39-t_p50)/1342.39*100, 1)}% |\n")
        f.write(f"| P95 | 1858.11 ms | {t_p95} ms | {round((1858.11-t_p95)/1858.11*100, 1)}% |\n")
        f.write(f"| P100 | 1858.11 ms | {report_latency['total_ms']['P100']} ms | - |\n")
        
        pct = (jina_invocations / total_queries) * 100 if total_queries else 0
        f.write(f"| Jina invocation | 100% | {pct}% | - |\n\n")
        
        f.write("## 2. Stage-by-Stage Latency (Warm)\n")
        for stage, data in report_latency.items():
            f.write(f"- **{stage}**: P50={data['P50']}ms, P95={data['P95']}ms\n")
            
        f.write("\n## 3. TTFT (Streaming Enabled)\n")
        f.write(f"- **TTFT P50**: {report_latency.get('TTFT_ms', {}).get('P50', 'N/A')}ms\n")
        
        f.write("\n## 4. Cold Start vs Warm Latency\n")
        f.write(f"- **Cold Start E2E**: {round(cold_metrics['total_ms'], 2)}ms\n")
        f.write(f"- **Warm Start P50**: {t_p50}ms\n")
        
        f.write("\n## 5. Retrieval Quality (Adaptive)\n")
        for k, v in metrics.items():
            f.write(f"- **{k}**: {v:.4f}\n")
            
        f.write("\n## 6. Honest Assessment\n")
        f.write("Achieving <200ms E2E is heavily bottlenecked by Groq's TTFT and model generation time, which natively sits at ~500ms even with connection pooling and max_tokens=64. The application is streaming, so perceived latency is lower, but full E2E is bound by remote network physics.")

    # Save Config
    with open("evaluation/results/optimization_config.json", "w") as f:
        json.dump({
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "faiss_index": "l2",
            "bm25": "okapi",
            "rrf": "k=60",
            "reranker": "jinaai/jina-reranker-v1-tiny-en",
            "rerank_top_k": 3,
            "confidence_threshold": "dense_rank=0 and sparse_rank=0",
            "llm_model": "openai/gpt-oss-120b",
            "max_tokens": 64,
            "top_n_context": 3,
            "streaming": True,
            "benchmark_queries": 10,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }, f, indent=2)

    print("Done! Check evaluation/results/optimization_report.md")

if __name__ == "__main__":
    run_comprehensive(20)
