import json
import time
import os
from collections import defaultdict
from backend.rag.generation.generation import get_generator
from backend.rag.retrieval.retrieval import HybridIndex
from backend.rag.reranking.rerank import rerank as rerank_chunks
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


def run_sweep():
    index = HybridIndex(str(INDEX_DIR))
    generator = get_generator("groq")
    queries = build_query_set(10)
    
    top_n_configs = [1, 2, 3]
    max_tokens_configs = [32, 48, 64, 96]
    
    results_report = {}
    
    print("Starting A/B test sweep...")
    for top_n in top_n_configs:
        for mt in max_tokens_configs:
            print(f"Testing top_n={top_n}, max_tokens={mt}...")
            
            # Since GroqGenerator doesn't accept max_tokens in generate(), we'll monkey-patch it for the test
            # or just change it if it's accessible.
            # It's hardcoded to 40 right now in generation.py.
            
            # Let's write a small patch for this test script
            def patch_generate(self, query, context_chunks):
                # (Same as generation.py but with custom max_tokens)
                import time
                from groq import Groq
                client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
                sources = "\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(context_chunks))
                system = "Answer using ONLY the sources below. Cite like [1]. ONE short sentence."
                messages = [{"role": "system", "content": system}, {"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {query}"}]
                
                t_req = time.perf_counter()
                try:
                    msg = client.chat.completions.create(model=self.model, max_tokens=mt, messages=messages, temperature=0.0)
                    content = msg.choices[0].message.content
                except Exception:
                    time.sleep(2)
                    return ""
                
                self.last_telemetry = {
                    "generation_total_ms": (time.perf_counter() - t_req)*1000,
                    "input_tokens": msg.usage.prompt_tokens,
                    "output_tokens": msg.usage.completion_tokens
                }
                return str(content) if content else ""
                
            generator.generate = patch_generate.__get__(generator)
            
            timings = []
            retrieval_res = []
            
            for q_obj in queries:
                q = q_obj["query"]
                candidates = index.hybrid_search(q, top_n=5)
                top_chunks = rerank_chunks(q, candidates, top_n=top_n, adaptive=True)
                
                generator.generate(q, top_chunks)
                timings.append(generator.last_telemetry["generation_total_ms"])
                
                retrieval_res.append({
                    "query": q,
                    "expected_chunk_ids": q_obj.get("expected_chunk_ids", []),
                    "retrieved_ids": [c["chunk_id"] for c in top_chunks]
                })
                time.sleep(1)
                
            metrics = calc_metrics(retrieval_res)
            
            config_key = f"top_n={top_n}_max_tokens={mt}"
            results_report[config_key] = {
                "P50_latency": round(percentile(timings, 50), 2),
                "P95_latency": round(percentile(timings, 95), 2),
                "MRR": round(metrics["mrr"], 4),
                "Recall@1": round(metrics["recall@1"], 4)
            }
            print(f"  P50: {results_report[config_key]['P50_latency']}ms, MRR: {results_report[config_key]['MRR']}")
            
    print(json.dumps(results_report, indent=2))
    with open("evaluation/results/prompt_sweep.json", "w") as f:
        json.dump(results_report, f, indent=2)

if __name__ == "__main__":
    os.makedirs("evaluation/results", exist_ok=True)
    run_sweep()
