import json
import time
from collections import defaultdict
from pathlib import Path

from backend.rag.generation.generation import get_generator
from backend.rag.retrieval.retrieval import HybridIndex
from evaluation.benchmark import build_query_set, percentile, ROOT, INDEX_DIR

def run_optimization_sweep():
    index = HybridIndex(str(INDEX_DIR))
    
    contexts = [1, 2, 3]
    tokens = [24, 32, 48]
    
    n = 10
    queries = build_query_set(n + 1)
    
    report = {}
    
    # Warmup Groq SDK
    generator = get_generator("groq")
    generator.model = "qwen/qwen3.6-27b"
    cold_q = queries[0]["query"]
    c_cold = index.hybrid_search(cold_q, top_n=3)
    generator.generate(cold_q, c_cold)
    
    for top_k in contexts:
        for max_t in tokens:
            config_name = f"ctx={top_k}_tokens={max_t}"
            print(f"--- Testing Config: {config_name} ---")
            
            metrics = defaultdict(list)
            
            for q_obj in queries[1:]:
                q = q_obj["query"]
                candidates = index.hybrid_search(q, top_n=top_k)
                
                # Mock setting max tokens, wait, our GroqGenerator hardcodes 64 right now!
                # Let's dynamically inject max_tokens into generator
                generator.max_tokens = max_t
                
                try:
                    generator.generate(q, candidates)
                    tel = generator.last_telemetry
                    for k, v in tel.items():
                        metrics[k].append(v)
                except Exception as e:
                    print(f"Error: {e}")
                time.sleep(1) # Rate limit protection
                
            report[config_name] = {}
            for k in ["TTFT_ms", "stream_generation_ms", "model_generation_ms"]:
                if k in metrics and len(metrics[k]) > 0:
                    report[config_name][k] = {
                        "P50": round(percentile(metrics[k], 50), 2),
                        "P95": round(percentile(metrics[k], 95), 2)
                    }
                    
    print(json.dumps(report, indent=2))
    
if __name__ == "__main__":
    run_optimization_sweep()
