import json
import time
from collections import defaultdict
from pathlib import Path

from backend.rag.generation.generation import get_generator
from backend.rag.retrieval.retrieval import HybridIndex
from evaluation.benchmark import build_query_set, percentile, ROOT, INDEX_DIR

def run_frozen_benchmark():
    config_path = "evaluation/results/benchmark_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
        
    index = HybridIndex(str(INDEX_DIR))
    generator = get_generator("groq")
    
    n = config["query_count"]
    # Add one query for warm-up
    queries = build_query_set(n + 1)
    
    print("Running WARM-UP query...")
    cold_q = queries[0]["query"]
    c_cold = index.hybrid_search(cold_q, top_n=config["top_k_retrieval"])
    generator.generate(cold_q, c_cold)
    
    metrics = defaultdict(list)
    slowest_requests = []
    
    print(f"Running {n} WARM queries for strict TTFT analysis...")
    
    for idx, q_obj in enumerate(queries[1:]):
        q = q_obj["query"]
        candidates = index.hybrid_search(q, top_n=config["top_k_retrieval"])
        
        try:
            generator.generate(q, candidates)
            tel = generator.last_telemetry
        except Exception as e:
            tel = {
                "generation_total_ms": 0,
                "error": str(e)
            }
            
        for k, v in tel.items():
            metrics[k].append(v)
            
        slowest_requests.append({
            "query_id": idx,
            "total_latency": tel.get("generation_total_ms", 0),
            "client_prepare": tel.get("client_prepare_ms", 0),
            "prompt_build": tel.get("prompt_build_ms", 0),
            "TTFT": tel.get("TTFT_ms", 0),
            "stream_gen": tel.get("stream_generation_ms", 0),
            "retries": tel.get("retry_count", 0),
            "error": tel.get("error", "None")
        })
        time.sleep(1) # Prevent rate limiting
        
    report = {}
    for k, vals in metrics.items():
        if not vals or isinstance(vals[0], str): continue
        report[k] = {
            "P50": round(percentile(vals, 50), 2),
            "P70": round(percentile(vals, 70), 2),
            "P95": round(percentile(vals, 95), 2),
            "P99": round(percentile(vals, 99), 2),
            "P100": round(percentile(vals, 100), 2)
        }
        
    slowest_requests.sort(key=lambda x: x["total_latency"], reverse=True)
    
    Path("evaluation/results").mkdir(parents=True, exist_ok=True)
    
    with open("evaluation/results/generation_forensics.md", "w") as f:
        f.write("# Generation Forensics Report (FROZEN CONFIG)\n\n")
        f.write("------------------------------------------------------------\n")
        f.write("Metric                  P50      P70      P95      P99      P100\n")
        f.write("------------------------------------------------------------\n")
        
        keys_to_print = [
            "client_prepare_ms", "prompt_build_ms", 
            "TTFT_ms", "stream_generation_ms", "model_generation_ms",
            "response_parse_ms", "generation_total_ms"
        ]
        for k in keys_to_print:
            if k in report:
                d = report[k]
                f.write(f"{k:<22} {d['P50']:<8.2f} {d['P70']:<8.2f} {d['P95']:<8.2f} {d['P99']:<8.2f} {d['P100']:<8.2f}\n")
                
        f.write("------------------------------------------------------------\n\n")
        f.write("## TOP SLOWEST GENERATION REQUESTS\n")
        f.write("| Query ID | Total (ms) | ClientPrep | Prompt | TTFT | StreamGen | Retries | Error |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---|\n")
        
        for req in slowest_requests[:10]:
            f.write(f"| {req['query_id']} | {req['total_latency']:.1f} | {req['client_prepare']:.2f} | {req['prompt_build']:.2f} | {req['TTFT']:.1f} | {req['stream_gen']:.1f} | {req['retries']} | {req['error']} |\n")

if __name__ == "__main__":
    run_frozen_benchmark()
