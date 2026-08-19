import os
import json
import statistics as stats
from backend.rag.generation.generation import get_generator
from evaluation.benchmark_canonical import build_query_set, percentile, AdaptivePipeline

def main():
    pipeline = AdaptivePipeline()
    generator = get_generator()
    queries = build_query_set(20)
    
    print("Warming up...")
    warmup_q = "warmup query"
    warmup_c, _ = pipeline.run(warmup_q, top_n=2)
    try:
        generator.generate(warmup_q, warmup_c)
    except:
        pass
    
    print("Running Forensics (20 queries)...")
    telemetry_data = []
    
    for q_obj in queries:
        q = q_obj["query"]
        top_chunks, _ = pipeline.run(q, top_n=2)
        try:
            generator.generate(q, top_chunks)
            telemetry_data.append(generator.last_telemetry.copy())
        except Exception as e:
            print(f"Error: {e}")
            
    if not telemetry_data:
        print("No telemetry data captured.")
        return
        
    metrics = [
        "generation_setup_ms", 
        "TTFT_ms", 
        "stream_duration_ms", 
        "response_parse_ms", 
        "generation_total_ms",
        "input_tokens",
        "output_tokens",
        "retry_count"
    ]
    
    print("\n========================================================")
    print("GENERATION LATENCY BREAKDOWN")
    print("========================================================")
    print(f"{'Metric':<25} {'P50':>8} {'P70':>8} {'P95':>8} {'P99':>8} {'P100':>8}")
    print("-" * 65)
    
    for m in metrics:
        vals = [d.get(m, 0.0) for d in telemetry_data]
        p50 = percentile(vals, 50)
        p70 = percentile(vals, 70)
        p95 = percentile(vals, 95)
        p99 = percentile(vals, 99)
        p100 = percentile(vals, 100)
        print(f"{m:<25} {p50:>8.2f} {p70:>8.2f} {p95:>8.2f} {p99:>8.2f} {p100:>8.2f}")

if __name__ == "__main__":
    main()
