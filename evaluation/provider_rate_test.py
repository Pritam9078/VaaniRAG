import time
import json
from backend.rag.generation.generation import get_generator
from evaluation.benchmark_canonical import build_query_set, AdaptivePipeline, percentile

def test_rate(name, queries, pipeline, generator, sleep_time):
    print(f"\n--- Running: {name} (Delay: {sleep_time}s) ---")
    
    ttfts = []
    gen_totals = []
    r_429s = 0
    retries = 0
    
    for q_obj in queries:
        q = q_obj["query"]
        top_chunks, _ = pipeline.run(q, top_n=2)
        try:
            generator.generate(q, top_chunks)
        except Exception:
            pass
            
        t = generator.last_telemetry
        if t.get("status_code") == 429 or t.get("exception"):
            r_429s += 1
            
        ttfts.append(t.get("TTFT_ms", 0))
        gen_totals.append(t.get("generation_total_ms", 0))
        retries += t.get("retry_count", 0)
        
        time.sleep(sleep_time)
        
    return {
        "rate": f"1 req / {sleep_time}s",
        "requests_per_minute": 60 / (sleep_time or 0.1),
        "P50": percentile(gen_totals, 50),
        "P95": percentile(gen_totals, 95),
        "P99": percentile(gen_totals, 99),
        "P100": percentile(gen_totals, 100),
        "TTFT_P50": percentile(ttfts, 50),
        "TTFT_P95": percentile(ttfts, 95),
        "429_count": r_429s,
        "retry_count": retries
    }

def main():
    pipeline = AdaptivePipeline()
    generator = get_generator()
    
    print("Warming up...")
    try:
        warmup_c, _ = pipeline.run("warmup", top_n=2)
        generator.generate("warmup", warmup_c)
    except:
        pass

    results = []
    
    # Phase 4: Spaced Control (3s delay)
    q_spaced = build_query_set(20)
    res_spaced = test_rate("Spaced Control", q_spaced, pipeline, generator, 3.0)
    results.append(res_spaced)
    
    # Phase 4: Consecutive (0s delay)
    q_consec = build_query_set(20)
    res_consec = test_rate("Consecutive", q_consec, pipeline, generator, 0.0)
    results.append(res_consec)
    
    # Phase 5: Rate Sweep
    results.append(test_rate("Sweep 2.0s", build_query_set(20), pipeline, generator, 2.0))
    results.append(test_rate("Sweep 1.0s", build_query_set(20), pipeline, generator, 1.0))
    results.append(test_rate("Sweep 0.5s", build_query_set(20), pipeline, generator, 0.5))

    with open("evaluation/results/provider_rate_test.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nSaved provider_rate_test.json")

if __name__ == "__main__":
    main()
