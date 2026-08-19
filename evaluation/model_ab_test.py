import json
import time
from backend.rag.generation.generation import GroqGenerator
from evaluation.benchmark_canonical import build_query_set, AdaptivePipeline, percentile

def test_model(model_name):
    print(f"\n--- Testing Model: {model_name} ---")
    pipeline = AdaptivePipeline()
    generator = GroqGenerator(model=model_name, max_tokens=32)
    
    # Init client
    generator._init_client()
    
    # Delay to let TPM clear
    print("Waiting 15s for TPM clearance...")
    time.sleep(15)
    
    queries = build_query_set(20)
    gen_totals = []
    r_429s = 0
    
    for q_obj in queries:
        q = q_obj["query"]
        top_chunks, _ = pipeline.run(q, top_n=2)
        try:
            generator.generate(q, top_chunks)
        except Exception:
            pass
            
        t = generator.last_telemetry
        if t.get("status_code") == 429:
            r_429s += 1
            
        gen_totals.append(t.get("generation_total_ms", 0))
        time.sleep(1) # Pace at 60 RPM
        
    return {
        "model": model_name,
        "P50": percentile(gen_totals, 50),
        "P95": percentile(gen_totals, 95),
        "429_count": r_429s
    }

def main():
    models = ["allam-2-7b", "gemma2-9b-it", "mixtral-8x7b-32768"]
    results = []
    for m in models:
        try:
            results.append(test_model(m))
        except Exception as e:
            print(f"Failed {m}: {e}")
            
    print("\nRESULTS:", json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
