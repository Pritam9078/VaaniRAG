import time
from collections import defaultdict
from backend.rag.generation.generation import GroqGenerator
from evaluation.benchmark_canonical import build_query_set, AdaptivePipeline

def run_model_ab():
    pipeline = AdaptivePipeline()
    queries = build_query_set(15)
    
    models = [
        "openai/gpt-oss-20b",
        "qwen/qwen3.6-27b",
        "allam-2-7b"
    ]
    
    print("Testing Models for TTFT (5 queries each)...")
    for model in models:
        print(f"\n--- Testing {model} ---")
        generator = GroqGenerator(model=model, max_tokens=16)
        
        # Warmup
        try:
            warmup_c, _ = pipeline.run("warmup", top_n=1)
            generator.generate("warmup", warmup_c)
        except Exception as e:
            print(f"Warmup failed: {e}")
            continue
            
        ttft_list = []
        for i in range(5):
            q = queries[i]["query"]
            top_chunks, _ = pipeline.run(q, top_n=1)
            try:
                generator.generate(q, top_chunks)
                ttft_list.append(generator.last_telemetry.get("TTFT_ms", 0.0))
            except Exception as e:
                print(f"Error on {model}: {e}")
                
        if ttft_list:
            avg_ttft = sum(ttft_list) / len(ttft_list)
            print(f"Avg TTFT: {avg_ttft:.2f} ms")
        else:
            print("No successful queries.")

if __name__ == "__main__":
    run_model_ab()
