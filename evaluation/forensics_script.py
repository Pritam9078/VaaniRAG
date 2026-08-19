import json
from backend.rag.generation.generation import get_generator
from evaluation.benchmark_canonical import build_query_set, AdaptivePipeline

def run_forensics():
    pipeline = AdaptivePipeline()
    generator = get_generator()
    queries = build_query_set(15)
    
    print("Warming up...")
    try:
        warmup_c, _ = pipeline.run("warmup", top_n=2)
        generator.generate("warmup", warmup_c)
    except:
        pass
        
    results = []
    
    print("Running 15 consecutive requests to trigger tail latency...")
    for q_obj in queries[:15]:
        q = q_obj["query"]
        top_chunks, _ = pipeline.run(q, top_n=2)
        try:
            generator.generate(q, top_chunks)
        except Exception as e:
            print(f"Failed query: {e}")
        
        telemetry = generator.last_telemetry.copy()
        results.append(telemetry)
        
    # Sort by generation_total_ms descending to find the slowest 10
    results.sort(key=lambda x: x.get("generation_total_ms", 0), reverse=True)
    slowest_10 = results[:10]
    
    output_path = "evaluation/results/generation_forensics.json"
    with open(output_path, "w") as f:
        json.dump(slowest_10, f, indent=2)
        
    print(f"Saved generation_forensics.json with {len(slowest_10)} slowest requests.")
    
if __name__ == "__main__":
    run_forensics()
