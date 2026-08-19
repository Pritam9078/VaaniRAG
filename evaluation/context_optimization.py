import time
from backend.rag.generation.generation import GroqGenerator
from evaluation.benchmark_canonical import build_query_set, AdaptivePipeline

def test_context_sizes():
    pipeline = AdaptivePipeline()
    queries = build_query_set(15)
    generator = GroqGenerator(model="allam-2-7b", max_tokens=32)
    
    for top_k in [1, 2, 3]:
        print(f"\n--- Testing Top-K = {top_k} ---")
        try:
            warmup_c, _ = pipeline.run("warmup", top_n=top_k)
            generator.generate("warmup", warmup_c)
        except Exception:
            pass
            
        ttft_list = []
        gen_list = []
        
        for i in range(5):
            q = queries[i]["query"]
            top_chunks, _ = pipeline.run(q, top_n=top_k)
            try:
                generator.generate(q, top_chunks)
                ttft_list.append(generator.last_telemetry.get("TTFT_ms", 0.0))
                gen_list.append(generator.last_telemetry.get("generation_total_ms", 0.0))
            except Exception as e:
                pass
                
        if ttft_list:
            print(f"Avg TTFT: {sum(ttft_list)/len(ttft_list):.2f} ms")
            print(f"Avg Total Gen: {sum(gen_list)/len(gen_list):.2f} ms")

if __name__ == "__main__":
    test_context_sizes()
