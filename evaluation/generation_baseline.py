import json
import time
from collections import defaultdict
from backend.rag.generation.generation import get_generator
from evaluation.benchmark import build_query_set, percentile, ROOT

def run_baseline(n=5):
    queries = build_query_set(n)
    generator = get_generator("groq")
    
    # Warmup
    generator.generate("hello", [{"text": "hello world", "chunk_id": "1", "language": "en"}])
    
    metrics = defaultdict(list)
    print(f"Running {n} baseline generation requests...")
    
    for i, q in enumerate(queries):
        context = [{"text": "some context here", "chunk_id": "1", "language": "en"}]
        # Simulate retrieval output format roughly
        _ = generator.generate(q["query"], context)
        tel = generator.last_telemetry
        
        for k, v in tel.items():
            metrics[k].append(v)
            
        time.sleep(2)
        
    report = {}
    for k, vals in metrics.items():
        if not vals: continue
        report[k] = {
            "P50": round(percentile(vals, 50), 3),
            "P95": round(percentile(vals, 95), 3),
            "P100": round(percentile(vals, 100), 3)
        }
        
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    run_baseline(10)
