import json
import numpy as np

with open("evaluation/results/latency_results.json") as f:
    results = json.load(f)

stages = ["retrieval_ms", "rerank_ms", "generation_ms", "guardrail_ms", "total_ms"]
stats = {}

for stage in stages:
    times = [r.get(stage, 0) for r in results]
    if times:
        stats[stage] = {
            "P50": np.percentile(times, 50),
            "P70": np.percentile(times, 70),
            "P100": np.max(times)
        }

for k, v in stats.items():
    print(f"{k}: P50={v['P50']:.2f}ms, P70={v['P70']:.2f}ms, P100={v['P100']:.2f}ms")
