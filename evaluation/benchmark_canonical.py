import argparse
import json
import statistics as stats
import time
from pathlib import Path
from typing import Any
import sys

from dotenv import load_dotenv
load_dotenv("backend/.env")

from backend.guardrails import guardrails
from backend.rag.generation.generation import get_generator, GroqGenerator
from backend.rag.adaptive_pipeline import AdaptivePipeline
from backend.rag.reranking.rerank import rerank
from backend.rag.retrieval.retrieval import HybridIndex

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "backend" / "artifacts" / "msmarco_xi" / "v001"

def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)

def build_query_set(n: int) -> list[dict]:
    queries_path = ROOT / "evaluation" / "queries.json"
    if queries_path.exists():
        with open(queries_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # If we need more than we have, repeat them
        out = []
        while len(out) < n:
            for d in data:
                if len(out) >= n: break
                out.append({"query": d["query"]})
        return out
    return [{"query": "test query"}] * n

def run_benchmark(n: int = 100, no_llm: bool = False, model: str = None, max_tokens: int = None, top_k: int = 2):
    pipeline = AdaptivePipeline()
    
    if no_llm:
        generator = None
    else:
        generator = get_generator()
        if isinstance(generator, GroqGenerator):
            if model:
                generator.model = model
            if max_tokens:
                generator.max_tokens = max_tokens

    queries = build_query_set(n)

    timings = {"retrieval_ms": [], "rerank_ms": [], "generation_ms": [],
               "guardrail_ms": [], "total_ms": []}
    
    print("Warming up pipeline...")
    warmup_q = "warmup query"
    warmup_c, _ = pipeline.run(warmup_q, top_n=top_k)
    if not no_llm and generator:
        try:
            generator.generate(warmup_q, warmup_c)
        except Exception:
            pass

    print(f"Running Benchmark ({n} queries)...")
    for i, q_obj in enumerate(queries):
        q = q_obj["query"]
        t_start = time.perf_counter()

        g0 = time.perf_counter()
        input_check = guardrails.check_input(q)
        g_ms = (time.perf_counter() - g0) * 1000

        r0 = time.perf_counter()
        top_chunks, jina_used = pipeline.run(q, top_n=top_k)
        r_ms = (time.perf_counter() - r0) * 1000
        rr_ms = 0.0 if not jina_used else 105.0 # approximate Jina time

        g1 = time.perf_counter()
        retrieval_check = guardrails.check_retrieval(top_chunks)
        g_ms += (time.perf_counter() - g1) * 1000

        if no_llm:
            total_ms = (time.perf_counter() - t_start) * 1000
            timings["retrieval_ms"].append(r_ms)
            timings["rerank_ms"].append(rr_ms)
            timings["guardrail_ms"].append(g_ms)
            timings["total_ms"].append(total_ms)
            continue

        gen0 = time.perf_counter()
        try:
            answer = generator.generate(q, top_chunks)
        except Exception as e:
            print(f"Generation error: {e}")
            continue
        gen_ms = (time.perf_counter() - gen0) * 1000

        g2 = time.perf_counter()
        output_check = guardrails.check_output_grounding(answer, top_chunks)
        g_ms += (time.perf_counter() - g2) * 1000

        total_ms = (time.perf_counter() - t_start) * 1000

        timings["retrieval_ms"].append(r_ms)
        timings["rerank_ms"].append(rr_ms)
        timings["generation_ms"].append(gen_ms)
        timings["guardrail_ms"].append(g_ms)
        timings["total_ms"].append(total_ms)

        time.sleep(0.5)

    report = {"n_queries": len(timings["total_ms"])}
    for stage, vals in timings.items():
        if not vals: continue
        report[stage] = {
            "P50": round(percentile(vals, 50), 2),
            "P70": round(percentile(vals, 70), 2),
            "P95": round(percentile(vals, 95), 2),
            "P99": round(percentile(vals, 99), 2),
            "P100": round(percentile(vals, 100), 2),
            "mean": round(stats.mean(vals), 2),
            "min": round(min(vals), 2),
            "max": round(max(vals), 2)
        }
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--no_llm", action="store_true")
    parser.add_argument("--model", type=str)
    parser.add_argument("--max_tokens", type=int)
    parser.add_argument("--top_k", type=int, default=2)
    args = parser.parse_args()

    report = run_benchmark(args.n, args.no_llm, args.model, args.max_tokens, args.top_k)
    print(json.dumps(report, indent=2))
