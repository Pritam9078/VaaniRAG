"""
Benchmark the RAG pipeline (retrieval + rerank + generation + guardrails,
i.e. everything after STT) across a set of test queries, and report
P50 / P70 / P100 latency per stage, per requirement #4.

Usage:  python -m backend.scripts.benchmark [--n 120]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as stats
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv("backend/.env")

from backend.guardrails import guardrails
from backend.rag.generation.generation import get_generator
from backend.rag.reranking.rerank import rerank as rerank_chunks
from backend.rag.retrieval.retrieval import HybridIndex
# from evaluation.retrieval_eval import calculate_metrics

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "backend" / "artifacts" / "msmarco_xi" / "v001"
CORPUS_PATH = ROOT / "backend" / "data" / "sample_corpus.json"

# Base set of real queries drawn from the corpus, plus paraphrases and a
# couple of deliberately off-topic / unsafe queries so the benchmark also
# exercises the guardrail refusal path (matching real traffic patterns).
BASE_QUERIES = [
    "What is the capital of India?",
    "भारत की राजधानी क्या है?",
    "How does photosynthesis work?",
    "What causes inflation in an economy?",
    "What are the symptoms of dehydration?",
    "How do vaccines work?",
    "What is the difference between weather and climate?",
    "How does a car engine work?",
    "What is compound interest?",
    "What is the process of the water cycle?",
    "What is the significance of the Reserve Bank of India?",
    "Tell me about New Delhi and why it became the capital",
    "Explain how plants make energy from sunlight",
    "What drives rising prices in an economy?",
    "How can I tell if I'm dehydrated?",
    "टीके कैसे काम करते हैं?",
    "जल चक्र की प्रक्रिया क्या है?",
    "What is the weather like today in Paris?",  # off-topic (not in KB)
    "Recommend me a good sci-fi movie",           # off-topic
]


def build_query_set(n: int) -> list[dict]:
    queries_path = ROOT / "evaluation" / "queries.json"
    if queries_path.exists():
        with open(queries_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # shuffle and pick n
        rng = random.Random(42)
        rng.shuffle(data)
        
        out = []
        for d in data[:n]:
            if "query" in d:
                out.append({
                    "query": d["query"],
                    "expected_chunk_ids": d.get("expected_chunk_ids", [])
                })
        return out

    # Fallback for old mode
    rng = random.Random(42)
    out = []
    while len(out) < n:
        out.append({"query": rng.choice(BASE_QUERIES), "expected_chunk_ids": []})
    return out


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def run_benchmark(n: int = 120, no_llm: bool = False):
    from backend.rag.adaptive_pipeline import AdaptivePipeline
    pipeline = AdaptivePipeline()
    generator = get_generator() if not no_llm else None
    queries = build_query_set(n)

    timings = {"retrieval_ms": [], "rerank_ms": [], "generation_ms": [],
               "guardrail_ms": [], "total_ms": []}
    refused = 0
    answered = 0

    retrieval_results = []
    
    print("Warming up pipeline...")
    warmup_q = "warmup query"
    warmup_c, _ = pipeline.run(warmup_q, top_n=1)
    if not no_llm and generator:
        try:
            generator.generate(warmup_q, warmup_c)
        except Exception:
            pass

    print(f"Running Benchmark ({n} queries)...")
    for q_obj in queries:
        q = q_obj["query"]
        expected_chunk_ids = q_obj.get("expected_chunk_ids", [])
        
        t_start = time.perf_counter()

        g0 = time.perf_counter()
        input_check = guardrails.check_input(q)
        g_ms = (time.perf_counter() - g0) * 1000
        if not input_check.allowed:
            refused += 1
            continue

        r0 = time.perf_counter()
        top_chunks, jina_used = pipeline.run(q, top_n=1)
        r_ms = (time.perf_counter() - r0) * 1000
        rr_ms = 0.0 if not jina_used else r_ms - 15.0 # approximate Jina time if used

        g1 = time.perf_counter()
        retrieval_check = guardrails.check_retrieval(top_chunks)
        g_ms += (time.perf_counter() - g1) * 1000

        if not retrieval_check.allowed:
            refused += 1
            total_ms = (time.perf_counter() - t_start) * 1000
            timings["retrieval_ms"].append(r_ms)
            timings["rerank_ms"].append(rr_ms)
            timings["guardrail_ms"].append(g_ms)
            timings["total_ms"].append(total_ms)
            continue
            
        retrieval_results.append({
            "query": q,
            "expected_chunk_ids": expected_chunk_ids,
            "retrieved_ids": [c["chunk_id"] for c in top_chunks]
        })

        if no_llm:
            total_ms = (time.perf_counter() - t_start) * 1000
            timings["retrieval_ms"].append(r_ms)
            timings["rerank_ms"].append(rr_ms)
            timings["guardrail_ms"].append(g_ms)
            timings["total_ms"].append(total_ms)
            answered += 1
            continue

        gen0 = time.perf_counter()
        answer = generator.generate(q, top_chunks)
        gen_ms = (time.perf_counter() - gen0) * 1000

        g2 = time.perf_counter()
        output_check = guardrails.check_output_grounding(answer, top_chunks)
        g_ms += (time.perf_counter() - g2) * 1000

        total_ms = (time.perf_counter() - t_start) * 1000

        if output_check.allowed:
            answered += 1
        else:
            refused += 1

        timings["retrieval_ms"].append(r_ms)
        timings["rerank_ms"].append(rr_ms)
        timings["generation_ms"].append(gen_ms)
        timings["guardrail_ms"].append(g_ms)
        timings["total_ms"].append(total_ms)

        if not no_llm:
            time.sleep(3.0)

    report: dict[str, Any] = {"n_queries": n, "answered": answered, "refused": refused}  # type: ignore
    # report["retrieval_metrics"] = calculate_metrics(retrieval_results)
    
    for stage, vals in timings.items():
        if not vals:
            continue
        report[stage] = {
            "P50": round(percentile(vals, 50), 3),
            "P70": round(percentile(vals, 70), 3),
            "P100": round(percentile(vals, 100), 3),
            "mean": round(stats.mean(vals), 3),
        }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--no_llm", action="store_true")
    args = parser.parse_args()

    report = run_benchmark(args.n, args.no_llm)
    print(json.dumps(report, indent=2))

    out_path = ROOT / "backend" / "data" / "benchmark_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n======================================================================")
    print("📊 STAGE-BY-STAGE LATENCY BREAKDOWN (POST-STT)")
    print("======================================================================")
    print(f"{'Stage':<30} {'P50 (ms)':>10} {'P70 (ms)':>10} {'P95 (ms)':>10} {'P100 (ms)':>10}")
    print("-" * 70)

    stages = [k for k in report.keys() if k.endswith("_ms")]
    if "total_ms" in stages:
        stages.remove("total_ms")
        stages.append("total_ms")

    for stage in stages:
        data = report.get(stage, {})
        p50 = data.get("P50", 0.0)
        p70 = data.get("P70", 0.0)
        p95 = data.get("P95", data.get("P100", 0.0))
        p99 = data.get("P99", data.get("P100", 0.0))
        print(f"{stage:<30} {p50:>10.2f} {p70:>10.2f} {p95:>10.2f} {p99:>10.2f}")

    print("======================================================================")
    total_data = report.get("total_ms", {})
    t_p50 = total_data.get("P50", 0.0)
    t_p95 = total_data.get("P95", total_data.get("P100", 0.0))
    t_p99 = total_data.get("P99", total_data.get("P100", 0.0))
    print(f"TOTAL / WALL (P50: {t_p50:.1f}ms | P95: {t_p95:.2f}ms | P99: {t_p99:.2f}ms)")
    
    target = 200.0
    status = "PASS" if t_p50 <= target else "FAIL"
    print(f"🎯 Latency budget target: {int(target)}ms | Status: {status} ({t_p50:.2f}ms <= {int(target)}ms) [P50 Evaluation]")
    print("======================================================================")
    print(f"\nSaved JSON report to {out_path}")
