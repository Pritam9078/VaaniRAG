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

from backend.app.generation import get_generator
from backend.app.rerank import rerank as rerank_chunks
from backend.app.retrieval import HybridIndex

from backend.app import guardrails

ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = ROOT / "backend" / "data" / "index"
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


def build_query_set(n: int) -> list[str]:
    rng = random.Random(42)
    out = []
    while len(out) < n:
        out.append(rng.choice(BASE_QUERIES))
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


def run_benchmark(n: int = 120):
    index = HybridIndex(str(INDEX_DIR))
    generator = get_generator("extractive")
    queries = build_query_set(n)

    timings = {"retrieval_ms": [], "rerank_ms": [], "generation_ms": [],
               "guardrail_ms": [], "total_ms": []}
    refused = 0
    answered = 0

    for q in queries:
        t_start = time.perf_counter()

        g0 = time.perf_counter()
        input_check = guardrails.check_input(q)
        g_ms = (time.perf_counter() - g0) * 1000
        if not input_check.allowed:
            refused += 1
            continue

        r0 = time.perf_counter()
        candidates = index.hybrid_search(q, top_n=8)
        r_ms = (time.perf_counter() - r0) * 1000

        rr0 = time.perf_counter()
        top_chunks = rerank_chunks(q, candidates, top_n=5)
        rr_ms = (time.perf_counter() - rr0) * 1000

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

    report = {"n_queries": n, "answered": answered, "refused": refused}
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
    args = parser.parse_args()

    report = run_benchmark(args.n)
    print(json.dumps(report, indent=2))

    out_path = ROOT / "backend" / "data" / "benchmark_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report to {out_path}")
