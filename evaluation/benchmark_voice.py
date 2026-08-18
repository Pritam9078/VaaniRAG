import argparse
import json
import time
import statistics as stats
from pathlib import Path

from backend.rag.retrieval.retrieval import HybridIndex
from backend.rag.reranking.rerank import rerank
from backend.rag.generation.generation import get_generator
from backend.guardrails.guardrails import check_output_grounding
from backend.voice.stt import get_stt

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "backend" / "artifacts" / "msmarco_xi" / "v001"
EVAL_DIR = ROOT / "evaluation"

def build_query_set(n: int) -> list[str]:
    with open(EVAL_DIR / "queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
    return [q["query"] for q in queries[:n]]

def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)

def run_voice_benchmark(n: int = 10):
    stt = get_stt("mock") # 0.5s delay
    index = HybridIndex(str(INDEX_DIR))
    generator = get_generator()
    queries = build_query_set(n)

    timings = {"stt_ms": [], "retrieval_ms": [], "rerank_ms": [], "generation_ms": [], "total_ms": []}

    print(f"Running End-to-End Voice Pipeline Benchmark ({n} queries)...")
    for idx, q in enumerate(queries):
        t_start = time.perf_counter()
        
        # 1. Simulate STT (audio -> text)
        audio_bytes = b"fake_audio_data"
        t0 = time.perf_counter()
        transcript, stt_latency = stt.transcribe(audio_bytes) 
        
        # Override the mock transcript with the real query for the rest of pipeline
        transcript = q 
        
        stt_ms = (time.perf_counter() - t0) * 1000

        # 2. Retrieval
        r0 = time.perf_counter()
        candidates = index.hybrid_search(transcript, top_n=5)
        r_ms = (time.perf_counter() - r0) * 1000

        # 3. Rerank
        rr0 = time.perf_counter()
        top_chunks = rerank(transcript, candidates, top_n=3)
        rr_ms = (time.perf_counter() - rr0) * 1000

        # 4. Generation
        g0 = time.perf_counter()
        answer = generator.generate(transcript, top_chunks)
        gen_ms = (time.perf_counter() - g0) * 1000

        # 5. Guardrail check
        check_output_grounding(answer, top_chunks)

        total_ms = (time.perf_counter() - t_start) * 1000
        
        timings["stt_ms"].append(stt_ms)
        timings["retrieval_ms"].append(r_ms)
        timings["rerank_ms"].append(rr_ms)
        timings["generation_ms"].append(gen_ms)
        timings["total_ms"].append(total_ms)

        time.sleep(1.5)

    report = {
        "n_queries": n,
        "stt_ms": {"P50": round(percentile(timings["stt_ms"], 50), 3)},
        "retrieval_ms": {"P50": round(percentile(timings["retrieval_ms"], 50), 3)},
        "rerank_ms": {"P50": round(percentile(timings["rerank_ms"], 50), 3)},
        "generation_ms": {"P50": round(percentile(timings["generation_ms"], 50), 3)},
        "total_ms": {
            "P50": round(percentile(timings["total_ms"], 50), 3),
            "P100": round(percentile(timings["total_ms"], 100), 3)
        }
    }
    
    return report

if __name__ == "__main__":
    report = run_voice_benchmark(10)
    print(json.dumps(report, indent=2))
    
    out_dir = EVAL_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "voice_e2e_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
