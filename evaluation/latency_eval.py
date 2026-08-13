import json
from pathlib import Path

# ==============================================================================
# LATENCY BOUNDARY DEFINITIONS
# ==============================================================================
# RAG latency =
#   query preprocessing
#   + embedding
#   + dense retrieval
#   + BM25 retrieval
#   + RRF
#   + reranking
#   + generation
#   + grounding/guardrails
#
# Voice E2E latency =
#   STT
#   + RAG latency
#   + network/response overhead
# ==============================================================================

def calculate_percentiles(latencies: list[float]) -> dict[str, float]:
    """Calculate P50, P70, P100 from a list of latencies in ms."""
    if not latencies:
        return {"P50": 0.0, "P70": 0.0, "P100": 0.0}
    
    sorted_lats = sorted(latencies)
    
    # Calculate indices for P50, P70, P100
    p50_idx = int(0.5 * len(sorted_lats))
    p70_idx = int(0.7 * len(sorted_lats))
    
    return {
        "P50": round(sorted_lats[p50_idx], 2),
        "P70": round(sorted_lats[p70_idx], 2),
        "P100": round(sorted_lats[-1], 2)
    }

def print_latency_report(query_count: int, all_stage_latencies: list[dict[str, float]]):
    """
    Prints the latency report matching the HH Goa requirements format.
    """
    print(f"Benchmark queries: {query_count}\n")
    
    if not all_stage_latencies:
        print("No latency data recorded.")
        return
        
    # Aggregate latencies per stage
    stages = ["stt_ms", "retrieval_ms", "rerank_ms", "generation_ms", "guardrail_ms", "total_ms"]
    
    # We will map 'total_ms' to 'Total RAG' and others appropriately
    stage_display_names = {
        "stt_ms": "STT",
        "retrieval_ms": "Retrieval",
        "rerank_ms": "Reranking",
        "generation_ms": "Generation",
        "guardrail_ms": "Guardrails",
        "total_ms": "Total RAG"
    }

    # Gather lists of latencies for each stage
    stage_data = {stage: [] for stage in stages}
    for lat_record in all_stage_latencies:
        for stage in stages:
            stage_data[stage].append(lat_record.get(stage, 0.0))

    # Calculate percentiles for Total RAG latency
    total_rag_lats = stage_data["total_ms"]
    total_percentiles = calculate_percentiles(total_rag_lats)
    
    print("RAG latency")
    print(f"P50  = {total_percentiles['P50']:<6.2f} ms")
    print(f"P70  = {total_percentiles['P70']:<6.2f} ms")
    print(f"P100 = {total_percentiles['P100']:<6.2f} ms\n")

    # Print tabular detailed latency
    print(f"{'Stage':<17} {'P50':<8} {'P70':<8} {'P100':<8}")
    print("-" * 48)
    
    for stage in stages:
        if stage == "stt_ms":
            continue # STT is separate from RAG boundary inner stages
            
        pcts = calculate_percentiles(stage_data[stage])
        name = stage_display_names[stage]
        print(f"{name:<17} {pcts['P50']:<8.2f} {pcts['P70']:<8.2f} {pcts['P100']:<8.2f}")

def run_benchmark():
    queries_path = Path(__file__).parent / "queries.json"
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    if not queries_path.exists():
        print(f"Error: {queries_path} not found.")
        return
        
    with open(queries_path, 'r') as f:
        queries = json.load(f)
        
    if len(queries) < 100:
        print(f"WARNING: Evaluation query set is too small ({len(queries)}). HH Goa benchmark requires 100-500 queries.")
        
    all_stage_latencies = []
    
    print(f"Executing benchmark for {len(queries)} queries against http://localhost:8000/query...")
    
    import requests
    
    for i, q in enumerate(queries):
        query_text = q.get("query", "")
        language = q.get("language", "en")
        
        try:
            resp = requests.post(
                "http://localhost:8000/query",
                json={"query": query_text, "language": language},
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            
            # The API returns latencies in the 'latencies' object
            lats = data.get("latencies", {})
            all_stage_latencies.append(lats)
            
        except requests.exceptions.RequestException as e:
            print(f"Error on query {i}: {e}")
            continue
            
    print_latency_report(len(queries), all_stage_latencies)
    
    # Save results
    with open(results_dir / "latency_results.json", "w") as f:
        json.dump(all_stage_latencies, f, indent=2)
        
    # In a full run, we would also generate latency_report.md here.
    print(f"\nResults saved to {results_dir}")
    
if __name__ == "__main__":
    run_benchmark()
