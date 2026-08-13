import json
import math
from pathlib import Path
from collections import defaultdict

def calculate_metrics(results_data):
    """
    Calculate Recall@1, Recall@5, Recall@10, MRR, and nDCG for the retrieval results.
    results_data: list of dicts with 'retrieved_ids' (list of str) and 'expected_id' (str)
    """
    total = len(results_data)
    if total == 0:
        return {}

    recall_1 = 0
    recall_5 = 0
    recall_10 = 0
    mrr_sum = 0.0
    ndcg_sum = 0.0

    for item in results_data:
        expected_ids = set(item.get("expected_chunk_ids", []))
        retrieved = item.get("retrieved_ids", [])
        
        # Recall
        if any(rid in expected_ids for rid in retrieved[:1]):
            recall_1 += 1
        if any(rid in expected_ids for rid in retrieved[:5]):
            recall_5 += 1
        if any(rid in expected_ids for rid in retrieved[:10]):
            recall_10 += 1
            
        # MRR
        # Find the rank of the first relevant document
        first_rel_rank = next((i + 1 for i, rid in enumerate(retrieved) if rid in expected_ids), None)
        if first_rel_rank is not None:
            mrr_sum += 1.0 / first_rel_rank
            ndcg_sum += 1.0 / math.log2(first_rel_rank + 1)

    return {
        "Recall@1": round(recall_1 / total, 4),
        "Recall@5": round(recall_5 / total, 4),
        "Recall@10": round(recall_10 / total, 4),
        "MRR": round(mrr_sum / total, 4),
        "nDCG": round(ndcg_sum / total, 4)
    }

if __name__ == "__main__":
    print("Run this script using the evaluation framework in benchmark.py")
