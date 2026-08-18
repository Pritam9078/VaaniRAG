import json
from datasets import load_dataset
from pathlib import Path
import random

def main():
    root = Path(__file__).resolve().parent.parent
    index_dir = root / "backend" / "artifacts" / "msmarco_xi" / "v001"
    eval_dir = root / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    
    print("Loading chunks.json to find valid query_ids...")
    with open(index_dir / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    # Extract query_id from doc_id, e.g. "rec1_1102432_p0_en" -> "1102432"
    valid_query_ids = set()
    query_id_to_chunk_ids = {}
    
    for c in chunks:
        doc_id = c["doc_id"]
        parts = doc_id.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            qid = int(parts[1])
            valid_query_ids.add(qid)
            # The MSMARCO passages have an 'is_selected' field, but since we chunked them,
            # we will just treat any chunk derived from the same query_id as a relevant chunk
            # for that query for benchmarking purposes (since MSMARCO pairs are typically QA pairs).
            if qid not in query_id_to_chunk_ids:
                query_id_to_chunk_ids[qid] = []
            query_id_to_chunk_ids[qid].append(c["chunk_id"])
            
    print(f"Found {len(valid_query_ids)} unique query_ids in chunks.")
    
    print("Streaming MSMARCO-XI from HuggingFace to extract 100 benchmark queries...")
    ds = load_dataset("ai4bharat/MSMARCO-XI", "default", split="validation", streaming=True)
    
    extracted_queries = []
    extracted_qrels = {}
    
    for row in ds:
        qid = row["query_id"]
        if qid in valid_query_ids:
            # Check if it's Hindi
            if row.get("target_lang") == "hin_Deva":
                extracted_queries.append({
                    "query_id": qid,
                    "query": row["query"],  # The Hindi query
                    "eng_query": row.get("Eng_Query", "")
                })
                extracted_qrels[str(qid)] = query_id_to_chunk_ids[qid]
                
                if len(extracted_queries) >= 100:
                    break
                    
    print(f"Extracted {len(extracted_queries)} queries.")
    
    queries_path = eval_dir / "queries.json"
    qrels_path = eval_dir / "qrels.json"
    
    with open(queries_path, "w", encoding="utf-8") as f:
        json.dump(extracted_queries, f, ensure_ascii=False, indent=2)
        
    with open(qrels_path, "w", encoding="utf-8") as f:
        json.dump(extracted_qrels, f, indent=2)
        
    # Also create benchmark_config.json
    config_path = eval_dir / "benchmark_config.json"
    config = {
        "dataset": "ai4bharat/MSMARCO-XI",
        "dataset_version": "default",
        "split": "validation",
        "number_of_queries": len(extracted_queries),
        "embedding_model": "tfidf_svd",
        "reranker_model": "JinaReranker (ONNX)",
        "index_version": "v001",
        "chunking_version": "multi_strategy_v001",
        "top_k": 5
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        
    print("✅ Created queries.json, qrels.json, and benchmark_config.json")

if __name__ == "__main__":
    main()
