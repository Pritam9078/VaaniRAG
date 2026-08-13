import argparse
import logging
import random
from collections.abc import Iterator
from typing import Any

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("msmarco_fetcher")

def stream_msmarco(
    language: str = "hin",
    split: str = "train",
    sample_size: int = 10000,
    seed: int = 42,
) -> Iterator[dict[str, Any]]:
    """
    Streams a deterministic sample of the MSMARCO-XI dataset directly from HuggingFace.
    Avoids loading the entire 55GB dataset into RAM.
    """
    random.seed(seed)
    
    # We load the specific parquet file for the language to speed up streaming
    data_files = {split: f"{split}/{language}{split}.parquet"}
    
    logger.info(f"Connecting to HuggingFace hub to stream ai4bharat/MSMARCO-XI (lang={language}, split={split})...")
    try:
        ds = load_dataset(
            "ai4bharat/MSMARCO-XI", 
            data_files=data_files, 
            split=split, 
            streaming=True
        )
    except Exception as e:
        logger.warning(f"Failed to load specific parquet file: {e}. Falling back to default config.")
        ds = load_dataset(
            "ai4bharat/MSMARCO-XI", 
            "default", 
            split=split, 
            streaming=True
        )

    total_seen = 0
    accepted = 0
    rejected = 0
    rejection_reasons = {}

    # Since streaming doesn't allow easy random sampling without reading all, 
    # we use reservoir sampling or just take the first N after a shuffle buffer.
    # Datasets streaming supports `.shuffle(seed=seed, buffer_size=10000)`.
    shuffled_ds = ds.shuffle(seed=seed, buffer_size=10000)

    for item in shuffled_ds:
        total_seen += 1
        
        # Verify language if we fell back to default
        if item.get("target_lang") != language and "target_lang" in item:
            rejected += 1
            rejection_reasons["wrong_language"] = rejection_reasons.get("wrong_language", 0) + 1
            continue

        # Extract passages
        try:
            eng_passages = item["passages"]["English_passages"]
            tgt_passages = item["passages"]["Translated_passages"]
            is_selected = item["passages"]["is_selected"]
            
            if not eng_passages or not tgt_passages:
                raise ValueError("Empty passages list")
                
            # Filter to only the selected/relevant passage for indexing
            # If is_selected is 1, it's the answer-bearing passage. 
            # We can index all of them, but let's just index all for robust search.
            doc_id = str(item.get("query_id", total_seen))
            
            yield {
                "doc_id": doc_id,
                "query_en": item.get("Eng_Query", ""),
                "query_tgt": item.get("query", ""),
                "passages": [
                    {"lang": "en", "text": p, "is_selected": is_selected[i] if i < len(is_selected) else 0} for i, p in enumerate(eng_passages)
                ] + [
                    {"lang": language, "text": p, "is_selected": is_selected[i] if i < len(is_selected) else 0} for i, p in enumerate(tgt_passages)
                ],
                "answer_en": item.get("Eng_Answer", ""),
                "answer_tgt": item.get("Answer", "")
            }
            accepted += 1
            
        except Exception as e:
            rejected += 1
            reason = str(e)
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            
        if accepted >= sample_size:
            break

    logger.info(f"Stream complete. Total seen: {total_seen}, Accepted: {accepted}, Rejected: {rejected}")
    if rejected > 0:
        logger.info(f"Rejection reasons: {rejection_reasons}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream and test MSMARCO-XI dataset.")
    parser.add_argument("--language", type=str, default="hin", help="Target language code (e.g. hin, tam)")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--sample-size", type=int, default=10, help="Number of records to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    for record in stream_msmarco(
        language=args.language,
        split=args.split,
        sample_size=args.sample_size,
        seed=args.seed
    ):
        print(f"Doc ID: {record['doc_id']}")
        print(f"English Query: {record['query_en']}")
        print(f"Hindi Query: {record['query_tgt']}")
        print(f"Passage count: {len(record['passages'])}")
        print("-" * 50)
