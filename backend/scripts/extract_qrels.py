import json
from datasets import load_dataset
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent.parent.parent
    index_dir = root / "backend" / "artifacts" / "msmarco_xi" / "v001"
    
    # Load chunks to get valid doc_ids
    print("Loading chunks.json...")
    with open(index_dir / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    valid_doc_ids = {c["doc_id"] for c in chunks}
    print(f"Loaded {len(valid_doc_ids)} unique doc_ids from chunks.")

    print("Loading MSMARCO-XI dataset...")
    # Try default config
    ds = load_dataset("ai4bharat/MSMARCO-XI", "default", split="validation", streaming=True)
    
    for row in ds:
        print("Row schema:")
        for k, v in row.items():
            print(f"{k}: {type(v)} - {str(v)[:100]}")
        break
    
if __name__ == "__main__":
    main()
