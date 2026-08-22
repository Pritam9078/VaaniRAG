"""
Run offline indexing:  python -m backend.scripts.build_index
"""
import json
from pathlib import Path
from backend.rag.indexing import build_index

ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=1000)
    args = parser.parse_args()

    # The new build_index streaming support will be added in indexing.py
    stats = build_index(
        corpus_path="backend/data/merged_corpus.json",
        out_dir=str(ROOT / "backend" / "data" / "index"),
        strategies=["fixed", "sentence", "semantic", "adaptive"],
        embedding_backend="tfidf_svd",
        sample_size=args.sample_size
    )
    print(json.dumps(stats, indent=2))
