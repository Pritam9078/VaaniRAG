"""
Run offline indexing:  python -m backend.scripts.build_index
"""
import json
from pathlib import Path

from backend.app.indexing import build_index

ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    stats = build_index(
        corpus_path=str(ROOT / "backend" / "data" / "sample_corpus.json"),
        out_dir=str(ROOT / "backend" / "data" / "index"),
        strategies=["fixed", "sentence", "semantic", "adaptive"],
        embedding_backend="tfidf_svd",
    )
    print(json.dumps(stats, indent=2))
