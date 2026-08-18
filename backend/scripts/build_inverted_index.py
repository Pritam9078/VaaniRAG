import json
import math
import pickle
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

_TOKEN_RE = re.compile(r"[A-Za-z\u0900-\u0D7F]{2,}")

def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())

class FastBM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.N = 0
        self.avgdl = 0.0
        self.idf = {}
        # term -> [(doc_idx, term_freq)]
        self.postings = defaultdict(list)
        self.doc_lengths = []

    def build(self, corpus: List[str]):
        self.N = len(corpus)
        total_len = 0
        df = defaultdict(int)

        print("Tokenizing and building postings...")
        for doc_idx, text in enumerate(corpus):
            if doc_idx % 20000 == 0:
                print(f"Processed {doc_idx}/{self.N} documents...")
            tokens = _tokenize(text)
            self.doc_lengths.append(len(tokens))
            total_len += len(tokens)

            # count term frequency for this doc
            tf = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            
            for t, count in tf.items():
                self.postings[t].append((doc_idx, count))
                df[t] += 1

        self.avgdl = total_len / self.N if self.N else 0

        print("Calculating IDF...")
        for t, freq in df.items():
            # Standard BM25 IDF formula
            idf_val = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1)
            # Clip negative IDF
            self.idf[t] = max(idf_val, 0.01)

        print("Precomputing full BM25 term weights...")
        optimized_postings = {}
        for token, docs in self.postings.items():
            idf = self.idf.get(token, 0)
            num_factor = idf * (self.k1 + 1)
            optimized_docs = []
            for doc_idx, tf in docs:
                doc_len = self.doc_lengths[doc_idx]
                doc_norm = self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl)) if self.avgdl > 0 else 0
                weight = num_factor * tf / (tf + doc_norm)
                optimized_docs.append((doc_idx, weight))
            optimized_postings[token] = optimized_docs
        
        self.postings = optimized_postings
        print("Done building FastBM25.")

    def get_scores(self, query_tokens: List[str]) -> Dict[int, float]:
        # Only returns non-zero scores!
        scores = defaultdict(float)
        
        for token in set(query_tokens):
            if token not in self.postings:
                continue
            
            for doc_idx, weight in self.postings[token]:
                scores[doc_idx] += weight
                
        return scores

def main():
    root = Path(__file__).resolve().parent.parent.parent
    index_dir = root / "backend" / "artifacts" / "msmarco_xi" / "v001"
    
    print("Loading chunks.json...")
    with open(index_dir / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    corpus = [c.get("text", "") for c in chunks]
    
    bm25 = FastBM25()
    bm25.build(corpus)
    
    print("Converting postings to contiguous numpy arrays...")
    import numpy as np
    for token, doc_list in bm25.postings.items():
        bm25.postings[token] = (
            np.array([p[0] for p in doc_list], dtype=np.int32),
            np.array([p[1] for p in doc_list], dtype=np.float32)
        )

    out_path = index_dir / "inverted_bm25.pkl"
    print(f"Saving fast BM25 index to {out_path}...")
    with open(out_path, "wb") as f:
        pickle.dump(bm25, f)
        
    print("✅ New sparse index generated.")

if __name__ == "__main__":
    main()
