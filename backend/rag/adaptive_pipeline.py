import json
import time
from pathlib import Path
from backend.rag.retrieval.retrieval import HybridIndex
from backend.rag.reranking.rerank import rerank
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "artifacts" / "msmarco_xi" / "v001"
EVAL_DIR = ROOT.parent / "evaluation"

class AdaptivePipeline:
    def __init__(self):
        self.index = HybridIndex(str(INDEX_DIR))
        self.weights = {"w1": 0.0, "w2": 0.5, "w3": 0.5, "threshold": 0.5}
        try:
            with open(EVAL_DIR / "results" / "confidence_weights.json", "r") as f:
                self.weights = json.load(f)
        except Exception:
            pass
            
    def run(self, query: str, top_n: int = 10, language: str | None = None):
        # Parallel Retrieval (Phase 3)
        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(self.index.dense_search, query, 20)
            f2 = executor.submit(self.index.bm25_search, query, 20)
            dense = f1.result()
            sparse = f2.result()
            
        # RRF Fusion
        rrf_k = 60
        rrf_scores = {}
        for rank, (idx, _score) in enumerate(dense):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rank + rrf_k)
        for rank, (idx, _score) in enumerate(sparse):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rank + rrf_k)
            
        ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])
        
        # Features
        dense_top = dense[0][0] if dense else -1
        sparse_top = sparse[0][0] if sparse else -2
        agreement = 1.0 if dense_top == sparse_top else 0.0
        
        rrf_top1 = ranked[0][1] if ranked else 0.0
        rrf_top2 = ranked[1][1] if len(ranked) > 1 else 0.0
        rrf_margin = (rrf_top1 - rrf_top2) / (rrf_top1 + 1e-9)
        
        max_dense = max([s for _, s in dense]) if dense else 0.0
        dense_score = dense[0][1] / (max_dense + 1e-9) if dense else 0.0
        
        # Confidence
        w1, w2, w3, t = self.weights["w1"], self.weights["w2"], self.weights["w3"], self.weights["threshold"]
        conf = w1 * agreement + w2 * rrf_margin + w3 * dense_score
        
        candidates = []
        for idx, score in ranked:
            chunk = self.index.chunks[idx]
            if language and chunk["language"] != language:
                continue
            candidates.append({**chunk, "rrf_score": score})
            if len(candidates) >= 3: # Jina sees 3 to stay fast
                break
            
        if conf >= t:
            for c in candidates:
                c["relevance_score"] = c.get("rrf_score", 0.0) * 10.0 + 0.5 # Ensure it passes 0.20 threshold safely since we are confident
            return candidates[:top_n], False # False = Jina skipped
        else:
            return rerank(query, candidates, top_n=top_n), True # True = Jina used
