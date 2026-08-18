"""
rerank.py
---------
Reranks the RRF-fused candidate chunks before they're passed to the LLM.

Production recommendation: a cross-encoder such as
'cross-encoder/ms-marco-MiniLM-L-6-v2' scores (query, chunk) pairs
jointly, which is more accurate than independently-embedded similarity.
That model requires downloading weights from huggingface.co, which this
sandbox cannot reach, so the default reranker here uses lexical
term-overlap (Jaccard over tokens) as a fast, dependency-free relevance
proxy, plus a redundancy filter to drop near-duplicate chunks.
"""

from __future__ import annotations

import re
import time
from typing import Any
import numpy as np

_TOKEN_RE = re.compile(r"[A-Za-z\u0900-\u0D7F]{2,}")


def _tokens(text: str) -> set:
    return set(_TOKEN_RE.findall(text.lower()))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class JinaReranker:
    def __init__(self, model_id: str = "jinaai/jina-reranker-v2-base-multilingual", max_length: int = 128):
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
        except ImportError:
            self._session = None
            return

        onnx_path = None
        for candidate in ("onnx/model_int8.onnx", "onnx/model_quantized.onnx", "onnx/model.onnx"):
            try:
                onnx_path = hf_hub_download(model_id, candidate)
                break
            except Exception:
                pass
        
        if not onnx_path:
            self._session = None
            return

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
        self._input_names = {i.name for i in self._session.get_inputs()}

        self._tok = Tokenizer.from_file(hf_hub_download(model_id, "tokenizer.json"))
        self._tok.enable_truncation(max_length=max_length)
        self._tok.enable_padding()

    def is_loaded(self):
        return self._session is not None

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        if not passages:
            return np.empty(0, dtype=np.float32)
        enc = self._tok.encode_batch([(query, p) for p in passages])
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed: dict[str, np.ndarray] = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(ids)
        logits = self._session.run(None, feed)[0]
        return np.asarray(logits, dtype=np.float32).reshape(-1)


_RERANKER = None

def get_reranker() -> JinaReranker:
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = JinaReranker()
    return _RERANKER


def rerank(query: str, candidates: list[dict[str, Any]],
           top_n: int = 5, dedup_threshold: float = 0.85, adaptive: bool = True) -> list[dict[str, Any]]:
    
    # Adaptive Confidence Gate
    is_confident = False
    if adaptive and candidates:
        top_cand = candidates[0]
        if top_cand.get("dense_rank") == 0 and top_cand.get("sparse_rank") == 0:
            is_confident = True

    reranker = get_reranker()
    if reranker.is_loaded() and not is_confident:
        scores = reranker.score(query, [c["text"] for c in candidates])
        import math
        def calibrated_sigmoid(x):
            return 1 / (1 + math.exp(-(x + 4.0))) if x > -20 else 0.0
            
        scored = [{**c, "relevance_score": float(calibrated_sigmoid(s))} for c, s in zip(candidates, scores)]
        scored.sort(key=lambda x: -x["relevance_score"])
    else:
        # Fallback to lexical or use RRF directly if confident
        if is_confident:
            scored = [{**c, "relevance_score": c.get("rrf_score", 0.0)} for c in candidates]
        else:
            q_tokens = _tokens(query)
            scored = []
            for c in candidates:
                lexical = _jaccard(q_tokens, _tokens(c["text"]))
                combined = 0.5 * c.get("rrf_score", 0.0) + 0.5 * lexical
                scored.append({**c, "relevance_score": combined})
            scored.sort(key=lambda x: -x["relevance_score"])

    # Drop near-duplicate chunks (redundancy filter)
    final: list[dict[str, Any]] = []
    seen_token_sets = []
    for c in scored:
        t = _tokens(c["text"])
        if any(_jaccard(t, s) > dedup_threshold for s in seen_token_sets):
            continue
        final.append(c)
        seen_token_sets.append(t)
        if len(final) >= top_n:
            break

    return final
