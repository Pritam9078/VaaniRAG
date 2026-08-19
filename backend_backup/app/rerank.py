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
from typing import Any

_TOKEN_RE = re.compile(r"[A-Za-z\u0900-\u097F]{2,}")


def _tokens(text: str) -> set:
    return set(_TOKEN_RE.findall(text.lower()))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def rerank(query: str, candidates: list[dict[str, Any]],
           top_n: int = 5, dedup_threshold: float = 0.85) -> list[dict[str, Any]]:
    q_tokens = _tokens(query)

    scored = []
    for c in candidates:
        lexical = _jaccard(q_tokens, _tokens(c["text"]))
        # Blend the retrieval-stage RRF score with the rerank-stage lexical score.
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
