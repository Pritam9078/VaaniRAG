"""
guardrails.py
-------------
Layered guardrails applied at multiple points in the pipeline:

  1. Safety Gate      -- reject malicious queries (regex screen).
  2. Scope Gate       -- reject out-of-domain conversational queries ("weather today").
  3. Degenerate Gate  -- reject queries with zero content tokens or zero BM25 hits.
  4. Retrieval Gate   -- reject if top cosine score < 0.45.
  5. Grounding Gate   -- check Tier 2 output grounding against context.

This is intentionally simple/dependency-free (no external moderation
API) so it runs fully offline in this environment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_TOKEN_RE = re.compile(r"[A-Za-z\u0900-\u0D7F]{2,}")

_UNSAFE_PATTERNS = [
    r"\bhow to (make|build) a (bomb|weapon|explosive)\b",
    r"\bkill (myself|someone)\b",
    r"\bhack (into|a) \w+ (account|system)\b",
]

_OOD_PATTERNS = [
    r"\border me a (pizza|burger|taxi)\b",
    r"\bwho (made|created|programmed) you\b",
    r"\b(what is the )?weather (today|tomorrow)\b",
]

RELEVANCE_THRESHOLD = 0.45          # τ = 0.45 from benchmarking
GROUNDING_THRESHOLD = 0.05          # min lexical overlap between answer and context


@dataclass
class GuardrailResult:
    allowed: bool
    stage: str
    reason: str | None = None


def _tokens(text: str) -> set:
    return set(_TOKEN_RE.findall(text.lower()))


def check_input(query: str) -> GuardrailResult:
    q = query.lower().strip()
    if not q:
        return GuardrailResult(False, "input", "Empty query.")
        
    # 1. Safety Gate
    for pattern in _UNSAFE_PATTERNS:
        if re.search(pattern, q):
            return GuardrailResult(False, "safety", "Query flagged as unsafe.")
            
    # 2. Scope Gate (Intent)
    for pattern in _OOD_PATTERNS:
        if re.fullmatch(pattern, q) or re.search(pattern, q):
            return GuardrailResult(False, "scope", "Query is out of scope (conversational/OOD).")
            
    # 3. Degenerate Gate (Tokens)
    if len(_tokens(q)) == 0:
         return GuardrailResult(False, "degenerate", "No lexical content tokens found.")
            
    return GuardrailResult(True, "input")


def check_retrieval(candidates: list[dict[str, Any]]) -> GuardrailResult:
    # 3. Degenerate Gate (BM25 hits check equivalent - if no candidates at all)
    if not candidates:
        return GuardrailResult(False, "degenerate", "No lexical evidence (0 BM25 hits).")
        
    # 4. Weak Retrieval Gate
    top_score = candidates[0].get("relevance_score", 0.0)
    if top_score < RELEVANCE_THRESHOLD:
        return GuardrailResult(
            False, "weak_retrieval",
            f"Top retrieval relevance ({top_score:.3f}) below floor "
            f"({RELEVANCE_THRESHOLD})."
        )
    return GuardrailResult(True, "retrieval")


def check_output_grounding(answer: str, context_chunks: list[dict[str, Any]]) -> GuardrailResult:
    if not answer.strip():
        return GuardrailResult(False, "output", "Empty generation.")
    answer_tokens = _tokens(answer)
    context_tokens = set()
    for c in context_chunks:
        context_tokens |= _tokens(c["text"])
    if not answer_tokens:
        return GuardrailResult(False, "output", "Answer has no extractable content.")
    
    # If the answer contains Indic characters, lexical overlap against English context won't work well
    # without translation. Skip for cross-lingual.
    if re.search(r"[\u0900-\u0D7F]", answer):
        return GuardrailResult(True, "output")
        
    overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
    if overlap < GROUNDING_THRESHOLD:
        return GuardrailResult(
            False, "output",
            f"Lexical grounding consistency ({overlap:.2f}) below threshold "
            f"({GROUNDING_THRESHOLD}); generative withheld."
        )
    return GuardrailResult(True, "output")


def grounding_score(answer: str, context_chunks: list[dict[str, Any]]) -> float:
    answer_tokens = _tokens(answer)
    if not answer_tokens:
        return 0.0
    context_tokens = set()
    for c in context_chunks:
        context_tokens |= _tokens(c["text"])
    return len(answer_tokens & context_tokens) / len(answer_tokens)
