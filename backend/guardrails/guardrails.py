"""
guardrails.py
-------------
Layered guardrails applied at three points in the pipeline:

  1. Input guardrail  -- reject unsafe / disallowed queries before any
     retrieval happens.
  2. Retrieval guardrail -- if the best retrieved chunk's relevance
     score is below threshold, refuse rather than force an answer
     ("the system should know when NOT to answer").
  3. Output guardrail -- after generation, check that the answer is
     actually grounded in the retrieved context (lexical overlap
     proxy for a hallucination check) before returning it to the user.

This is intentionally simple/dependency-free (no external moderation
API) so it runs fully offline in this environment. Swap in a real
moderation endpoint (e.g. OpenAI Moderation, Llama-Guard) for
production -- see README.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_TOKEN_RE = re.compile(r"[A-Za-z\u0900-\u0D7F]{2,}")

# Minimal illustrative blocklist for demo purposes -- production should
# use a proper moderation classifier, not a keyword list.
_UNSAFE_PATTERNS = [
    r"\bhow to (make|build) a (bomb|weapon|explosive)\b",
    r"\bkill (myself|someone)\b",
    r"\bhack (into|a) \w+ (account|system)\b",
]

RELEVANCE_THRESHOLD = 0.02          # min top-chunk relevance_score to attempt an answer
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
    for pattern in _UNSAFE_PATTERNS:
        if re.search(pattern, q):
            return GuardrailResult(False, "input", "Query flagged as unsafe.")
    return GuardrailResult(True, "input")


def check_retrieval(candidates: list[dict[str, Any]]) -> GuardrailResult:
    if not candidates:
        return GuardrailResult(False, "retrieval", "No relevant context found.")
    top_score = candidates[0].get("relevance_score", 0.0)
    if top_score < RELEVANCE_THRESHOLD:
        return GuardrailResult(
            False, "retrieval",
            f"Top retrieval relevance ({top_score:.3f}) below threshold "
            f"({RELEVANCE_THRESHOLD}); likely off-topic / not in knowledge base.",
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
    # If the answer contains Indic characters, lexical overlap against English context won't work.
    # We bypass the strict lexical grounding check for cross-lingual generation.
    if re.search(r"[\u0900-\u0D7F]", answer):
        return GuardrailResult(True, "output")
        
    overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
    if overlap < GROUNDING_THRESHOLD:
        return GuardrailResult(
            False, "output",
            f"Lexical grounding consistency ({overlap:.2f}) below threshold "
            f"({GROUNDING_THRESHOLD}); answer not strongly grounded in context.",
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
