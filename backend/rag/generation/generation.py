"""
generation.py
-------------
Pluggable answer-generation backend.

Production recommendation: call a fast hosted or local LLM (see
`AnthropicGenerator` / `OpenAIGenerator` stubs below) with a grounded
system prompt that forces citation of source chunk indices, e.g.:

    "Answer ONLY using the numbered sources below. Cite sources
     like [1], [2]. If the answer isn't in the sources, say so."

This sandbox has no injected API credentials, so the default backend
(`ExtractiveGenerator`) is a dependency-free, template-based composer
that builds a cited answer directly from the top-ranked chunks. It's
deterministic and fast (good for latency benchmarking) and keeps the
end-to-end pipeline runnable without any external API key. Swapping in
a real LLM is a one-line change (`GENERATOR_BACKEND` env var / config).
"""

from __future__ import annotations

import os
import re
from typing import Any


class BaseGenerator:
    def generate(self, query: str, context_chunks: list[dict[str, Any]]) -> str:
        raise NotImplementedError


class ExtractiveGenerator(BaseGenerator):
    """Composes a grounded, cited answer from the top retrieved chunks
    without calling any external LLM. Deterministic + offline-safe."""

    def generate(self, query: str, context_chunks: list[dict[str, Any]]) -> str:
        if not context_chunks:
            return "I don't have enough information to answer that."

        # Pick the 1-2 sentences from the top chunk most lexically similar
        # to the query as the core answer, then attach citations.
        top = context_chunks[0]
        sentences = re.split(r"(?<=[.!?।])\s+", top["text"])
        q_tokens = set(re.findall(r"[A-Za-z\u0900-\u097F]{2,}", query.lower()))

        def score(s: str) -> int:
            s_tokens = set(re.findall(r"[A-Za-z\u0900-\u097F]{2,}", s.lower()))
            return len(q_tokens & s_tokens)

        best_sentences = sorted(sentences, key=score, reverse=True)[:2]
        answer_core = " ".join(best_sentences) if any(score(s) for s in best_sentences) else sentences[0]

        citations = " ".join(f"[{i+1}]" for i in range(min(len(context_chunks), 3)))
        return f"{answer_core.strip()} {citations}"


class AnthropicGenerator(BaseGenerator):
    """Real LLM backend using the Anthropic Messages API. Requires
    ANTHROPIC_API_KEY to be set in the environment. Not used by default."""

    def __init__(self, model: str = "claude-sonnet-5"):
        self.model = model

    def generate(self, query: str, context_chunks: list[dict[str, Any]]) -> str:
        import anthropic  # noqa: local import, optional dependency

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        sources = "\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(context_chunks))
        system = (
            "Answer the user's question using ONLY the numbered sources below. "
            "Cite sources inline like [1], [2]. If the answer is not contained "
            "in the sources, say you don't have enough information."
        )
        msg = client.messages.create(
            model=self.model,
            max_tokens=300,
            system=system,
            messages=[{"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {query}"}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")


class GroqGenerator(BaseGenerator):
    """Real LLM backend using the Groq API (extremely low latency). Requires
    GROQ_API_KEY to be set in the environment."""

    def __init__(self, model: str = "llama-3.1-8b-instant"):
        self.model = model

    def generate(self, query: str, context_chunks: list[dict[str, Any]]) -> str:
        from groq import Groq

        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        sources = "\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(context_chunks))
        system = (
            "Answer using ONLY the sources below. Cite like [1]. ONE short sentence."
        )
        msg = client.chat.completions.create(
            model=self.model,
            max_tokens=40,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {query}"}
            ],
            temperature=0.0
        )
        content = msg.choices[0].message.content
        return str(content) if content else ""


def get_generator(backend: str = None) -> BaseGenerator:
    backend = backend or os.environ.get("GENERATOR_BACKEND", "extractive")
    if backend == "extractive":
        return ExtractiveGenerator()
    if backend == "anthropic":
        return AnthropicGenerator()
    if backend == "groq":
        return GroqGenerator()
    raise ValueError(f"Unknown generator backend: {backend}")
