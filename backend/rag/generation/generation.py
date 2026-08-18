"""
generation.py
-------------
Pluggable answer-generation backend.
"""

from __future__ import annotations

import os
import re
from typing import Any
import time


class BaseGenerator:
    def generate(self, query: str, context_chunks: list[dict[str, Any]]) -> str:
        raise NotImplementedError


class ExtractiveGenerator(BaseGenerator):
    """Composes a grounded, cited answer from the top retrieved chunks
    without calling any external LLM. Deterministic + offline-safe."""

    def generate(self, query: str, context_chunks: list[dict[str, Any]]) -> str:
        if not context_chunks:
            return "I don't have enough information to answer that."

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
    def __init__(self, model: str = "claude-sonnet-5"):
        self.model = model

    def generate(self, query: str, context_chunks: list[dict[str, Any]]) -> str:
        import anthropic

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
    def __init__(self, model: str = "allam-2-7b", max_tokens: int = 6):
        self.model = model
        self.max_tokens = max_tokens
        self.last_telemetry = {}
        self.client = None
        self.http_client = None

    def _init_client(self):
        if self.client is None:
            from groq import Groq
            import httpx
            
            # Use persistent HTTPX client with connection pooling
            self.http_client = httpx.Client(
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                timeout=httpx.Timeout(15.0)
            )
            self.client = Groq(
                api_key=os.environ.get("GROQ_API_KEY"),
                http_client=self.http_client
            )

    def generate(self, query: str, context_chunks: list[dict[str, Any]]) -> str:
        t_start = time.perf_counter()
        
        t_setup_start = time.perf_counter()
        self._init_client()
        t_setup_end = time.perf_counter()
        
        self.last_telemetry["client_prepare_ms"] = (t_setup_end - t_setup_start) * 1000
        
        sources = "\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(context_chunks))
        system = "Answer using ONLY the sources below. Cite like [1]. ONE short sentence."
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {query}"}
        ]
        
        t_prompt_build_end = time.perf_counter()
        self.last_telemetry["prompt_build_ms"] = (t_prompt_build_end - t_setup_end) * 1000
        
        retries = 0
        while True:
            try:
                t_req_send = time.perf_counter()
                
                # Streaming call
                msg_stream = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=getattr(self, "max_tokens", 64),
                    messages=messages,
                    temperature=0.0,
                    stream=True
                )
                
                content_chunks = []
                ttft_recorded = False
                t_first_token = 0
                
                for chunk in msg_stream:
                    if not ttft_recorded:
                        t_first_token = time.perf_counter()
                        # Strict TTFT measurement (first token minus request send)
                        self.last_telemetry["TTFT_ms"] = (t_first_token - t_req_send) * 1000
                        ttft_recorded = True
                        
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        content_chunks.append(delta)
                        
                t_req_end = time.perf_counter()
                content = "".join(content_chunks)
                self.last_telemetry["stream_generation_ms"] = (t_req_end - t_first_token) * 1000
                self.last_telemetry["model_generation_ms"] = (t_req_end - t_req_send) * 1000
                break
                
            except Exception as e:
                if "rate limit" in str(e).lower() or "429" in str(e):
                    retries += 1
                    time.sleep(1)
                else:
                    self.last_telemetry["error_type"] = type(e).__name__
                    raise e
                    
        self.last_telemetry["retry_count"] = retries
        
        t_parse_start = time.perf_counter()
        self.last_telemetry["output_tokens"] = len(content) // 4
        t_parse_end = time.perf_counter()
        self.last_telemetry["response_parse_ms"] = (t_parse_end - t_parse_start) * 1000
        self.last_telemetry["generation_total_ms"] = (t_parse_end - t_start) * 1000
        
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
