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

        return answer_core.strip()


class AnthropicGenerator(BaseGenerator):
    def __init__(self, model: str = "claude-sonnet-5"):
        self.model = model

    def generate(self, query: str, context_chunks: list[dict[str, Any]]) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        sources = "\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(context_chunks))
        system = (
            "Answer the user's question conversationally using ONLY the sources below. "
            "Do NOT use any citations, brackets, or reference numbers (like [1]) since your "
            "response will be read aloud by a voice assistant. Keep it to one short sentence. "
            "If the answer is not contained in the sources, say you don't have enough information."
        )
        msg = client.messages.create(
            model=self.model,
            max_tokens=300,
            system=system,
            messages=[{"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {query}"}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")


class GroqGenerator(BaseGenerator):
    def __init__(self, model: str = "allam-2-7b", max_tokens: int = 300):
        self.model = model
        self.max_tokens = max_tokens
        self.last_telemetry = {}
        self._http_client = None

    def _init_client(self):
        if self._http_client is None:
            import httpx
            # Phase 6: Persistent connection pool
            self._http_client = httpx.Client(
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
                timeout=httpx.Timeout(20.0),
                headers={
                    "Authorization": f"Bearer {os.environ.get('GROQ_API_KEY')}",
                    "Content-Type": "application/json"
                }
            )

    def generate(self, query: str, context_chunks: list[dict[str, Any]]) -> str:
        import uuid
        import json
        t_start = time.perf_counter()
        
        request_id = str(uuid.uuid4())
        
        t_setup_start = time.perf_counter()
        self._init_client()
        t_setup_end = time.perf_counter()
        
        sources = "\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(context_chunks))
        system = "Answer conversationally using ONLY the sources below. Do NOT use citations like [1]. ONE short sentence."
        
        payload = {
            "model": self.model,
            "max_tokens": getattr(self, "max_tokens", 32),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {query}"}
            ],
            "temperature": 0.0,
            "stream": True
        }
        
        retries = 0
        retry_total_delay_ms = 0
        content_chunks = []
        
        # Tracking variables
        t_conn_acquired = 0
        t_req_sent = 0
        t_first_token = 0
        t_req_end = 0
        status_code = 0
        rate_limit_headers = {}
        exception_name = None
        
        while retries <= 1: # Bounded retry policy (Phase 7)
            try:
                t_req_prep = time.perf_counter()
                
                # We use stream() to measure TTFT manually
                with self._http_client.stream("POST", "https://api.groq.com/openai/v1/chat/completions", json=payload) as response:
                    t_conn_acquired = time.perf_counter()
                    t_req_sent = t_conn_acquired # Approximated as the moment stream yields
                    
                    status_code = response.status_code
                    
                    # Capture headers for Phase 3
                    for k, v in response.headers.items():
                        if k.lower().startswith("x-ratelimit"):
                            rate_limit_headers[k] = v
                    
                    if status_code != 200:
                        error_text = response.read().decode("utf-8")
                        if status_code in (429, 502, 503):
                            raise Exception(f"HTTP {status_code}: {error_text}")
                        else:
                            raise Exception(f"HTTP {status_code}: {error_text}")
                    
                    ttft_recorded = False
                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            if line == "data: [DONE]":
                                break
                            
                            if not ttft_recorded:
                                t_first_token = time.perf_counter()
                                ttft_recorded = True
                                
                            data_str = line[6:]
                            try:
                                chunk = json.loads(data_str)
                                delta = chunk["choices"][0].get("delta", {}).get("content", "")
                                if delta:
                                    content_chunks.append(delta)
                            except:
                                pass
                                
                    t_req_end = time.perf_counter()
                    break # Success!
                    
            except Exception as e:
                exception_name = type(e).__name__
                if "429" in str(e) or "503" in str(e) or "rate limit" in str(e).lower():
                    retries += 1
                    if retries <= 1:
                        delay = 0.5
                        time.sleep(delay)
                        retry_total_delay_ms += (delay * 1000)
                    else:
                        break # Exceeded bounds
                else:
                    break # Don't retry on non-transient errors
                    
        content = "".join(content_chunks)
        t_parse_end = time.perf_counter()
        
        # Telemetry payload matching Phase 1/2 requirements
        self.last_telemetry = {
            "request_id": request_id,
            "query_id": hash(query) % 1000000,
            "request_start": t_start,
            "connection_acquisition_ms": (t_conn_acquired - t_req_prep) * 1000 if t_conn_acquired else 0,
            "request_send_ms": 0, # Included in conn acquisition
            "TTFT_ms": (t_first_token - t_req_sent) * 1000 if t_first_token else 0,
            "stream_generation_ms": (t_req_end - t_first_token) * 1000 if t_first_token else 0,
            "response_parse_ms": (t_parse_end - t_req_end) * 1000 if t_req_end else 0,
            "generation_total_ms": (t_parse_end - t_start) * 1000,
            "model": self.model,
            "input_tokens": len(query + sources) // 4,
            "output_tokens": len(content) // 4,
            "max_tokens": self.max_tokens,
            "status_code": status_code,
            "retry_count": retries,
            "retry_delay_ms": retry_total_delay_ms,
            "exception": exception_name,
            "rate_limit_headers": rate_limit_headers
        }
        
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
