from __future__ import annotations

from pydantic import BaseModel


class SourceChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    chunk_type: str
    language: str
    relevance_score: float


class StageLatencies(BaseModel):
    stt_ms: float
    retrieval_ms: float
    rerank_ms: float
    generation_ms: float
    guardrail_ms: float
    total_ms: float


class AskResponse(BaseModel):
    status: str  # "answered" | "refused"
    transcript: str | None = None
    answer: str | None = None
    refusal_reason: str | None = None
    refusal_stage: str | None = None
    sources: list[SourceChunk] = []
    grounding_score: float | None = None
    latencies: StageLatencies


class TextQueryRequest(BaseModel):
    query: str
    language: str | None = None
