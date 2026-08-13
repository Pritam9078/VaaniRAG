"""
main.py
-------
FastAPI harness for VaaniRAG. This is the orchestration layer requested
by requirement #5: every stage (STT -> guardrail -> retrieval ->
rerank -> guardrail -> generation -> guardrail) is an explicit,
independently-timed, retried, and error-handled step -- not a single
raw prompt-in/text-out call.

Endpoints:
  POST /transcribe   - audio/text -> transcript
  POST /query        - text query -> answer (skips STT; useful for testing)
  POST /ask          - audio -> full pipeline -> answer  (primary endpoint)
  GET  /health        - liveness check
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.guardrails import guardrails
from backend.rag.generation.generation import get_generator
from backend.rag.reranking.rerank import rerank as rerank_chunks
from backend.rag.retrieval.retrieval import HybridIndex
from backend.schemas.ask import (
    AskResponse,
    SourceChunk,
    StageLatencies,
    TextQueryRequest,
)
from backend.voice.stt import get_stt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vaanirag")

APP_DIR = Path(__file__).resolve().parent
INDEX_DIR = APP_DIR / "data" / "index"

load_dotenv(APP_DIR / ".env")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _index
    if not (INDEX_DIR / "faiss.index").exists():
        logger.warning(
            "No index found at %s -- run `python -m backend.scripts.build_index` first.",
            INDEX_DIR,
        )
    else:
        _index = HybridIndex(str(INDEX_DIR))
        logger.info("Loaded index with %d chunks.", len(_index.chunks))
    yield

app = FastAPI(title="VaaniRAG", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# --- Lazily-initialized global components (loaded once at startup) ------
_index: HybridIndex | None = None
_generator = get_generator()
_stt = get_stt()

T = TypeVar("T")


def with_retry(fn: Callable[[], T], attempts: int = 3, label: str = "", backoff_factor: float = 0.5) -> T:
    """Retry wrapper with exponential backoff.
    Instantly fails on permanent errors (ValueError, TypeError, etc).
    """
    PERMANENT_ERRORS = (ValueError, TypeError, KeyError, AttributeError, NotImplementedError)
    last_exc = None
    
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except PERMANENT_ERRORS as exc:
            logger.error("Stage '%s' encountered permanent error: %s", label, exc)
            raise
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < attempts:
                sleep_time = backoff_factor * (2 ** (attempt - 1))
                logger.warning("Stage '%s' attempt %d/%d failed: %s. Retrying in %.2fs...", 
                               label, attempt, attempts, exc, sleep_time)
                time.sleep(sleep_time)
            else:
                logger.error("Stage '%s' failed after %d attempts: %s", label, attempts, exc)
                
    raise last_exc  # type: ignore[misc]





@app.get("/health")
def health():
    return {"status": "ok", "index_loaded": _index is not None}


# --------------------------------------------------------------------------
def run_pipeline(transcript: str, stt_ms: float, language: str | None = None) -> AskResponse:
    t_start = time.perf_counter()

    # ---- Stage: input guardrail ----
    g0 = time.perf_counter()
    input_check = guardrails.check_input(transcript)
    guardrail_ms = (time.perf_counter() - g0) * 1000

    if not input_check.allowed:
        total_ms = (time.perf_counter() - t_start) * 1000 + stt_ms
        return AskResponse(
            status="refused",
            transcript=transcript,
            refusal_reason=input_check.reason,
            refusal_stage=input_check.stage,
            latencies=StageLatencies(
                stt_ms=stt_ms, retrieval_ms=0, rerank_ms=0,
                generation_ms=0, guardrail_ms=guardrail_ms, total_ms=total_ms,
            ),
        )

    idx = _index
    if idx is None:
        raise HTTPException(status_code=503, detail="Index not loaded. Run build_index first.")

    import re
    search_query = transcript
    if re.search(r"[\u0900-\u0D7F]", transcript):
        try:
            translation = _generator.generate(
                query=f"Translate the following user query to English. Return ONLY the English translation without any extra text or quotes.\n\nQuery: {transcript}",
                context_chunks=[]
            )
            # Remove any possible quotes from the translation
            search_query = translation.strip().strip('"\'')
            # Since we translated the query to English, we must search the English chunks
            language = "en"
        except Exception as e:
            print(f"Query translation failed: {e}")

    # ---- Stage: hybrid retrieval (dense + BM25 + RRF) ----
    r0 = time.perf_counter()
    try:
        candidates = with_retry(
            lambda: idx.hybrid_search(search_query, language=language, top_n=3),
            label="retrieval",
        )
    except Exception as e:  # noqa: BLE001
        total_ms = (time.perf_counter() - t_start) * 1000 + stt_ms
        return AskResponse(
            status="refused",
            transcript=transcript,
            refusal_reason=f"Retrieval failure: {e!s}",
            refusal_stage="retrieval",
            latencies=StageLatencies(
                stt_ms=stt_ms, retrieval_ms=(time.perf_counter() - r0)*1000, rerank_ms=0,
                generation_ms=0, guardrail_ms=guardrail_ms, total_ms=total_ms,
            ),
        )
        
    retrieval_ms = (time.perf_counter() - r0) * 1000

    # ---- Stage: rerank ----
    rr0 = time.perf_counter()
    top_chunks = rerank_chunks(search_query, candidates, top_n=2)
    rerank_ms = (time.perf_counter() - rr0) * 1000

    # ---- Stage: retrieval guardrail (relevance threshold) ----
    g1 = time.perf_counter()
    retrieval_check = guardrails.check_retrieval(top_chunks)
    guardrail_ms += (time.perf_counter() - g1) * 1000

    if not retrieval_check.allowed:
        total_ms = (time.perf_counter() - t_start) * 1000 + stt_ms
        return AskResponse(
            status="refused",
            transcript=transcript,
            refusal_reason=retrieval_check.reason,
            refusal_stage=retrieval_check.stage,
            sources=[],
            latencies=StageLatencies(
                stt_ms=stt_ms, retrieval_ms=retrieval_ms, rerank_ms=rerank_ms,
                generation_ms=0, guardrail_ms=guardrail_ms, total_ms=total_ms,
            ),
        )

    # ---- Stage: generation ----
    gen0 = time.perf_counter()
    try:
        answer = with_retry(lambda: _generator.generate(transcript, top_chunks), label="generation")
    except Exception as e:  # noqa: BLE001  # noqa: BLE001
        total_ms = (time.perf_counter() - t_start) * 1000 + stt_ms
        return AskResponse(
            status="refused",
            transcript=transcript,
            refusal_reason=f"Generation failure: {e!s}",
            refusal_stage="generation",
            sources=[SourceChunk(**{k: c[k] for k in
                     ("chunk_id", "doc_id", "text", "chunk_type", "language")},
                     relevance_score=c["relevance_score"]) for c in top_chunks],
            latencies=StageLatencies(
                stt_ms=stt_ms, retrieval_ms=retrieval_ms, rerank_ms=rerank_ms,
                generation_ms=(time.perf_counter() - gen0)*1000, guardrail_ms=guardrail_ms, total_ms=total_ms,
            ),
        )
        
    generation_ms = (time.perf_counter() - gen0) * 1000

    # ---- Stage: output guardrail (grounding check) ----
    g2 = time.perf_counter()
    output_check = guardrails.check_output_grounding(answer, top_chunks)
    ground_score = guardrails.grounding_score(answer, top_chunks)
    guardrail_ms += (time.perf_counter() - g2) * 1000

    total_ms = (time.perf_counter() - t_start) * 1000 + stt_ms

    if not output_check.allowed:
        return AskResponse(
            status="refused",
            transcript=transcript,
            refusal_reason=output_check.reason,
            refusal_stage=output_check.stage,
            sources=[SourceChunk(**{k: c[k] for k in
                     ("chunk_id", "doc_id", "text", "chunk_type", "language")},
                     relevance_score=c["relevance_score"]) for c in top_chunks],
            grounding_score=ground_score,
            latencies=StageLatencies(
                stt_ms=stt_ms, retrieval_ms=retrieval_ms, rerank_ms=rerank_ms,
                generation_ms=generation_ms, guardrail_ms=guardrail_ms, total_ms=total_ms,
            ),
        )

    return AskResponse(
        status="answered",
        transcript=transcript,
        answer=answer,
        sources=[SourceChunk(**{k: c[k] for k in
                 ("chunk_id", "doc_id", "text", "chunk_type", "language")},
                 relevance_score=c["relevance_score"]) for c in top_chunks],
        grounding_score=ground_score,
        latencies=StageLatencies(
            stt_ms=stt_ms, retrieval_ms=retrieval_ms, rerank_ms=rerank_ms,
            generation_ms=generation_ms, guardrail_ms=guardrail_ms, total_ms=total_ms,
        ),
    )


@app.post("/query", response_model=AskResponse)
def query_text(req: TextQueryRequest):
    """Text-only entry point (skips STT) -- useful for testing/benchmarking
    the retrieval+guardrail+generation pipeline in isolation."""
    return run_pipeline(req.query, stt_ms=0.0, language=req.language)


@app.post("/transcribe")
def transcribe(audio: UploadFile = File(...)):  # noqa: B008
    audio_bytes = audio.file.read()
    transcript, latency_s = with_retry(lambda: _stt.transcribe(audio_bytes), label="stt")
    return {"transcript": transcript, "stt_ms": latency_s * 1000}


@app.post("/ask", response_model=AskResponse)
def ask(audio: UploadFile = File(...), language: str | None = None):  # noqa: B008
    """Primary endpoint: full pipeline, audio in -> structured answer out."""
    audio_bytes = audio.file.read()
    transcript, latency_s = with_retry(lambda: _stt.transcribe(audio_bytes), label="stt")
    return run_pipeline(transcript, stt_ms=latency_s * 1000, language=language)


# Serve the demo frontend (static single-page app) at /
frontend_dir = APP_DIR.parent / "frontend" / "static"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
