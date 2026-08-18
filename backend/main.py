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
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
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
from backend.voice.sarvam_ws import SarvamStream, Transcript, SpeechEvent
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vaanirag")

APP_DIR = Path(__file__).resolve().parent
INDEX_DIR = APP_DIR / "artifacts" / "msmarco_xi" / "v001"

load_dotenv(APP_DIR / ".env")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _index
    if not (INDEX_DIR / "dense.index").exists():
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


@app.websocket("/ws/voice")
async def websocket_voice_endpoint(websocket: WebSocket):
    await websocket.accept()
    # Initialize generators
    ext_gen = get_generator("extractive")
    groq_gen = get_generator("groq")
    
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        await websocket.close(code=1011, reason="No STT API key")
        return

    try:
        async with SarvamStream(api_key) as stream:
            
            # Start receiving STT events in background
            async def receive_stt():
                final_transcript = None
                stt_latency = 0.0
                async for event in stream.receive():
                    if isinstance(event, Transcript):
                        await websocket.send_json({
                            "type": "transcript_partial",
                            "text": event.text,
                            "is_final": event.is_final
                        })
                        if event.is_final:
                            final_transcript = event.text
                            stt_latency = event.elapsed_ms
                return final_transcript, stt_latency

            receive_task = asyncio.create_task(receive_stt())

            # Receive audio from browser
            while True:
                data = await websocket.receive_bytes()
                if not data: # EOF / flush
                    await stream.flush()
                    break
                await stream.send_audio(data)

            transcript, stt_ms = await receive_task
            
            if not transcript:
                await websocket.send_json({"type": "error", "message": "No transcript received"})
                return

            t_start = time.perf_counter()

            # Guardrail 1-3
            g0 = time.perf_counter()
            input_check = guardrails.check_input(transcript)
            guardrail_ms = (time.perf_counter() - g0) * 1000

            if not input_check.allowed:
                await websocket.send_json({"type": "refused", "reason": input_check.reason, "stage": input_check.stage})
                return

            # Retrieval
            r0 = time.perf_counter()
            candidates = _index.hybrid_search(transcript, top_n=3)
            retrieval_ms = (time.perf_counter() - r0) * 1000

            # Rerank
            rr0 = time.perf_counter()
            top_chunks = rerank_chunks(transcript, candidates, top_n=2)
            rerank_ms = (time.perf_counter() - rr0) * 1000

            # Guardrail 4
            g1 = time.perf_counter()
            ret_check = guardrails.check_retrieval(top_chunks)
            guardrail_ms += (time.perf_counter() - g1) * 1000
            if not ret_check.allowed:
                await websocket.send_json({"type": "refused", "reason": ret_check.reason, "stage": ret_check.stage, "score": top_chunks[0].get("relevance_score", 0)})
                return

            # Tier 1: Extractive
            tier1_start = time.perf_counter()
            tier1_answer = ext_gen.generate(transcript, top_chunks)
            tier1_ms = (time.perf_counter() - tier1_start) * 1000
            
            sources_out = [
                {
                    "chunk_id": c["chunk_id"], "doc_id": c["doc_id"], 
                    "text": c["text"], "language": c["language"], 
                    "relevance_score": c["relevance_score"]
                } for c in top_chunks
            ]

            await websocket.send_json({
                "type": "tier1",
                "answer": tier1_answer,
                "sources": sources_out,
                "latencies": {
                    "stt_ms": stt_ms,
                    "retrieval_ms": retrieval_ms,
                    "rerank_ms": rerank_ms,
                    "tier1_ms": tier1_ms,
                    "guardrail_ms": guardrail_ms
                }
            })

            # Tier 2: Generative
            t2_start = time.perf_counter()
            try:
                tier2_answer = await asyncio.to_thread(groq_gen.generate, transcript, top_chunks)
                t2_ms = (time.perf_counter() - t2_start) * 1000
                
                g2 = time.perf_counter()
                out_check = guardrails.check_output_grounding(tier2_answer, top_chunks)
                guardrail_ms += (time.perf_counter() - g2) * 1000

                if out_check.allowed:
                    await websocket.send_json({
                        "type": "tier2",
                        "answer": tier2_answer,
                        "grounding_score": guardrails.grounding_score(tier2_answer, top_chunks),
                        "latencies": {
                            "generation_ms": t2_ms,
                            "guardrail_ms": guardrail_ms
                        }
                    })
                else:
                    await websocket.send_json({
                        "type": "tier2_refused",
                        "reason": out_check.reason
                    })
            except Exception as e:
                await websocket.send_json({"type": "tier2_error", "message": str(e)})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
        



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
