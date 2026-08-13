# VaaniRAG — Voice-Enabled RAG

A working, end-to-end voice RAG pipeline: **speech → transcription → hybrid
retrieval (dense + BM25 + RRF) → rerank → guardrails → grounded LLM answer**,
served through a FastAPI harness with structured I/O, retries, and a live
demo UI.

```
User speech
   │  (browser SpeechRecognition demo / Sarvam STT in production)
   ▼
Query text
   │  input guardrail (safety / off-topic)
   ▼
Hybrid Retrieval  ──┬── Dense (FAISS, HNSW)
                     └── Sparse (BM25)
   │  Reciprocal Rank Fusion
   ▼
Rerank + dedup
   │  retrieval guardrail (relevance threshold → refuse if below)
   ▼
LLM generation (grounded, cited)
   │  output guardrail (grounding / hallucination check)
   ▼
Answer + sources + per-stage latency
```

## Run it

```bash
cd backend
pip install -r requirements.txt

# 1. Build the offline index (chunking + embedding + FAISS/BM25)
python -m backend.scripts.build_index

# 2. Start the server (also serves the demo UI at /)
cd ..
python -m uvicorn backend.app.main:app --reload

# 3. Open http://localhost:8000  — hold the mic button or type a question
```

Run the latency benchmark (requirement #4):

```bash
python -m backend.scripts.benchmark --n 120
```

## What's implemented against each requirement

| # | Requirement | Where |
|---|---|---|
| 1 | STT (Sarvam **or** ElevenLabs, pick one) | `backend/app/stt.py` — standardized on **Sarvam (Saaras v3)**, chosen for its 22-Indic-language coverage matching MSMARCO-XI. Real integration included; see limitations below. |
| 2 | Vast chunking strategy | `backend/app/chunking.py` — **4 strategies** run over every passage: fixed-size w/ overlap, sentence-window, semantic (embedding similarity topic-shift detection), adaptive (chunk size scales with lexical complexity). Every chunk carries metadata (doc_id, language, chunk_type, char position, keywords) for filtering/traceability. |
| 3 | <200ms end-to-end | Benchmarked at **P50 ≈ 1.5ms, P70 ≈ 1.7ms, P100 ≈ 11ms** for retrieval→rerank→generation→guardrails (see `backend/data/benchmark_report.json`). This excludes STT, which is a separate, external network call — see honest caveat below. |
| 4 | P50/P70/P100 latency analytics | `backend/scripts/benchmark.py` — runs 120 queries (real + paraphrased + deliberately off-topic/unsafe) through the full pipeline and reports percentiles per stage. |
| 5 | Harness (not a raw prompt call) | `backend/app/main.py` — explicit staged orchestration (STT → guardrail → retrieval → rerank → guardrail → generation → guardrail), each stage independently timed, wrapped in `with_retry()`, structured Pydantic request/response schemas (`backend/app/schemas.py`). |
| 6 | Guardrails | `backend/app/guardrails.py` — three layers: input safety/off-topic rejection, retrieval relevance-threshold refusal ("don't hallucinate when nothing relevant was found"), and post-generation grounding check that blocks ungrounded answers. All three are exercised in the demo (try an off-topic question). |

## Honest limitations (read before demoing)

This was built inside a sandboxed environment with **no network route to
huggingface.co**, so two production components had to be substituted with
offline-friendly equivalents. The architecture and interfaces are real and
swappable — only the specific weights/API calls are stubbed:

- **Dataset**: MSMARCO-XI is ~11.45M examples / ~55.6GB and could not be
  downloaded here. `backend/data/sample_corpus.json` is a small
  representative corpus (10 topics, English + Hindi, MSMARCO-XI's exact
  schema) so the full pipeline is genuinely exercised end-to-end. **Before
  submitting**, swap this for a real subset, e.g.:
  ```python
  from datasets import load_dataset
  ds = load_dataset("ai4bharat/MSMARCO-XI", "hi", split="train[:20000]")
  ```
  and adapt `backend/app/indexing.py`'s loader accordingly. Do not attempt
  to index all 11M rows for a hackathon timeline — scope to 1-2 language
  splits and a bounded row count, and say so explicitly in your submission.

- **Embeddings**: default backend is TF-IDF + TruncatedSVD
  (`backend/app/embeddings.py`), not a pretrained sentence encoder, since
  downloading `sentence-transformers` weights requires huggingface.co
  access. Swap in `SentenceTransformerEmbedder` (already stubbed in the
  same file) with `paraphrase-multilingual-MiniLM-L12-v2` once you have
  network access — it's a one-line change (`get_embedder("sentence_transformer")`).

- **Reranker**: uses lexical Jaccard overlap as a fast proxy
  (`backend/app/rerank.py`) instead of a cross-encoder. Swap in
  `cross-encoder/ms-marco-MiniLM-L-6-v2` for real semantic reranking.

- **Generation**: default backend is a deterministic, dependency-free
  extractive composer (`ExtractiveGenerator` in `backend/app/generation.py`)
  so the pipeline runs with zero API keys. An `AnthropicGenerator` stub
  using the real Messages API is included — set `ANTHROPIC_API_KEY` and
  `GENERATOR_BACKEND=anthropic` to use it. **A real LLM call will add
  meaningful latency** (typically 150-400ms+ for a hosted API round trip)
  — budget for this against the 200ms target; a small local/quantized
  model will get you closer than a hosted API.

- **STT**: `SarvamSTT` in `backend/app/stt.py` is real integration code
  against Sarvam's REST endpoint (needs `SARVAM_API_KEY`, verify the
  current endpoint contract against Sarvam's docs before submitting). The
  demo UI instead uses the **browser's built-in SpeechRecognition API**
  for live voice capture, since this sandbox can't reach Sarvam's servers
  either — this still gives you a genuinely working "speak → hear it
  transcribed → get an answer" demo without needing an API key to test
  locally. Wire up `SarvamSTT` for your actual submission per requirement #1.

- **Latency numbers**: the 1-11ms figures above are real, measured on this
  environment, but against the small sample corpus and TF-IDF embeddings —
  not the full MSMARCO-XI corpus with a production embedding model and
  hosted LLM call. Re-run `benchmark.py` after swapping in the real
  dataset/embeddings/generator and report those numbers, since retrieval
  latency scales with corpus size and LLM latency will dominate your
  budget once it's a real API call.

## Project layout

```
backend/
  app/
    main.py         FastAPI harness (orchestration, retries, endpoints)
    chunking.py      4 chunking strategies + metadata
    embeddings.py    pluggable embedding backend
    indexing.py      offline indexing job (chunk → embed → FAISS/BM25)
    retrieval.py     hybrid search + RRF fusion
    rerank.py        relevance reranking + dedup
    guardrails.py    input / retrieval / output guardrails
    generation.py    pluggable LLM generation backend
    stt.py           pluggable STT backend (Sarvam / mock)
    schemas.py       Pydantic request/response models
  data/
    sample_corpus.json   representative MSMARCO-XI-shaped corpus
    index/                built FAISS + BM25 + chunk store (generated)
  scripts/
    build_index.py   run offline indexing
    benchmark.py     P50/P70/P100 latency benchmark
frontend/
  static/index.html  demo UI (hold-to-speak, transcript, answer, sources, live latency)
```
