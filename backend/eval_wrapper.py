import time
from typing import Any
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from backend.rag.retrieval.embeddings import get_embedder
from backend.rag.generation.generation import GroqGenerator
from backend.guardrails import guardrails

# Global instances for the eval loop to reuse
_embedder = None
_generator = None

def get_model():
    """Initializes the models. Only side effect is required by the eval loop."""
    global _embedder, _generator
    if _embedder is None:
        _embedder = get_embedder(backend="model2vec")
    if _generator is None:
        _generator = GroqGenerator()
    return _embedder

def embed(texts: list[str]):
    get_model()
    # The JinaEmbedder.encode returns a list of vectors, we must return an array-like
    embeddings = _embedder.encode(texts)
    return np.array(embeddings)

def embed_one(text: str):
    return embed([text])[0]

class EvalAnswer:
    def __init__(self, text: str, grounded: bool, generation_ms: float, model: str):
        self.text = text
        self.grounded = grounded
        self.generation_ms = generation_ms
        self.model = model

def generate_answer(query: str, results: list) -> EvalAnswer:
    get_model()
    
    # Map eval loop's `results` to our `chunks` format expected by GroqGenerator
    # eval loop result has `.text` and `.source`
    chunks = [
        {"text": getattr(r, "text", ""), "doc_id": getattr(r, "source", ""), "chunk_id": f"chunk_{i}", "relevance_score": 1.0}
        for i, r in enumerate(results)
    ]
    
    t0 = time.perf_counter()
    answer_text = _generator.generate(query, chunks)
    generation_ms = (time.perf_counter() - t0) * 1000
    
    # Evaluate grounding using our guardrail
    out_check = guardrails.check_output_grounding(answer_text, chunks)
    ground_score = guardrails.grounding_score(answer_text, chunks)
    
    is_grounded = out_check.allowed and ground_score >= guardrails.GROUNDING_THRESHOLD
    
    return EvalAnswer(
        text=answer_text,
        grounded=bool(is_grounded),
        generation_ms=generation_ms,
        model=_generator.model
    )
