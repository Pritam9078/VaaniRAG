"""
chunking.py
-----------
Implements multiple chunking strategies over the MSMARCO-XI-style corpus:

  1. Fixed-size (token/word) chunking with overlap
  2. Sentence-window chunking (groups of N sentences)
  3. Semantic chunking (splits at topic-shift boundaries detected via
     sentence-embedding cosine-similarity drops)
  4. Adaptive chunking (chunk size varies with local sentence-length /
     lexical-complexity, so dense text gets smaller chunks and simple
     text gets larger ones)

Every chunk is emitted with rich metadata (doc_id, language, chunk_type,
position, keywords) so the retriever can filter / trace provenance.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------------------------------------------------------
# Basic sentence splitter (works reasonably for both English and Devanagari
# text; avoids a hard NLTK/spaCy model-download dependency since this
# sandbox cannot reach huggingface.co / spaCy model CDNs).
# --------------------------------------------------------------------------
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?।])\s+")


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sents = _SENT_SPLIT_RE.split(text)
    return [s.strip() for s in sents if s.strip()]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    chunk_type: str
    language: str
    position: list[int]  # [start_char, end_char] in original doc
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "chunk_type": self.chunk_type,
            "language": self.language,
            "position": self.position,
            "keywords": self.keywords,
        }


def _top_keywords(text: str, n: int = 5) -> list[str]:
    words = re.findall(r"[A-Za-z\u0900-\u097F]{3,}", text.lower())
    freq: dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:n]]


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


# --------------------------------------------------------------------------
# 1. Fixed-size chunking (word-based proxy for token-based) with overlap
# --------------------------------------------------------------------------
def fixed_size_chunks(doc_id: str, text: str, language: str,
                       size: int = 60, overlap: int = 15) -> list[Chunk]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    cursor_char = 0
    # Precompute char offsets for each word start
    offsets = []
    idx = 0
    for w in words:
        idx = text.find(w, idx)
        offsets.append(idx)
        idx += len(w)

    while start < len(words):
        end = min(start + size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        start_char = offsets[start]
        end_char = offsets[end - 1] + len(words[end - 1])
        chunks.append(Chunk(
            chunk_id=_new_id(), doc_id=doc_id, text=chunk_text,
            chunk_type="fixed", language=language,
            position=[start_char, end_char], keywords=_top_keywords(chunk_text),
        ))
        if end == len(words):
            break
        start += max(size - overlap, 1)
    return chunks


# --------------------------------------------------------------------------
# 2. Sentence-window chunking (group of N sentences, no mid-sentence cuts)
# --------------------------------------------------------------------------
def sentence_window_chunks(doc_id: str, text: str, language: str,
                            window: int = 3, overlap: int = 1) -> list[Chunk]:
    sents = split_sentences(text)
    if not sents:
        return []
    chunks = []
    i = 0
    while i < len(sents):
        group = sents[i:i + window]
        chunk_text = " ".join(group)
        start_char = text.find(group[0])
        end_char = start_char + len(chunk_text) if start_char >= 0 else len(chunk_text)
        chunks.append(Chunk(
            chunk_id=_new_id(), doc_id=doc_id, text=chunk_text,
            chunk_type="sentence", language=language,
            position=[max(start_char, 0), end_char], keywords=_top_keywords(chunk_text),
        ))
        if i + window >= len(sents):
            break
        i += max(window - overlap, 1)
    return chunks


# --------------------------------------------------------------------------
# 3. Semantic chunking: embed each sentence (TF-IDF proxy for a sentence
#    encoder), walk through sentences, and cut a new chunk whenever
#    consecutive-sentence similarity drops below a threshold (topic shift).
# --------------------------------------------------------------------------
def semantic_chunks(doc_id: str, text: str, language: str,
                     sim_threshold: float = 0.15, max_sentences: int = 6) -> list[Chunk]:
    sents = split_sentences(text)
    if len(sents) <= 1:
        return sentence_window_chunks(doc_id, text, language, window=len(sents) or 1)

    vec = TfidfVectorizer().fit(sents)
    X = vec.transform(sents).toarray()

    def cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    chunks = []
    current = [sents[0]]
    for i in range(1, len(sents)):
        sim = cos(X[i - 1], X[i])
        if sim < sim_threshold or len(current) >= max_sentences:
            chunk_text = " ".join(current)
            start_char = text.find(current[0])
            chunks.append(Chunk(
                chunk_id=_new_id(), doc_id=doc_id, text=chunk_text,
                chunk_type="semantic", language=language,
                position=[max(start_char, 0), max(start_char, 0) + len(chunk_text)],
                keywords=_top_keywords(chunk_text),
            ))
            current = [sents[i]]
        else:
            current.append(sents[i])
    if current:
        chunk_text = " ".join(current)
        start_char = text.find(current[0])
        chunks.append(Chunk(
            chunk_id=_new_id(), doc_id=doc_id, text=chunk_text,
            chunk_type="semantic", language=language,
            position=[max(start_char, 0), max(start_char, 0) + len(chunk_text)],
            keywords=_top_keywords(chunk_text),
        ))
    return chunks


# --------------------------------------------------------------------------
# 4. Adaptive chunking: chunk size (in sentences) shrinks for
#    lexically-dense/complex sentences and grows for simple ones.
#    Complexity proxy = average word length + sentence length (no external
#    model needed, so it runs fully offline).
# --------------------------------------------------------------------------
def adaptive_chunks(doc_id: str, text: str, language: str,
                     base_window: int = 3) -> list[Chunk]:
    sents = split_sentences(text)
    if not sents:
        return []

    def complexity(s: str) -> float:
        words = s.split()
        if not words:
            return 0.0
        avg_len = sum(len(w) for w in words) / len(words)
        return 0.6 * avg_len + 0.4 * len(words)

    chunks = []
    i = 0
    while i < len(sents):
        c = complexity(sents[i])
        # High complexity -> smaller window (min 1); low complexity -> larger window
        if c > 9:
            window = max(base_window - 2, 1)
        elif c < 5:
            window = base_window + 2
        else:
            window = base_window
        group = sents[i:i + window]
        chunk_text = " ".join(group)
        start_char = text.find(group[0])
        chunks.append(Chunk(
            chunk_id=_new_id(), doc_id=doc_id, text=chunk_text,
            chunk_type="adaptive", language=language,
            position=[max(start_char, 0), max(start_char, 0) + len(chunk_text)],
            keywords=_top_keywords(chunk_text),
        ))
        i += window
    return chunks


STRATEGIES = {
    "fixed": fixed_size_chunks,
    "sentence": sentence_window_chunks,
    "semantic": semantic_chunks,
    "adaptive": adaptive_chunks,
}


def chunk_document(doc_id: str, text: str, language: str,
                    strategies: list[str] = None) -> list[Chunk]:
    """Run every requested chunking strategy over one passage and return
    the union of all resulting chunks (each tagged with its chunk_type)."""
    strategies = strategies or list(STRATEGIES.keys())
    out: list[Chunk] = []
    for name in strategies:
        fn = STRATEGIES[name]
        out.extend(fn(doc_id, text, language))
    return out
