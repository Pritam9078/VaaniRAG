"""
stt.py
------
Speech-to-text backend. Per the challenge requirements we standardize
on Sarvam (Saaras v3) for STT, since it's optimized for the 22 Indian
languages that match the MSMARCO-XI dataset.

`SarvamSTT` below is the real integration (needs SARVAM_API_KEY and
network access to api.sarvam.ai, which this sandbox does not have).
`MockSTT` lets the rest of the pipeline (retrieval/generation/
guardrails/benchmarking) be built, tested, and demoed end-to-end
without live audio infrastructure -- it simply returns pre-supplied
text as if it had been transcribed, while still recording a realistic
STT latency figure for the benchmark report.
"""

from __future__ import annotations

import io
import os
import time


class BaseSTT:
    def transcribe(self, audio_bytes: bytes, language_hint: str = None) -> tuple[str, float]:
        """Returns (transcript, latency_seconds)."""
        raise NotImplementedError


class SarvamSTT(BaseSTT):
    """Real Sarvam AI (Saaras v3) integration.

        pip install requests
        export SARVAM_API_KEY=...

    See https://docs.sarvam.ai for the current REST/WebSocket contract;
    endpoint/field names should be verified against their live docs
    before production use.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("SARVAM_API_KEY")

    def transcribe(self, audio_bytes: bytes, language_hint: str = None) -> tuple[str, float]:
        import requests

        t0 = time.perf_counter()
        resp = requests.post(
            "https://api.sarvam.ai/speech-to-text",
            headers={"api-subscription-key": self.api_key},
            files={"file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")},
            data={"model": "saaras:v3", "language_code": language_hint or "unknown"},
            timeout=10,
        )
        resp.raise_for_status()
        transcript = resp.json().get("transcript", "")
        return transcript, time.perf_counter() - t0


class MockSTT(BaseSTT):
    """Offline stand-in: treats the incoming 'audio_bytes' as UTF-8 text
    (the demo frontend sends the typed/spoken-then-manually-entered
    query this way when no live STT key is configured) and simulates a
    realistic Sarvam-class transcription latency."""

    def __init__(self, simulated_latency_s: float = 0.04):
        self.simulated_latency_s = simulated_latency_s

    def transcribe(self, audio_bytes: bytes, language_hint: str = None) -> tuple[str, float]:
        t0 = time.perf_counter()
        try:
            text = audio_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = ""
        # simulate STT processing time so benchmark numbers are representative
        elapsed = time.perf_counter() - t0
        remaining = max(self.simulated_latency_s - elapsed, 0)
        time.sleep(remaining)
        return text, time.perf_counter() - t0


def get_stt(backend: str = None) -> BaseSTT:
    backend = backend or os.environ.get("STT_BACKEND", "mock")
    if backend == "mock":
        return MockSTT()
    if backend == "sarvam":
        return SarvamSTT()
    raise ValueError(f"Unknown STT backend: {backend}")
