import os
import time
from groq import Groq
import httpx

client = Groq(api_key=os.environ.get("GROQ_API_KEY"), http_client=httpx.Client())

for m in ["llama-3.1-8b-instant", "gemma2-9b-it", "mixtral-8x7b-32768", "llama3-8b-8192"]:
    try:
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}],
            model=m,
            max_tokens=6,
            stream=False
        )
        t1 = time.perf_counter()
        print(f"{m}: {(t1-t0)*1000:.2f} ms")
    except Exception as e:
        print(f"{m}: {e}")
