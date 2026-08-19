import os
import time
from groq import Groq
import httpx

client = Groq(api_key=os.environ.get("GROQ_API_KEY"), http_client=httpx.Client())
models = client.models.list()
for m in models.data:
    print(m.id)
