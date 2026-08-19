import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
try:
    models = client.models.list()
    print([m.id for m in models.data])
except Exception as e:
    print("Error:", e)
