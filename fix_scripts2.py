import os
import re

replacements = {
    r"from backend\.retrieval import": "from backend.rag.retrieval.retrieval import",
    r"from backend\.rerank import": "from backend.rag.reranking.rerank import",
    r"from backend\.generation import": "from backend.rag.generation.generation import",
    r"from backend\.guardrails import": "from backend.guardrails.guardrails import",
    r"from backend\.schemas import": "from backend.schemas.ask import",
    r"from backend\.main import": "from backend.main import",
}

for root, _, files in os.walk('backend/scripts'):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            with open(filepath, 'r') as f:
                content = f.read()
            for old, new in replacements.items():
                content = re.sub(old, new, content)
            with open(filepath, 'w') as f:
                f.write(content)
