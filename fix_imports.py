import os
import re

replacements = {
    r"from \. import guardrails": "import backend.guardrails.guardrails as guardrails",
    r"from \.retrieval import": "from backend.rag.retrieval.retrieval import",
    r"from \.rerank import": "from backend.rag.reranking.rerank import",
    r"from \.generation import": "from backend.rag.generation.generation import",
    r"from \.stt import": "from backend.voice.stt import",
    r"from \.schemas import": "from backend.schemas.ask import",
    r"from \.embeddings import": "from backend.rag.retrieval.embeddings import",
    r"from \.chunking import": "from backend.rag.chunking.chunking import"
}

def fix_file(filepath):
    if not os.path.exists(filepath): return
    with open(filepath, 'r') as f:
        content = f.read()
    
    orig = content
    for old, new in replacements.items():
        content = re.sub(old, new, content)
        
    if content != orig:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Fixed {filepath}")

for root, _, files in os.walk('backend'):
    for file in files:
        if file.endswith('.py'):
            fix_file(os.path.join(root, file))

