import os
import re

replacements = {
    r"backend\.app\.indexing": "backend.rag.indexing",
    r"backend\.app\.": "backend."
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
