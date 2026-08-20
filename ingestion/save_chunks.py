import json
import os

from loader import load_all_pdfs
from chunker import split_into_chunks


PDF_FOLDER = "../data/pdfs"

all_chunks = []

documents = load_all_pdfs(PDF_FOLDER)

chunk_id = 0

for doc in documents:

    chunks = split_into_chunks(doc["text"])

    for chunk in chunks:

        all_chunks.append({
            "id": chunk_id,
            "filename": doc["filename"],
            "text": chunk
        })

        chunk_id += 1

os.makedirs("../data/processed", exist_ok=True)

with open("../data/processed/chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=4, ensure_ascii=False)

print(f"Saved {len(all_chunks)} chunks.")