import json
import os
import pickle
import faiss

from embedder import generate_embeddings

# Path to the chunks created in Step 2
CHUNK_FILE = "../data/processed/chunks.json"

with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [chunk["text"] for chunk in chunks]

print(f"Loaded {len(texts)} chunks")

# Generate embeddings
embeddings = generate_embeddings(texts)

dimension = embeddings.shape[1]

# Create FAISS index
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index
os.makedirs("../vectorstore", exist_ok=True)
faiss.write_index(index, "../vectorstore/faiss_index.bin")

# Save metadata (chunks)
with open("../vectorstore/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("FAISS index created successfully!")
print(f"Indexed {index.ntotal} chunks.")