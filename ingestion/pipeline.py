from loader import load_all_pdfs
from chunker import split_into_chunks


PDF_FOLDER = "../data/pdfs"

documents = load_all_pdfs(PDF_FOLDER)

print("=" * 60)
print("Loaded Documents:", len(documents))
print("=" * 60)

total_chunks = 0

for doc in documents:

    print(f"\nDocument: {doc['filename']}")

    chunks = split_into_chunks(doc["text"])

    print(f"Chunks: {len(chunks)}")

    total_chunks += len(chunks)

    print("\nFirst Chunk:\n")
    print(chunks[0][:500])

print("\nTotal Chunks:", total_chunks)