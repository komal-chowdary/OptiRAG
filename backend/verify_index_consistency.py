import json
from pathlib import Path
import chromadb

chunks_path = Path("data/processed/chunks.jsonl")

bm25_ids = {
    json.loads(line)["chunk_id"]
    for line in chunks_path.open(encoding="utf-8")
    if line.strip()
}

client = chromadb.PersistentClient(path="storage/chroma_db")
collection = client.get_collection("microsoft_docs_v2")

vector_ids = set()
offset = 0
batch_size = 1000
total = collection.count()

while offset < total:
    result = collection.get(
        limit=batch_size,
        offset=offset,
        include=[]
    )
    vector_ids.update(result["ids"])
    offset += batch_size

print("BM25 chunks:", len(bm25_ids))
print("Vector chunks:", len(vector_ids))
print("Common IDs:", len(bm25_ids & vector_ids))
print("BM25-only IDs:", len(bm25_ids - vector_ids))
print("Vector-only IDs:", len(vector_ids - bm25_ids))
