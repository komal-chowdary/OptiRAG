
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.ingestion.chunking import chunk_document
from app.ingestion.semantic_chunking import SemanticChunker


documents_file = BACKEND_DIR / "data/processed/documents.jsonl"

with documents_file.open(encoding="utf-8") as file:
    document = json.loads(next(file))

print("Testing document:", document["title"])

# Test fixed-size chunking.
fixed_chunks = chunk_document(
    document,
    chunk_size=1000,
    chunk_overlap=150,
)

assert fixed_chunks, "No fixed-size chunks created."

fixed = fixed_chunks[0]

required_fields = [
    "document_id",
    "title",
    "source",
    "source_url",
    "source_url_status",
    "relative_path",
    "chunk_index",
    "text",
    "chunking_strategy",
]

for field in required_fields:
    assert field in fixed, f"Missing field: {field}"

assert fixed["chunking_strategy"] == "fixed_size"

print("\nFixed-size chunk metadata:")
print(json.dumps(fixed, indent=2, ensure_ascii=False)[:2000])

# Test semantic chunking.
semantic_chunker = SemanticChunker()
semantic_chunks = semantic_chunker.chunk_document(document)

assert semantic_chunks, "No semantic chunks created."

semantic = semantic_chunks[0]

assert semantic["chunking_strategy"] == "semantic"
assert semantic["source_url_status"] == document["source_url_status"]
assert semantic["document_id"] == document["document_id"]

print("\nSemantic chunk metadata:")
print(json.dumps(semantic, indent=2, ensure_ascii=False)[:2000])

print("\nChunk metadata test passed!")
