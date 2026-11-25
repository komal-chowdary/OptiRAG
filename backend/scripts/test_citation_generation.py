
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.generation.answer_generator import AnswerGenerator


generator = AnswerGenerator()

retrieved_chunks = [
    {
        "text": (
            "Azure Blob Storage organizes data using storage "
            "accounts, containers, and blobs."
        ),
        "metadata": {
            "document_id": "document-001",
            "title": "Introduction to Azure Blob Storage",
            "source_url": "",
            "source_url_status": "local_only",
            "relative_path": "storage/blobs/introduction.md",
            "chunk_index": 0,
        },
    },
    {
        "text": (
            "A container organizes a set of blobs. "
            "A blob stores unstructured data."
        ),
        "metadata": {
            "document_id": "document-001",
            "title": "Introduction to Azure Blob Storage",
            "source_url": "",
            "source_url_status": "local_only",
            "relative_path": "storage/blobs/introduction.md",
            "chunk_index": 1,
        },
    },
    {
        "text": (
            "A storage account provides a unique namespace "
            "for Azure Storage data."
        ),
        "metadata": {
            "document_id": "document-002",
            "title": "Storage account overview",
            "source_url": (
                "https://example.invalid/unverified"
            ),
            "source_url_status": "candidate_unverified",
            "relative_path": "storage/common/storage-account-overview.md",
            "chunk_index": 0,
        },
    },
]

result = generator.generate(
    question="How does Azure Blob Storage organize data?",
    retrieved_chunks=retrieved_chunks,
)

print("\n--- GENERATED ANSWER ---")
print(result["answer"])

print("\n--- CITATIONS ---")
print(json.dumps(result["citations"], indent=2))

print("\n--- VALIDATION ---")

assert len(result["citations"]) == 2
assert result["citations"][0]["source_id"] == 1
assert result["citations"][0]["chunk_indices"] == [0, 1]

assert result["citations"][0]["source_url"] is None
assert result["citations"][1]["source_url"] is None

assert (
    result["citations"][1]["source_url_status"]
    == "candidate_unverified"
)

print("Document-level citation grouping: PASS")
print("Unverified URL protection: PASS")
print("Citation metadata preservation: PASS")
print("\nCitation generation test completed!")
