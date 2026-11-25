
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.ingestion.semantic_indexing import SemanticVectorIndexer


INPUT_FILE = (
    BACKEND_DIR / "data" / "processed" / "semantic_chunks.jsonl"
)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"File not found: {INPUT_FILE}"
        )

    chunks = []

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                chunk = json.loads(line)

                required_fields = [
                    "chunk_id",
                    "document_id",
                    "chunk_index",
                    "text",
                ]

                for field in required_fields:
                    if field not in chunk:
                        raise ValueError(f"Missing field: {field}")

                chunks.append(chunk)

            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid record on line {line_number}: {exc}"
                ) from exc

    if not chunks:
        print("No semantic chunks found.")
        return

    indexer = SemanticVectorIndexer()

    indexed = indexer.index_chunks(chunks)

    print("\nSemantic indexing completed successfully.")
    print(f"Chunks indexed: {indexed}")
    print(f"Total semantic chunks in ChromaDB: {indexer.get_count()}")
    print("Collection: microsoft_docs_semantic_v2")
    print(f"Database: {BACKEND_DIR / 'storage' / 'chroma_db'}")


if __name__ == "__main__":
    main()
