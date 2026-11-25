
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.ingestion.semantic_chunking import SemanticChunker


INPUT_FILE = BACKEND_DIR / "data" / "processed" / "documents.jsonl"
OUTPUT_FILE = (
    BACKEND_DIR / "data" / "processed" / "semantic_chunks.jsonl"
)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}. "
            "Run scripts/ingest_documents.py first."
        )

    chunker = SemanticChunker()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    document_count = 0
    chunk_count = 0
    skipped_count = 0

    with (
        INPUT_FILE.open("r", encoding="utf-8") as input_file,
        OUTPUT_FILE.open("w", encoding="utf-8") as output_file,
    ):
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue

            try:
                document = json.loads(line)

                chunks = chunker.chunk_document(document)

                if not chunks:
                    skipped_count += 1
                    continue

                for chunk in chunks:
                    output_file.write(
                        json.dumps(chunk, ensure_ascii=False) + "\n"
                    )

                document_count += 1
                chunk_count += len(chunks)

            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                skipped_count += 1
                print(f"Skipping line {line_number}: {exc}")

    print("\nSemantic chunking completed.")
    print(f"Documents processed: {document_count}")
    print(f"Semantic chunks created: {chunk_count}")
    print(f"Records skipped: {skipped_count}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
