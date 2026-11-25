
import json
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BACKEND_DIR / "data" / "processed" / "documents.jsonl"
OUTPUT_FILE = BACKEND_DIR / "data" / "processed" / "chunks.jsonl"

# Make the backend package importable when running this script directly.
import sys

sys.path.insert(0, str(BACKEND_DIR))

from app.ingestion.chunking import chunk_document


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run scripts/ingest_documents.py first."
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    document_count = 0
    chunk_count = 0
    skipped_count = 0

    with INPUT_FILE.open("r", encoding="utf-8") as source_file, \
            OUTPUT_FILE.open("w", encoding="utf-8") as output_file:

        for line_number, line in enumerate(source_file, start=1):
            if not line.strip():
                continue

            try:
                document = json.loads(line)

                if not document.get("document_id"):
                    raise ValueError("Missing document_id.")

                chunks = chunk_document(
                    document,
                    chunk_size=1000,
                    chunk_overlap=150,
                )

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

    print("\nChunking completed.")
    print(f"Documents chunked: {document_count}")
    print(f"Chunks created: {chunk_count}")
    print(f"Documents/records skipped: {skipped_count}")
    print(f"Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
