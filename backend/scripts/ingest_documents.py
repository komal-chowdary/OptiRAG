
import hashlib
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.ingestion.document_loader import (
    SUPPORTED_EXTENSIONS,
    load_document,
)
from app.ingestion.preprocessing import (
    clean_text,
    extract_front_matter,
)


DATA_DIR = BACKEND_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "microsoft_docs"
OUTPUT_FILE = DATA_DIR / "processed" / "documents.jsonl"


def create_document_id(relative_path: str) -> str:
    """Generate a stable ID from the document's relative path."""
    return hashlib.sha256(
        relative_path.encode("utf-8")
    ).hexdigest()


def get_document_title(
    metadata: dict,
    content: str,
    fallback: str,
) -> str:
    """Get the best available title for a document."""
    title = metadata.get("title")

    if title:
        return str(title).strip()

    for line in content.splitlines():
        line = line.strip()

        if line.startswith("# "):
            return line[2:].strip()

    return fallback.replace("-", " ").replace("_", " ").title()


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(
        path
        for path in RAW_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        print(f"No supported documents found in: {RAW_DIR}")
        return

    processed = 0
    skipped = 0
    failed = 0
    url_candidates = 0
    local_only = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as output:
        for file_path in files:
            try:
                document = load_document(file_path)

                front_matter, content = extract_front_matter(
                    document["text"]
                )

                cleaned_text = clean_text(content)

                if not cleaned_text:
                    print(f"Skipped empty document: {file_path.name}")
                    skipped += 1
                    continue

                relative_path = document.get("relative_path")

                if not relative_path:
                    relative_path = file_path.relative_to(
                        RAW_DIR
                    ).as_posix()

                record = {
                    "document_id": create_document_id(relative_path),
                    "title": get_document_title(
                        front_matter,
                        cleaned_text,
                        file_path.stem,
                    ),
                    "description": front_matter.get("description"),
                    "text": cleaned_text,
                    "source": document["source"],
                    "source_url": document["source_url"],
                    "source_url_status": document["source_url_status"],
                    "relative_path": relative_path,
                    "file_type": document["file_type"],
                    "author": front_matter.get("author"),
                    "ms_author": front_matter.get("ms.author"),
                    "ms_service": front_matter.get("ms.service"),
                    "ms_topic": front_matter.get("ms.topic"),
                    "ms_date": front_matter.get("ms.date"),
                }

                output.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

                processed += 1

                if document["source_url_status"] == "candidate_unverified":
                    url_candidates += 1
                else:
                    local_only += 1

                print(f"Processed: {file_path.name}")

            except Exception as exc:
                failed += 1
                print(f"Failed: {file_path}: {exc}")

    print("\n========== INGESTION SUMMARY ==========")
    print(f"Documents processed: {processed}")
    print(f"Documents skipped: {skipped}")
    print(f"Documents failed: {failed}")
    print(f"Candidate URLs (unverified): {url_candidates}")
    print(f"Local-only sources: {local_only}")
    print(f"Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
