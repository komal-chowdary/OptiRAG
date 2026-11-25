
import csv
import json
from pathlib import Path
from collections import defaultdict


# backend/ is the project root for this script
BACKEND_DIR = Path(__file__).resolve().parents[1]

EVALUATION_DIR = BACKEND_DIR / "data" / "evaluation"
PROCESSED_DIR = BACKEND_DIR / "data" / "processed"

CSV_PATH = EVALUATION_DIR / "golden_review_sheet.csv"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

OUTPUT_JSONL = EVALUATION_DIR / "golden_dataset.jsonl"
OUTPUT_MAPPING = EVALUATION_DIR / "document_id_mapping.json"
OUTPUT_REPORT = EVALUATION_DIR / "conversion_report.json"


def normalize_path(path):
    """Normalize Windows/Linux paths for reliable matching."""
    return path.strip().replace("\\", "/").lower().lstrip("./")


def load_document_mapping():
    """
    Build a mapping from source file paths to document IDs
    using the existing fixed-size chunks file.
    """
    mapping = defaultdict(set)

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_PATH}"
        )

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            chunk = json.loads(line)
            source = chunk.get("source")
            document_id = chunk.get("document_id")

            if source and document_id:
                normalized = normalize_path(source)
                mapping[normalized].add(document_id)

    return {
        path: sorted(document_ids)
        for path, document_ids in mapping.items()
    }


def resolve_document_ids(source_path, document_mapping):
    """
    Match a relative CSV source path against the full source
    paths stored in chunks.jsonl.
    """
    target = normalize_path(source_path)

    matches = set()

    for stored_path, document_ids in document_mapping.items():
        if (
            stored_path == target
            or stored_path.endswith("/" + target)
        ):
            matches.update(document_ids)

    return sorted(matches)


def split_document_paths(value):
    if not value.strip():
        return []

    return [
        item.strip()
        for item in value.split(";")
        if item.strip()
    ]


def add_evidence(record, row):
    """Append evidence from a main or continuation CSV row."""

    quote = row.get("evidence_quote", "").strip()
    passage = row.get("passage_around_evidence", "").strip()
    document_paths = split_document_paths(
        row.get("evidence_documents", "")
    )

    # Ignore rows that contain no evidence information.
    if not quote and not passage and not document_paths:
        return

    record["evidence"].append({
        "quote": quote,
        "passage": passage,
        "document_paths": document_paths,
        "verified": (
            row.get("VERIFIED (yes/no)", "").strip().lower() == "yes"
        )
    })

    for path in document_paths:
        if path not in record["evidence_document_paths"]:
            record["evidence_document_paths"].append(path)


def convert_dataset():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Golden review CSV not found: {CSV_PATH}"
        )

    document_mapping = load_document_mapping()

    # Save the path-to-document-ID mapping for inspection.
    with OUTPUT_MAPPING.open("w", encoding="utf-8") as file:
        json.dump(
            document_mapping,
            file,
            indent=2,
            ensure_ascii=False
        )

    records = []
    current_record = None
    continuation_rows = 0

    with CSV_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            question_id = row.get("id", "").strip()

            # A row with an ID starts a new question.
            if question_id:
                if current_record is not None:
                    records.append(current_record)

                answerable = (
                    row.get("answerable", "").strip().lower() == "yes"
                )

                difficulty = row.get("difficulty", "").strip()

                current_record = {
                    "id": question_id,
                    "type": row.get("type", "").strip(),
                    "difficulty": difficulty or None,
                    "topic": row.get("topic", "").strip(),
                    "answerable": answerable,
                    "expected_behavior": (
                        "answer" if answerable else "abstain"
                    ),
                    "question": row.get("question", "").strip(),
                    "expected_answer": (
                        row.get("expected_answer", "").strip() or None
                    ),
                    "verified": (
                        row.get("VERIFIED (yes/no)", "")
                        .strip()
                        .lower() == "yes"
                    ),
                    "reviewer_notes": (
                        row.get("reviewer_notes", "").strip() or None
                    ),
                    "evidence": [],
                    "evidence_document_paths": [],
                    "relevant_document_ids": []
                }

                add_evidence(current_record, row)

            # Rows without an ID are continuation evidence rows.
            elif current_record is not None:
                add_evidence(current_record, row)
                continuation_rows += 1

    if current_record is not None:
        records.append(current_record)

    # Resolve all evidence source paths to document IDs.
    unresolved_paths = set()

    for record in records:
        document_ids = set()

        for source_path in record["evidence_document_paths"]:
            matches = resolve_document_ids(
                source_path,
                document_mapping
            )

            if matches:
                document_ids.update(matches)
            else:
                unresolved_paths.add(source_path)

        record["relevant_document_ids"] = sorted(document_ids)

    # Write one JSON object per line.
    with OUTPUT_JSONL.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    answerable_count = sum(
        1 for record in records if record["answerable"]
    )
    unanswerable_count = len(records) - answerable_count
    unverified_questions = [
        record["id"]
        for record in records
        if not record["verified"]
    ]
    questions_without_document_ids = [
        record["id"]
        for record in records
        if record["answerable"]
        and not record["relevant_document_ids"]
    ]

    report = {
        "total_questions": len(records),
        "answerable_questions": answerable_count,
        "unanswerable_questions": unanswerable_count,
        "continuation_rows_combined": continuation_rows,
        "unverified_questions": unverified_questions,
        "questions_without_document_ids": (
            questions_without_document_ids
        ),
        "unresolved_source_paths": sorted(unresolved_paths),
        "output_file": str(OUTPUT_JSONL),
        "mapping_file": str(OUTPUT_MAPPING)
    }

    with OUTPUT_REPORT.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\nGolden dataset conversion completed.")
    print(f"Total questions: {len(records)}")
    print(f"Answerable: {answerable_count}")
    print(f"Unanswerable: {unanswerable_count}")
    print(f"Continuation rows combined: {continuation_rows}")
    print(f"Unverified questions: {len(unverified_questions)}")
    print(
        "Answerable questions without document IDs:",
        len(questions_without_document_ids)
    )
    print(f"Unresolved source paths: {len(unresolved_paths)}")
    print(f"\nCreated: {OUTPUT_JSONL}")
    print(f"Created: {OUTPUT_MAPPING}")
    print(f"Created: {OUTPUT_REPORT}")


if __name__ == "__main__":
    convert_dataset()
