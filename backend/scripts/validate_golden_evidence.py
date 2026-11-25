
import json
import re
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]

CHUNKS_PATH = BACKEND_DIR / "data" / "processed" / "chunks.jsonl"
GOLDEN_PATH = BACKEND_DIR / "data" / "evaluation" / "golden_dataset.jsonl"
REPORT_PATH = BACKEND_DIR / "data" / "evaluation" / "evidence_validation_report.json"


def normalize_path(path):
    return path.strip().replace("\\", "/").lower().lstrip("./")


def normalize_markdown(text):
    """Remove common Markdown formatting without removing the text."""

    # Convert Markdown links to their visible text.
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Remove bold, italic and strikethrough markers.
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.*?)\*(?!\*)", r"\1", text)
    text = re.sub(r"(?<!_)_(?!_)(.*?)_(?!_)", r"\1", text)
    text = re.sub(r"~~(.*?)~~", r"\1", text)

    # Remove inline code markers.
    text = re.sub(r"`+", "", text)

    # Normalize whitespace and case.
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip().casefold()


def split_alternatives(quote):
    """Keep primary and explicitly accepted alternatives separate."""
    parts = re.split(
        r"\s*\|\|\s*ALSO ACCEPTED:\s*",
        quote,
        flags=re.IGNORECASE
    )

    return [part.strip() for part in parts if part.strip()]


def load_source_lookup():
    source_lookup = {}

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            chunk = json.loads(line)
            source = chunk.get("source")

            if source:
                source_lookup.setdefault(
                    normalize_path(source), set()
                ).add(source)

    return source_lookup


def find_source_files(relative_path, source_lookup):
    target = normalize_path(relative_path)
    matches = set()

    for stored_path, actual_paths in source_lookup.items():
        if stored_path == target or stored_path.endswith("/" + target):
            matches.update(actual_paths)

    return sorted(matches)


def load_golden_dataset():
    records = []

    with GOLDEN_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def validate():
    for path in (CHUNKS_PATH, GOLDEN_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    source_lookup = load_source_lookup()
    records = load_golden_dataset()

    quote_checks = []
    missing_files = []
    missing_quotes = []
    empty_quotes = []

    for record in records:
        if not record["answerable"]:
            continue

        for evidence_index, evidence in enumerate(record["evidence"]):
            original_quote = evidence.get("quote", "").strip()
            source_paths = evidence.get("document_paths", [])

            if not original_quote:
                empty_quotes.append({
                    "question_id": record["id"],
                    "evidence_index": evidence_index
                })
                continue

            candidate_files = []

            for source_path in source_paths:
                matches = find_source_files(
                    source_path, source_lookup
                )

                if not matches:
                    missing_files.append({
                        "question_id": record["id"],
                        "source_path": source_path
                    })
                else:
                    candidate_files.extend(matches)

            candidate_files = sorted(set(candidate_files))

            # Check primary quote and accepted alternatives separately.
            quote_variants = split_alternatives(original_quote)

            for variant_index, quote in enumerate(quote_variants):
                normalized_quote = normalize_markdown(quote)
                found_in = []

                for source in candidate_files:
                    source_file = Path(source)

                    if not source_file.exists():
                        missing_files.append({
                            "question_id": record["id"],
                            "source_path": source,
                            "reason": "File does not exist"
                        })
                        continue

                    content = source_file.read_text(
                        encoding="utf-8",
                        errors="replace"
                    )

                    normalized_content = normalize_markdown(content)

                    if normalized_quote in normalized_content:
                        found_in.append(source)

                result = {
                    "question_id": record["id"],
                    "evidence_index": evidence_index,
                    "variant_index": variant_index,
                    "quote": quote,
                    "quote_found": bool(found_in),
                    "matched_files": found_in,
                    "candidate_file_count": len(candidate_files)
                }

                quote_checks.append(result)

                if not found_in:
                    missing_quotes.append(result)

    report = {
        "total_questions": len(records),
        "answerable_questions": sum(
            1 for record in records if record["answerable"]
        ),
        "evidence_variants_checked": len(quote_checks),
        "evidence_variants_found": sum(
            1 for item in quote_checks if item["quote_found"]
        ),
        "evidence_variants_missing": len(missing_quotes),
        "missing_source_files": len(missing_files),
        "empty_evidence_quotes": len(empty_quotes),
        "quote_validation_results": quote_checks,
        "missing_quotes": missing_quotes,
        "missing_files": missing_files,
        "empty_quotes": empty_quotes
    }

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\nMarkdown-aware evidence validation completed.")
    print(f"Questions: {len(records)}")
    print(f"Evidence variants checked: {len(quote_checks)}")
    print(f"Evidence variants found: {report['evidence_variants_found']}")
    print(f"Evidence variants missing: {len(missing_quotes)}")
    print(f"Missing source files: {len(missing_files)}")
    print(f"Empty evidence quotes: {len(empty_quotes)}")
    print(f"\nReport created: {REPORT_PATH}")


if __name__ == "__main__":
    validate()
