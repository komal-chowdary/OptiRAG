
import json
import re
from pathlib import Path
from difflib import SequenceMatcher


BACKEND_DIR = Path(__file__).resolve().parents[1]

REPORT_PATH = (
    BACKEND_DIR
    / "data"
    / "evaluation"
    / "evidence_validation_report.json"
)

OUTPUT_PATH = (
    BACKEND_DIR
    / "data"
    / "evaluation"
    / "evidence_diagnostics.txt"
)


def normalize(text):
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().casefold()


def split_alternatives(quote):
    """Separate explicitly marked alternative answers."""
    parts = re.split(
        r"\s*\|\|\s*ALSO ACCEPTED:\s*",
        quote,
        flags=re.IGNORECASE
    )
    return [part.strip() for part in parts if part.strip()]


def find_best_matches(quote, content, limit=2):
    """
    Compare the quote with nearby word windows in the source.
    This is diagnostic only; it does not approve or modify evidence.
    """
    source_words = list(re.finditer(r"\S+", content))
    quote_words = quote.split()

    if not quote_words or not source_words:
        return []

    target_length = len(quote_words)
    window_sizes = {
        max(1, target_length - 5),
        target_length,
        target_length + 5
    }

    candidates = []

    for window_size in window_sizes:
        for start in range(0, len(source_words), 2):
            end = start + window_size

            if end > len(source_words):
                break

            char_start = source_words[start].start()
            char_end = source_words[end - 1].end()

            snippet = content[char_start:char_end]

            score = SequenceMatcher(
                None,
                normalize(quote),
                normalize(snippet)
            ).ratio()

            candidates.append((score, snippet))

    candidates.sort(key=lambda item: item[0], reverse=True)

    results = []
    seen = set()

    for score, snippet in candidates:
        cleaned = normalize(snippet)

        if cleaned in seen:
            continue

        seen.add(cleaned)
        results.append((score, snippet))

        if len(results) >= limit:
            break

    return results


def main():
    with REPORT_PATH.open("r", encoding="utf-8") as file:
        report = json.load(file)

    output = []

    for item in report.get("missing_quotes", []):
        question_id = item["question_id"]
        quote = item["quote"]

        output.append("=" * 100)
        output.append(f"QUESTION: {question_id}")

        alternatives = split_alternatives(quote)

        for alternative_index, alternative in enumerate(alternatives):
            output.append(
                f"\nEVIDENCE VERSION {alternative_index + 1}:"
            )
            output.append(alternative)

            for source in item.get("candidate_files", []):
                source_path = Path(source)

                if not source_path.exists():
                    continue

                content = source_path.read_text(
                    encoding="utf-8",
                    errors="replace"
                )

                matches = find_best_matches(alternative, content)

                output.append(f"\nSOURCE: {source}")

                for rank, (score, snippet) in enumerate(matches, 1):
                    output.append(
                        f"\nMATCH {rank} — similarity: {score:.3f}"
                    )
                    output.append(snippet)

    OUTPUT_PATH.write_text(
        "\n".join(output),
        encoding="utf-8"
    )

    print("Evidence diagnostics completed.")
    print(f"Questions inspected: {len(report.get('missing_quotes', []))}")
    print(f"Report created: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
