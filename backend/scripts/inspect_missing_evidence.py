
import json
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]

REPORT_PATH = (
    BACKEND_DIR
    / "data"
    / "evaluation"
    / "evidence_validation_report.json"
)

with REPORT_PATH.open("r", encoding="utf-8") as file:
    report = json.load(file)

missing = report.get("missing_quotes", [])

print(f"\nTotal unmatched quotes: {len(missing)}\n")

for item in missing:
    print("=" * 90)
    print(f"Question ID: {item.get('question_id')}")
    print(f"Evidence index: {item.get('evidence_index')}")
    print(f"Quote:\n{item.get('quote')}")
    print("\nCandidate files:")

    for source in item.get("candidate_files", []):
        print(f"  - {source}")

    if not item.get("candidate_files"):
        print("  No candidate files found.")
