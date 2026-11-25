
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = BASE_DIR / "data" / "evaluation"

TARGET_IDS = {"G009", "G012"}

def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]

def find_question_id(record):
    for key in ("id", "question_id", "query_id"):
        if key in record:
            return record[key]
    return None

golden = load_jsonl(EVAL_DIR / "golden_dataset.jsonl")
details = load_jsonl(EVAL_DIR / "retrieval_evaluation_details.jsonl")

print("\n========== GOLDEN DATASET RECORDS ==========")

for record in golden:
    if find_question_id(record) in TARGET_IDS:
        print(json.dumps(record, indent=2, ensure_ascii=False))

print("\n========== RETRIEVAL DETAIL RECORDS ==========")

for record in details:
    if find_question_id(record) in TARGET_IDS:
        print(json.dumps(record, indent=2, ensure_ascii=False))

if not any(find_question_id(r) in TARGET_IDS for r in golden):
    print("Could not identify target IDs in the golden dataset.")

if not any(find_question_id(r) in TARGET_IDS for r in details):
    print("Could not identify target IDs in retrieval details.")
