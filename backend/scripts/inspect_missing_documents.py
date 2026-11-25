
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PROCESSED = BASE / "data" / "processed"
EVAL = BASE / "data" / "evaluation"

TARGETS = [
    "data-protection-overview.md",
    "storage-blob-object-model.md",
    "storage-blobs-introduction.md",
]

def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def search_values(obj, target):
    if isinstance(obj, dict):
        return any(search_values(v, target) for v in obj.values())
    if isinstance(obj, list):
        return any(search_values(v, target) for v in obj)
    if isinstance(obj, str):
        return target.lower() in obj.lower()
    return False

for filename in ["documents.jsonl", "chunks.jsonl", "semantic_chunks.jsonl"]:
    path = PROCESSED / filename

    if not path.exists():
        continue

    records = load_jsonl(path)
    print(f"\n========== {filename} ==========")

    for target in TARGETS:
        matches = [
            r for r in records
            if search_values(r.get("source", ""), target)
            or search_values(r.get("file_path", ""), target)
            or search_values(r.get("path", ""), target)
        ]

        print(f"\nTarget: {target}")
        print(f"Exact source/path matches: {len(matches)}")

        for record in matches[:5]:
            print(json.dumps({
                "document_id": record.get("document_id"),
                "chunk_id": record.get("chunk_id"),
                "title": record.get("title"),
                "source": record.get("source"),
                "file_path": record.get("file_path"),
                "path": record.get("path"),
                "chunk_index": record.get("chunk_index"),
                "text_preview": record.get("text", "")[:500],
            }, indent=2, ensure_ascii=False))

print("\n========== GOLDEN MAPPING ==========")

mapping_path = EVAL / "document_id_mapping.json"
mapping = json.loads(mapping_path.read_text(encoding="utf-8"))

for target in TARGETS:
    print(f"\n{target}")
    for path, ids in mapping.items():
        if path.lower().endswith(target.lower()):
            print("Mapped path:", path)
            print("Mapped IDs:", ids)
