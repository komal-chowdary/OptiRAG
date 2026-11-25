
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.ingestion.document_loader import load_document


test_files = [
    BACKEND_DIR / "data/raw/microsoft_docs/networking/architecture-guides.md",
    BACKEND_DIR / "data/raw/microsoft_docs/storage/blobs/access-tiers-best-practices.md",
    BACKEND_DIR / "data/raw/microsoft_docs/sample.md",
]

for path in test_files:
    document = load_document(path)

    print("\nFile:", path.name)
    print("Title:", document["title"])
    print("Relative path:", document["relative_path"])
    print("Source URL:", document["source_url"])
    print("URL status:", document["source_url_status"])

print("\nSource metadata test completed.")
