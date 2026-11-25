
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.ingestion.document_loader import load_document
from app.ingestion.preprocessing import extract_front_matter, clean_text


test_file = (
    BACKEND_DIR
    / "data"
    / "raw"
    / "microsoft_docs"
    / "networking"
    / "architecture-guides.md"
)

document = load_document(test_file)

metadata, content = extract_front_matter(document["text"])
cleaned_content = clean_text(content)

print("\n--- EXTRACTED METADATA ---")
print(metadata)

print("\n--- CLEANED DOCUMENT CONTENT ---")
print(cleaned_content[:1500])

assert metadata.get("title") == "Azure Networking architecture documentation"
assert "Customer intent" not in cleaned_content
assert "# Azure Networking architecture documentation" in cleaned_content
assert "| Title | Description |" in cleaned_content

print("\nPreprocessing test passed!")
