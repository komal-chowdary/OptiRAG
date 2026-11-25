
import pytest

from app.ingestion.chunking import chunk_text, chunk_document


def test_empty_text():
    assert chunk_text("") == []


def test_short_text():
    assert chunk_text("OptiRAG", 100, 10) == ["OptiRAG"]


def test_chunk_overlap():
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    chunks = chunk_text(
        text,
        chunk_size=10,
        chunk_overlap=3,
    )

    assert chunks[0] == "ABCDEFGHIJ"
    assert chunks[1] == "HIJKLMNOPQ"


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("Hello", chunk_size=0)


def test_document_metadata():
    document = {
        "document_id": "doc_1",
        "title": "Azure Storage",
        "source": "sample.md",
        "text": "Azure Storage provides cloud storage.",
    }

    chunks = chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0]["document_id"] == "doc_1"
    assert chunks[0]["title"] == "Azure Storage"
    assert chunks[0]["chunk_index"] == 0

