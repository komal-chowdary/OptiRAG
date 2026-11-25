
from typing import Any


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[str]:
    """Split text into overlapping character-based chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)
    step = chunk_size - chunk_overlap

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


def chunk_document(
    document: dict[str, Any],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[dict[str, Any]]:
    """Split one document while preserving citation metadata."""
    document_id = document.get("document_id")

    if not document_id:
        raise ValueError("Document is missing document_id.")

    chunks = chunk_text(
        text=document.get("text", ""),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    records = []

    for index, text in enumerate(chunks):
        records.append({
            "chunk_id": f"{document_id}_chunk_{index}",
            "document_id": document_id,
            "title": document.get("title", ""),
            "description": document.get("description") or "",
            "source": document.get("source", ""),
            "source_url": document.get("source_url") or "",
            "source_url_status": document.get(
                "source_url_status", "local_only"
            ),
            "relative_path": document.get("relative_path") or "",
            "file_type": document.get("file_type") or "",
            "author": document.get("author") or "",
            "ms_author": document.get("ms_author") or "",
            "ms_service": document.get("ms_service") or "",
            "ms_topic": document.get("ms_topic") or "",
            "ms_date": document.get("ms_date") or "",
            "chunk_index": index,
            "text": text,
            "chunking_strategy": "fixed_size",
        })

    return records
