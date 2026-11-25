from pathlib import Path
from typing import Any

import chromadb

from app.config import settings
from app.ingestion.embeddings import EmbeddingService


BACKEND_DIR = Path(__file__).resolve().parents[2]

CHROMA_PATH = Path(settings.CHROMA_PATH)

if not CHROMA_PATH.is_absolute():
    CHROMA_PATH = BACKEND_DIR / CHROMA_PATH

COLLECTION_NAME = settings.SEMANTIC_COLLECTION


class SemanticVectorIndexer:
    """Index semantic chunks in a separate ChromaDB collection."""

    def __init__(self):
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_PATH)
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        self.embedding_service = EmbeddingService()

    def index_chunks(
        self,
        chunks: list[dict[str, Any]],
        batch_size: int = 64,
    ) -> int:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        indexed_count = 0
        seen_ids = set()

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]

            ids = [item["chunk_id"] for item in batch]
            texts = [item["text"] for item in batch]

            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate chunk IDs found.")

            if seen_ids.intersection(ids):
                raise ValueError("Duplicate chunk IDs across batches.")

            if any(not text.strip() for text in texts):
                raise ValueError("Empty chunk text found.")

            seen_ids.update(ids)

            embeddings = self.embedding_service.embed_documents(texts)

            metadatas = [
                {
                    "chunk_id": str(item["chunk_id"]),
                    "document_id": str(item["document_id"]),
                    "title": str(item.get("title", "")),
                    "description": str(item.get("description", "")),
                    "source": str(item.get("source", "")),
                    "source_url": str(item.get("source_url", "")),
                    "source_url_status": str(
                        item.get("source_url_status", "local_only")
                    ),
                    "relative_path": str(item.get("relative_path", "")),
                    "file_type": str(item.get("file_type", "")),
                    "author": str(item.get("author", "")),
                    "ms_author": str(item.get("ms_author", "")),
                    "ms_service": str(item.get("ms_service", "")),
                    "ms_topic": str(item.get("ms_topic", "")),
                    "ms_date": str(item.get("ms_date", "")),
                    "chunk_index": int(item["chunk_index"]),
                    "chunking_strategy": "semantic",
                }
                for item in batch
            ]

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            indexed_count += len(batch)

        return indexed_count

    def get_count(self) -> int:
        return self.collection.count()