
from typing import Any

from app.ingestion.semantic_indexing import SemanticVectorIndexer


class SemanticVectorRetriever:
    """Retrieve semantically chunked documents using vector similarity."""

    def __init__(self):
        self.indexer = SemanticVectorIndexer()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        collection = self.indexer.collection
        total_chunks = collection.count()

        if total_chunks == 0:
            return []

        query_embedding = (
            self.indexer.embedding_service.embed_query(query)
        )

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, total_chunks),
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]

        for chunk_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            metadata = metadata or {}

            retrieved.append({
                "chunk_id": metadata.get("chunk_id") or chunk_id,
                "text": text,
                "metadata": metadata,
                "score": float(distance),
            })

        return retrieved
