
from functools import lru_cache

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=4)
def get_embedding_model(model_name: str):
    """
    Load and cache embedding models.
    Repeated calls with the same model name reuse one instance.
    """
    return SentenceTransformer(model_name)


class EmbeddingService:
    """Generate vector embeddings for documents and queries."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model_name = model_name
        self.model = get_embedding_model(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Convert document chunks into vectors."""
        if not texts:
            return []

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Convert a user query into a vector."""
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        vector = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        return vector.tolist()
