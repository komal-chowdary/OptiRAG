
from app.retrieval.semantic_vector_retriever import (
    SemanticVectorRetriever,
)


class SemanticQueryService:
    """Application service for semantic chunking and vector retrieval."""

    def __init__(self):
        self.retriever = SemanticVectorRetriever()

    def search(self, question: str, top_k: int = 5) -> dict:
        results = self.retriever.retrieve(
            query=question,
            top_k=top_k,
        )

        return {
            "question": question,
            "retrieval_strategy": "semantic_chunking_vector",
            "result_count": len(results),
            "results": results,
        }
