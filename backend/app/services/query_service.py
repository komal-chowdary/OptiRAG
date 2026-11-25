
from app.retrieval.vector_retriever import VectorRetriever


class QueryService:
    """Application service for document retrieval."""

    def __init__(self):
        self.retriever = VectorRetriever()

    def search(self, question: str, top_k: int = 5) -> dict:
        results = self.retriever.retrieve(
            query=question,
            top_k=top_k,
        )

        return {
            "question": question,
            "retrieval_strategy": "basic_vector",
            "result_count": len(results),
            "results": results,
        }
