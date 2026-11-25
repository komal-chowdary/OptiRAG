
from app.retrieval.hybrid_retriever import HybridRetriever


class HybridQueryService:
    """Application service for keyword and vector retrieval."""

    def __init__(self):
        self.retriever = HybridRetriever()

    def search(self, question: str, top_k: int = 5) -> dict:
        results = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            candidate_k=max(20, top_k),
        )

        return {
            "question": question,
            "retrieval_strategy": "hybrid_bm25_vector_rrf",
            "result_count": len(results),
            "results": results,
        }
