
from app.retrieval.reranked_hybrid_retriever import (
    RerankedHybridRetriever,
)


class RerankedQueryService:
    """Application service for hybrid retrieval with reranking."""

    def __init__(self, hybrid_retriever=None):
        self.retriever = RerankedHybridRetriever(
            hybrid_retriever=hybrid_retriever
        )

    def search(self, question: str, top_k: int = 5) -> dict:
        results = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            candidate_k=max(20, top_k),
        )

        return {
            "question": question,
            "retrieval_strategy": "hybrid_bm25_vector_reranking",
            "result_count": len(results),
            "results": results,
        }
