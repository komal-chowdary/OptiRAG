
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker


class RerankedHybridRetriever:
    """Hybrid retrieval followed by cross-encoder reranking."""

    def __init__(self, hybrid_retriever=None):
        self.hybrid_retriever = (
            hybrid_retriever
            if hybrid_retriever is not None
            else HybridRetriever()
        )

        self.reranker = CrossEncoderReranker()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[dict]:

        candidates = self.hybrid_retriever.retrieve(
            query=query,
            top_k=candidate_k,
            candidate_k=candidate_k,
        )

        return self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
        )
