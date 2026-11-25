
from functools import lru_cache

from sentence_transformers import CrossEncoder


@lru_cache(maxsize=4)
def get_cross_encoder(model_name: str):
    """Load and cache Cross-Encoder models."""
    return CrossEncoder(model_name, max_length=512)


class CrossEncoderReranker:
    """Rerank retrieved passages using query-passage relevance."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model_name = model_name
        self.model = get_cross_encoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if not candidates:
            return []

        pairs = [
            (query, candidate.get("text", ""))
            for candidate in candidates
        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        reranked = []

        for candidate, score in zip(candidates, scores):
            item = dict(candidate)
            item["reranker_score"] = float(score)
            reranked.append(item)

        reranked.sort(
            key=lambda item: item["reranker_score"],
            reverse=True,
        )

        return reranked[:top_k]
