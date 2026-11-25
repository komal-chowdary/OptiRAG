
from app.ingestion.embeddings import (
    EmbeddingService,
    get_embedding_model,
)

from app.retrieval.reranker import (
    CrossEncoderReranker,
    get_cross_encoder,
)


def test_embedding_model_is_shared():
    service1 = EmbeddingService()
    service2 = EmbeddingService()

    assert service1.model is service2.model


def test_cross_encoder_is_shared():
    reranker1 = CrossEncoderReranker()
    reranker2 = CrossEncoderReranker()

    assert reranker1.model is reranker2.model
