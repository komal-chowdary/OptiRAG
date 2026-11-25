
import pytest

import app.services.rag_service as rag_module

from app.core.exceptions import RetrievalError, GenerationError


class FakeRetrievalService:
    def __init__(self, **kwargs):
        self.retriever = object()

    def search(self, question, top_k=5):
        if question == "trigger retrieval failure":
            raise RuntimeError("Internal database error")

        return {
            "question": question,
            "retrieval_strategy": "test_strategy",
            "result_count": 1,
            "results": [
                {
                    "text": "Azure Blob Storage stores unstructured data.",
                    "metadata": {
                        "document_id": "test-document",
                        "title": "Blob Storage",
                    },
                    "score": 0.95,
                }
            ],
        }


class FakeAnswerGenerator:
    def generate(self, question, retrieved_chunks):
        if question == "trigger generation failure":
            raise GenerationError()

        return {
            "answer": "Azure Blob Storage stores unstructured data.",
            "citations": [],
            "model": "test-model",
        }


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr(
        rag_module,
        "QueryService",
        FakeRetrievalService,
    )
    monkeypatch.setattr(
        rag_module,
        "SemanticQueryService",
        FakeRetrievalService,
    )
    monkeypatch.setattr(
        rag_module,
        "HybridQueryService",
        FakeRetrievalService,
    )
    monkeypatch.setattr(
        rag_module,
        "RerankedQueryService",
        FakeRetrievalService,
    )
    monkeypatch.setattr(
        rag_module,
        "AnswerGenerator",
        FakeAnswerGenerator,
    )

    return rag_module.RAGService()


@pytest.mark.parametrize(
    "strategy",
    [
        "basic_vector",
        "semantic_vector",
        "hybrid",
        "hybrid_reranked",
    ],
)
def test_all_retrieval_strategies(service, strategy):
    result = service.answer_question(
        question="What is Azure Blob Storage?",
        strategy=strategy,
        top_k=5,
    )

    assert result["retrieval_strategy"] == strategy
    assert result["result_count"] == 1
    assert result["answer"]
    assert result["model"] == "test-model"


def test_empty_question_is_rejected(service):
    with pytest.raises(ValueError):
        service.answer_question(question="   ")


def test_unsupported_strategy_is_rejected(service):
    with pytest.raises(ValueError):
        service.answer_question(
            question="Test question",
            strategy="invalid_strategy",
        )


def test_invalid_top_k_is_rejected(service):
    with pytest.raises(ValueError):
        service.answer_question(
            question="Test question",
            top_k=25,
        )


def test_retrieval_failure_becomes_retrieval_error(service):
    with pytest.raises(RetrievalError):
        service.answer_question(
            question="trigger retrieval failure",
        )


def test_generation_failure_is_preserved(service):
    with pytest.raises(GenerationError):
        service.answer_question(
            question="trigger generation failure",
        )
