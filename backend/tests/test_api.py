
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.routes import query as query_module
from app.core.exceptions import RetrievalError, GenerationError


client = TestClient(app)


class FakeRAGService:
    def answer_question(self, question, strategy="basic_vector", top_k=5):
        if question == "trigger retrieval failure":
            raise RetrievalError()

        if question == "trigger generation failure":
            raise GenerationError()

        return {
            "question": question,
            "retrieval_strategy": strategy,
            "result_count": 1,
            "answer": "Azure Blob Storage stores unstructured data.",
            "citations": [],
            "model": "test-model",
            "retrieved_context": [],
        }


class FakeRetrievalService:
    def search(self, question, top_k=5):
        return {
            "question": question,
            "retrieval_strategy": "test_strategy",
            "result_count": 1,
            "results": [
                {
                    "text": "Azure Blob Storage stores unstructured data.",
                    "metadata": {},
                    "score": 0.95,
                }
            ],
        }


@pytest.fixture(autouse=True)
def mock_services(monkeypatch):
    monkeypatch.setattr(
        query_module,
        "rag_service",
        FakeRAGService(),
    )

    monkeypatch.setattr(
        query_module,
        "query_service",
        FakeRetrievalService(),
    )

    monkeypatch.setattr(
        query_module,
        "hybrid_query_service",
        FakeRetrievalService(),
    )

    monkeypatch.setattr(
        query_module,
        "semantic_query_service",
        FakeRetrievalService(),
    )

    monkeypatch.setattr(
        query_module,
        "reranked_query_service",
        FakeRetrievalService(),
    )


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "alive"
    assert "environment" in response.json()

def test_readiness_endpoint(monkeypatch):
    from app.main import health_service

    monkeypatch.setattr(
        health_service,
        "check_readiness",
        lambda: {
            "status": "ready",
            "checks": {
                "chromadb": {"status": "available"},
                "ollama": {"status": "available"},
            },
        },
    )

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_readiness_failure(monkeypatch):
    from app.main import health_service

    monkeypatch.setattr(
        health_service,
        "check_readiness",
        lambda: {
            "status": "not_ready",
            "checks": {
                "chromadb": {"status": "unavailable"},
                "ollama": {"status": "available"},
            },
        },
    )

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"

@pytest.mark.parametrize(
    "endpoint",
    [
        "/api/query/retrieve",
        "/api/query/answer",
        "/api/query/hybrid-retrieve",
        "/api/query/semantic-retrieve",
        "/api/query/reranked-retrieve",
        "/api/query/answer-strategy",
    ],
)
def test_query_endpoints_reject_empty_question(endpoint):
    response = client.post(
        endpoint,
        json={"question": ""},
    )

    assert response.status_code == 422


def test_basic_retrieval_endpoint():
    response = client.post(
        "/api/query/retrieve",
        json={
            "question": "What is Azure Blob Storage?",
            "top_k": 5,
        },
    )

    assert response.status_code == 200
    assert response.json()["result_count"] == 1


def test_hybrid_retrieval_endpoint():
    response = client.post(
        "/api/query/hybrid-retrieve",
        json={"question": "What is Azure Blob Storage?"},
    )

    assert response.status_code == 200


def test_semantic_retrieval_endpoint():
    response = client.post(
        "/api/query/semantic-retrieve",
        json={"question": "What is Azure Blob Storage?"},
    )

    assert response.status_code == 200


def test_reranked_retrieval_endpoint():
    response = client.post(
        "/api/query/reranked-retrieve",
        json={"question": "What is Azure Blob Storage?"},
    )

    assert response.status_code == 200


@pytest.mark.parametrize(
    "strategy",
    [
        "basic_vector",
        "semantic_vector",
        "hybrid",
        "hybrid_reranked",
    ],
)
def test_answer_strategy_endpoint(strategy):
    response = client.post(
        "/api/query/answer-strategy",
        json={
            "question": "What is Azure Blob Storage?",
            "strategy": strategy,
            "top_k": 5,
        },
    )

    assert response.status_code == 200
    assert response.json()["retrieval_strategy"] == strategy
    assert response.json()["model"] == "test-model"


def test_invalid_top_k():
    response = client.post(
        "/api/query/retrieve",
        json={
            "question": "Test question",
            "top_k": 25,
        },
    )

    assert response.status_code == 422


def test_invalid_strategy():
    response = client.post(
        "/api/query/answer-strategy",
        json={
            "question": "Test question",
            "strategy": "invalid_strategy",
        },
    )

    assert response.status_code == 422


def test_retrieval_error_response():
    response = client.post(
        "/api/query/answer-strategy",
        json={"question": "trigger retrieval failure"},
    )

    assert response.status_code == 503
    assert response.json()["error"] == "RetrievalError"


def test_generation_error_response():
    response = client.post(
        "/api/query/answer-strategy",
        json={"question": "trigger generation failure"},
    )

    assert response.status_code == 503
    assert response.json()["error"] == "GenerationError"
