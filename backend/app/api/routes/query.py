
import logging
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.exceptions import OptiRAGError, RetrievalError
from app.services.rag_service import RAGService


logger = logging.getLogger(__name__)

router = APIRouter(tags=["Query"])

rag_service = RAGService()


class LazyRetrievalService:
    """Resolve a retrieval service only when a request uses it."""

    def __init__(self, strategy: str):
        self.strategy = strategy

    def search(self, question: str, top_k: int = 5):
        service = rag_service._get_retrieval_service(self.strategy)
        return service.search(question=question, top_k=top_k)


# Preserve module-level service names for API compatibility and testing.
query_service = LazyRetrievalService("basic_vector")
semantic_query_service = LazyRetrievalService("semantic_vector")
hybrid_query_service = LazyRetrievalService("hybrid")
reranked_query_service = LazyRetrievalService("hybrid_reranked")


class QueryRequest(BaseModel):
    question: str = Field(
        min_length=1,
        description="Question about the indexed Microsoft documentation",
    )
    top_k: int = Field(default=5, ge=1, le=20)


class RAGRequest(BaseModel):
    question: str = Field(
        min_length=1,
        description="Question about Microsoft documentation",
    )
    strategy: Literal[
        "basic_vector",
        "semantic_vector",
        "hybrid",
        "hybrid_reranked",
    ] = Field(
        default="basic_vector",
        description="Retrieval strategy to use",
    )
    top_k: int = Field(default=5, ge=1, le=20)


def _retrieve(service, question: str, top_k: int):
    """Run a retrieval service with consistent error handling."""
    try:
        return service.search(
            question=question,
            top_k=top_k,
        )

    except ValueError:
        raise

    except OptiRAGError:
        raise

    except Exception as exc:
        logger.exception("Document retrieval endpoint failed.")
        raise RetrievalError() from exc


@router.post("/query/retrieve")
def retrieve_documents(request: QueryRequest):
    return _retrieve(
        query_service,
        request.question,
        request.top_k,
    )


@router.post("/query/answer")
def answer_question(request: QueryRequest):
    result = rag_service.answer_question(
        question=request.question,
        strategy="basic_vector",
        top_k=request.top_k,
    )

    return {
        "question": result["question"],
        "retrieval_strategy": "basic_vector",
        "result_count": result["result_count"],
        "answer": result["answer"],
        "citations": result["citations"],
        "model": result["model"],
    }


@router.post("/query/hybrid-retrieve")
def retrieve_hybrid_documents(request: QueryRequest):
    return _retrieve(
        hybrid_query_service,
        request.question,
        request.top_k,
    )


@router.post("/query/semantic-retrieve")
def retrieve_semantic_documents(request: QueryRequest):
    return _retrieve(
        semantic_query_service,
        request.question,
        request.top_k,
    )


@router.post("/query/reranked-retrieve")
def retrieve_reranked_documents(request: QueryRequest):
    return _retrieve(
        reranked_query_service,
        request.question,
        request.top_k,
    )


@router.post("/query/answer-strategy")
def answer_with_selected_strategy(request: RAGRequest):
    return rag_service.answer_question(
        question=request.question,
        strategy=request.strategy,
        top_k=request.top_k,
    )
