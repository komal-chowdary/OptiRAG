import logging

from app.core.exceptions import (
    OptiRAGError,
    RetrievalError,
    GenerationError,
)
from app.generation.answer_generator import AnswerGenerator
from app.services.query_service import QueryService
from app.services.semantic_query_service import SemanticQueryService
from app.services.hybrid_query_service import HybridQueryService
from app.services.reranked_query_service import RerankedQueryService

logger = logging.getLogger(__name__)


class RAGService:
    """Coordinate retrieval and evidence-grounded answer generation."""

    def __init__(self):
        self.answer_generator = AnswerGenerator()
        self.retrieval_services = {}

    def _get_retrieval_service(self, strategy: str):
        """Initialize and cache a retrieval service when first requested."""

        if strategy in self.retrieval_services:
            return self.retrieval_services[strategy]

        if strategy == "basic_vector":
            service = QueryService()

        elif strategy == "semantic_vector":
            service = SemanticQueryService()

        elif strategy == "hybrid":
            service = HybridQueryService()

        elif strategy == "hybrid_reranked":
            hybrid_service = self.retrieval_services.get("hybrid")

            if hybrid_service is None:
                hybrid_service = HybridQueryService()
                self.retrieval_services["hybrid"] = hybrid_service

            service = RerankedQueryService(
                hybrid_retriever=hybrid_service.retriever
            )

        else:
            raise ValueError(
                f"Unsupported retrieval strategy: {strategy}"
            )

        self.retrieval_services[strategy] = service
        return service

    def answer_question(
        self,
        question: str,
        strategy: str = "basic_vector",
        top_k: int = 5,
    ) -> dict:

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        supported_strategies = {
            "basic_vector",
            "semantic_vector",
            "hybrid",
            "hybrid_reranked",
        }

        if strategy not in supported_strategies:
            raise ValueError(
                f"Unsupported retrieval strategy: {strategy}"
            )

        if not isinstance(top_k, int) or not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20.")

        try:
            retrieval_service = self._get_retrieval_service(strategy)

            retrieval = retrieval_service.search(
                question=question,
                top_k=top_k,
            )

            if not isinstance(retrieval, dict):
                raise RuntimeError(
                    "Retrieval service returned an invalid response."
                )

            retrieved_chunks = retrieval.get("results")

            if not isinstance(retrieved_chunks, list):
                raise RuntimeError(
                    "Retrieval response does not contain a valid results list."
                )

        except ValueError:
            raise
        except OptiRAGError:
            raise
        except Exception as exc:
            logger.exception(
                "Retrieval failed for strategy '%s'.",
                strategy,
            )
            raise RetrievalError() from exc

        try:
            generated = self.answer_generator.generate(
                question=question,
                retrieved_chunks=retrieved_chunks,
            )

        except ValueError:
            raise
        except GenerationError:
            raise
        except Exception as exc:
            logger.exception(
                "Unexpected generation failure for strategy '%s'.",
                strategy,
            )
            raise GenerationError() from exc

        return {
            "question": question,
            "retrieval_strategy": strategy,
            "result_count": len(retrieved_chunks),
            "answer": generated["answer"],
            "citations": generated["citations"],
            "model": generated["model"],
            "retrieved_context": retrieved_chunks,
        }
