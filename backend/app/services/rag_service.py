
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

        # Initialize the hybrid service only once.
        self.hybrid_query_service = HybridQueryService()

        # Reuse its HybridRetriever for reranked retrieval.
        self.reranked_query_service = RerankedQueryService(
            hybrid_retriever=self.hybrid_query_service.retriever
        )

        self.retrieval_services = {
            "basic_vector": QueryService(),
            "semantic_vector": SemanticQueryService(),
            "hybrid": self.hybrid_query_service,
            "hybrid_reranked": self.reranked_query_service,
        }

    def answer_question(
        self,
        question: str,
        strategy: str = "basic_vector",
        top_k: int = 5,
    ) -> dict:

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if strategy not in self.retrieval_services:
            raise ValueError(
                f"Unsupported retrieval strategy: {strategy}"
            )

        if not isinstance(top_k, int) or not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20.")

        retrieval_service = self.retrieval_services[strategy]

        try:
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
