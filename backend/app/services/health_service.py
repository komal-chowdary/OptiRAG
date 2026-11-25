
import logging
from pathlib import Path

import chromadb
import requests

from app.config import settings
from app.ingestion.indexing import BACKEND_DIR


logger = logging.getLogger(__name__)


class HealthService:
    """Check availability of OptiRAG backend dependencies."""

    def check_liveness(self) -> dict:
        return {
            "status": "alive",
            "environment": settings.APP_ENV,
        }

    def check_readiness(self) -> dict:
        checks = {}

        # 1. Check ChromaDB storage directory.
        try:
            chroma_path = Path(settings.CHROMA_PATH)

            if not chroma_path.is_absolute():
                chroma_path = BACKEND_DIR / chroma_path

            if not chroma_path.exists():
                raise FileNotFoundError("ChromaDB directory not found.")

            client = chromadb.PersistentClient(
                path=str(chroma_path)
            )

            vector_collection = client.get_collection(
                settings.VECTOR_COLLECTION
            )

            semantic_collection = client.get_collection(
                settings.SEMANTIC_COLLECTION
            )

            vector_count = vector_collection.count()
            semantic_count = semantic_collection.count()

            collections_populated = (
                vector_count > 0 and semantic_count > 0
            )

            checks["chromadb"] = {
                "status": (
                    "available"
                    if collections_populated
                    else "empty_collection"
                ),
                "vector_chunks": vector_count,
                "semantic_chunks": semantic_count,
            }

        except Exception:
            logger.exception("ChromaDB readiness check failed.")
            checks["chromadb"] = {
                "status": "unavailable",
            }

        # 2. Check Ollama and configured model availability.
        try:
            response = requests.get(
                f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags",
                timeout=3,
            )
            response.raise_for_status()

            available_models = [
                model.get("name", "")
                for model in response.json().get("models", [])
            ]

            model_available = any(
                model == settings.OLLAMA_MODEL
                or model.startswith(settings.OLLAMA_MODEL + ":")
                for model in available_models
            )

            checks["ollama"] = {
                "status": (
                    "available" if model_available else "model_unavailable"
                ),
                "model": settings.OLLAMA_MODEL,
            }

        except Exception:
            logger.exception("Ollama readiness check failed.")
            checks["ollama"] = {
                "status": "unavailable",
                "model": settings.OLLAMA_MODEL,
            }

        ready = all(
            check["status"] == "available"
            for check in checks.values()
        )

        return {
            "status": "ready" if ready else "not_ready",
            "checks": checks,
        }
