
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from app.retrieval.vector_retriever import VectorRetriever


BACKEND_DIR = Path(__file__).resolve().parents[2]
CHUNKS_FILE = BACKEND_DIR / "data" / "processed" / "chunks.jsonl"


def tokenize(text: str) -> list[str]:
    """Tokenize text for keyword retrieval."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def get_chunk_id(item: dict[str, Any]) -> str:
    """Extract a stable chunk ID from a retrieved chunk."""

    metadata = item.get("metadata") or {}

    chunk_id = (
        item.get("chunk_id")
        or metadata.get("chunk_id")
    )

    if chunk_id:
        return str(chunk_id)

    # Fallback for records without an explicit chunk ID.
    identity = "|".join([
        str(metadata.get("document_id", "")),
        str(metadata.get("source", "")),
        str(metadata.get("chunk_index", "")),
        str(item.get("text", "")),
    ])

    return hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()


def get_metadata(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize metadata from BM25 and vector results."""

    if isinstance(item.get("metadata"), dict):
        metadata = dict(item["metadata"])
    else:
        metadata = {
            key: value
            for key, value in item.items()
            if key not in {"text", "score", "chunk_id"}
        }

    metadata.setdefault("chunk_id", get_chunk_id(item))

    return metadata


class HybridRetriever:
    """Combine BM25 and vector retrieval using RRF."""

    def __init__(self):
        if not CHUNKS_FILE.exists():
            raise FileNotFoundError(
                f"Chunks file not found: {CHUNKS_FILE}. "
                "Run ingestion and chunking first."
            )

        self.chunks: list[dict[str, Any]] = []

        with CHUNKS_FILE.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in chunks.jsonl "
                        f"at line {line_number}."
                    ) from exc

                if chunk.get("text", "").strip():
                    self.chunks.append(chunk)

        if not self.chunks:
            raise ValueError(
                "No usable chunks found for BM25 indexing."
            )

        tokenized_chunks = [
            tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(tokenized_chunks)
        self.vector_retriever = VectorRetriever()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
        rrf_constant: int = 60,
    ) -> list[dict[str, Any]]:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if candidate_k < 1:
            raise ValueError("candidate_k must be at least 1.")

        if rrf_constant < 1:
            raise ValueError("rrf_constant must be at least 1.")

        candidate_k = min(candidate_k, len(self.chunks))

        # -------------------------------------------------
        # 1. BM25 keyword retrieval
        # -------------------------------------------------

        stop_words = {
            "what", "is", "the", "a", "an", "of",
            "in", "on", "for", "to", "and", "or",
            "how", "does", "do", "can", "are", "with"
        }

        query_tokens = [
            token
            for token in tokenize(query)
            if token not in stop_words
        ]

        bm25_scores = self.bm25.get_scores(query_tokens)

        # Exclude documents with zero or negative relevance.
        bm25_indices = [
            index
            for index, score in enumerate(bm25_scores)
            if float(score) > 0
        ]

        bm25_indices.sort(
            key=lambda index: float(bm25_scores[index]),
            reverse=True,
        )

        bm25_indices = bm25_indices[:candidate_k]

        bm25_ranked = [
            {
                "chunk_id": str(self.chunks[index]["chunk_id"]),
                "text": self.chunks[index]["text"],
                "metadata": {
                    key: value
                    for key, value in self.chunks[index].items()
                    if key != "text"
                },
            }
            for index in bm25_indices
        ]

        # -------------------------------------------------
        # 2. Vector similarity retrieval
        # -------------------------------------------------

        vector_ranked = self.vector_retriever.retrieve(
            query=query,
            top_k=candidate_k,
        )

        # -------------------------------------------------
        # 3. Reciprocal Rank Fusion
        # -------------------------------------------------

        fused_scores: dict[str, float] = {}
        chunk_lookup: dict[str, dict[str, Any]] = {}

        for ranked_chunks in (bm25_ranked, vector_ranked):

            for rank, item in enumerate(ranked_chunks, start=1):

                text = item.get("text", "").strip()

                if not text:
                    continue

                chunk_id = get_chunk_id(item)
                metadata = get_metadata(item)

                fused_scores[chunk_id] = (
                    fused_scores.get(chunk_id, 0.0)
                    + 1.0 / (rrf_constant + rank)
                )

                if chunk_id not in chunk_lookup:
                    chunk_lookup[chunk_id] = {
                        "chunk_id": chunk_id,
                        "text": text,
                        "metadata": metadata,
                    }
                else:
                    # Preserve metadata from both retrieval sources.
                    chunk_lookup[chunk_id]["metadata"].update(
                        metadata
                    )

        
        # -------------------------------------------------
        # 4. Sort fused results
        # -------------------------------------------------

        ordered_ids = sorted(
            fused_scores,
            key=lambda chunk_id: fused_scores[chunk_id],
            reverse=True,
        )

        results = []

        for chunk_id in ordered_ids[:top_k]:
            results.append({
                **chunk_lookup[chunk_id],
                "score": fused_scores[chunk_id],
            })

        return results
