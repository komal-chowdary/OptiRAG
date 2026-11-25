
import re
from typing import Any

from sentence_transformers import SentenceTransformer, util


class SemanticChunker:
    """Split documents at semantic boundaries using sentence embeddings."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.55,
        max_chunk_chars: int = 1800,
    ):
        if not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                "similarity_threshold must be between -1 and 1."
            )

        if max_chunk_chars < 1:
            raise ValueError("max_chunk_chars must be positive.")

        self.similarity_threshold = similarity_threshold
        self.max_chunk_chars = max_chunk_chars
        self.model = SentenceTransformer(model_name)

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        """Split text into sentence-like units."""
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return []

        sentences = re.split(
            r"(?<=[.!?])\s+(?=[A-Z0-9])",
            text,
        )

        # Keep short documents and text without punctuation usable.
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def chunk_text(self, text: str) -> list[str]:
        """Group semantically related sentences into chunks."""
        sentences = self.split_sentences(text)

        if not sentences:
            return []

        if len(sentences) == 1:
            return sentences

        embeddings = self.model.encode(
            sentences,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        chunks: list[str] = []
        current_sentences = [sentences[0]]
        current_length = len(sentences[0])

        for index in range(1, len(sentences)):
            previous_embedding = embeddings[index - 1]
            current_embedding = embeddings[index]

            similarity = float(
                util.cos_sim(
                    previous_embedding,
                    current_embedding,
                ).item()
            )

            sentence = sentences[index]
            proposed_length = current_length + 1 + len(sentence)

            # Split when meaning changes or the chunk becomes too large.
            should_split = (
                similarity < self.similarity_threshold
                or proposed_length > self.max_chunk_chars
            )

            if should_split:
                chunks.append(" ".join(current_sentences))
                current_sentences = [sentence]
                current_length = len(sentence)
            else:
                current_sentences.append(sentence)
                current_length = proposed_length

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks

    
    def chunk_document(
        self,
        document: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Chunk one document while preserving citation metadata."""
        document_id = document.get("document_id")

        if not document_id:
            raise ValueError("Document is missing document_id.")

        chunks = self.chunk_text(document.get("text", ""))

        records = []

        for index, text in enumerate(chunks):
            records.append({
                "chunk_id": f"{document_id}_semantic_{index}",
                "document_id": document_id,
                "title": document.get("title", ""),
                "description": document.get("description") or "",
                "source": document.get("source", ""),
                "source_url": document.get("source_url") or "",
                "source_url_status": document.get(
                    "source_url_status", "local_only"
                ),
                "relative_path": document.get("relative_path") or "",
                "file_type": document.get("file_type") or "",
                "author": document.get("author") or "",
                "ms_author": document.get("ms_author") or "",
                "ms_service": document.get("ms_service") or "",
                "ms_topic": document.get("ms_topic") or "",
                "ms_date": document.get("ms_date") or "",
                "chunk_index": index,
                "text": text,
                "chunking_strategy": "semantic",
            })

        return records
