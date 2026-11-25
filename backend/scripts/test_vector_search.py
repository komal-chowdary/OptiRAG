
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.ingestion.indexing import VectorIndexer  # noqa: E402


def main():
    indexer = VectorIndexer()

    if indexer.get_count() == 0:
        print("No indexed chunks found. Run index_documents.py first.")
        return

    query = input("Enter your question: ").strip()

    if not query:
        print("Please enter a question.")
        return

    query_embedding = indexer.embedding_service.embed_query(query)

    results = indexer.collection.query(
        query_embeddings=[query_embedding],
        n_results=min(3, indexer.get_count()),
        include=["documents", "metadatas", "distances"],
    )

    for rank, (text, metadata, distance) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        start=1,
    ):
        print(f"\n--- Result {rank} ---")
        print(f"Title: {metadata.get('title', '')}")
        print(f"Source: {metadata.get('source', '')}")
        print(f"Cosine distance: {distance:.4f}")
        print(f"Text: {text}")


if __name__ == "__main__":
    main()
