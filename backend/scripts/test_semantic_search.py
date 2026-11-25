
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.retrieval.semantic_vector_retriever import (
    SemanticVectorRetriever,
)


def main():
    retriever = SemanticVectorRetriever()

    questions = [
        "How does Azure Blob Storage organize data?",
        "How can Azure virtual networks control network traffic?",
    ]

    for question in questions:
        print(f"\n{'=' * 70}")
        print(f"Question: {question}")
        print("=" * 70)

        results = retriever.retrieve(
            query=question,
            top_k=5,
        )

        for rank, result in enumerate(results, start=1):
            metadata = result["metadata"]

            print(f"\nResult {rank}")
            print(f"Title: {metadata.get('title', '')}")
            print(f"Source: {metadata.get('source', '')}")
            print(f"Cosine distance: {result['score']:.4f}")
            print(f"Text: {result['text'][:500]}")


if __name__ == "__main__":
    main()
