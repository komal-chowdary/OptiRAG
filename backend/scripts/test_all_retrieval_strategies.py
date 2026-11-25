
from app.services.rag_service import RAGService


def main():
    question = "What is Azure Blob Storage?"

    rag_service = RAGService()

    strategies = [
        "basic_vector",
        "semantic_vector",
        "hybrid",
        "hybrid_reranked",
    ]

    print("\nOPTIRAG RETRIEVAL SMOKE TEST")
    print("=" * 75)
    print(f"Question: {question}")

    for strategy in strategies:
        print("\n" + "=" * 75)
        print(f"STRATEGY: {strategy}")
        print("=" * 75)

        try:
            service = rag_service.retrieval_services[strategy]

            response = service.search(
                question=question,
                top_k=5,
            )

            results = response.get("results", [])

            print(f"Result count: {len(results)}")

            if not results:
                print("WARNING: No results returned.")
                continue

            for rank, item in enumerate(results, start=1):
                metadata = item.get("metadata") or {}

                print(f"\nRank: {rank}")
                print(f"Document ID: {item.get('document_id') or metadata.get('document_id')}")
                print(f"Chunk ID: {item.get('chunk_id') or metadata.get('chunk_id')}")
                print(f"Score: {item.get('score')}")
                print(
                    f"Reranker score: "
                    f"{item.get('reranker_score', 'N/A')}"
                )
                print(
                    f"Text: {item.get('text', '')[:250]}"
                )

                if not metadata.get("document_id"):
                    print("WARNING: document_id is missing!")

        except Exception as exc:
            print(f"ERROR in {strategy}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
