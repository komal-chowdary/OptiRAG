
from app.retrieval.hybrid_retriever import HybridRetriever


class FakeBM25:
    def __init__(self, scores):
        self.scores = scores

    def get_scores(self, tokens):
        return self.scores


class FakeVectorRetriever:
    def __init__(self, results):
        self.results = results

    def retrieve(self, query, top_k):
        return self.results[:top_k]


def create_retriever(chunks, bm25_scores, vector_results):
    retriever = HybridRetriever.__new__(HybridRetriever)

    retriever.chunks = chunks
    retriever.bm25 = FakeBM25(bm25_scores)
    retriever.vector_retriever = FakeVectorRetriever(
        vector_results
    )

    return retriever


def test_zero_score_bm25_candidates_are_filtered():
    chunks = [
        {
            "chunk_id": "chunk_1",
            "text": "Azure Blob Storage stores objects.",
        },
        {
            "chunk_id": "chunk_2",
            "text": "Unrelated networking information.",
        },
    ]

    retriever = create_retriever(
        chunks,
        bm25_scores=[1.5, 0.0],
        vector_results=[],
    )

    results = retriever.retrieve(
        query="Azure Blob Storage",
        top_k=5,
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk_1"


def test_same_chunk_from_both_retrievers_is_merged():
    chunks = [
        {
            "chunk_id": "chunk_1",
            "text": "Azure Blob Storage stores objects.",
            "title": "Blob Storage",
        },
    ]

    vector_results = [
        {
            "text": "Azure Blob Storage stores objects.",
            "metadata": {
                "chunk_id": "chunk_1",
                "title": "Blob Storage",
            },
            "score": 0.2,
        }
    ]

    retriever = create_retriever(
        chunks,
        bm25_scores=[1.5],
        vector_results=vector_results,
    )

    results = retriever.retrieve(
        query="Azure Blob Storage",
        top_k=5,
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk_1"

    # Both retrieval systems contributed to the RRF score.
    assert results[0]["score"] > 0


def test_identical_text_with_different_ids_is_preserved():
    chunks = [
        {
            "chunk_id": "document_A_chunk_1",
            "text": "Azure provides cloud storage.",
            "title": "Document A",
        },
        {
            "chunk_id": "document_B_chunk_1",
            "text": "Azure provides cloud storage.",
            "title": "Document B",
        },
    ]

    retriever = create_retriever(
        chunks,
        bm25_scores=[1.5, 1.2],
        vector_results=[],
    )

    results = retriever.retrieve(
        query="Azure cloud storage",
        top_k=5,
    )

    assert len(results) == 2

    ids = {result["chunk_id"] for result in results}

    assert ids == {
        "document_A_chunk_1",
        "document_B_chunk_1",
    }

    titles = {
        result["metadata"]["title"]
        for result in results
    }

    assert titles == {"Document A", "Document B"}
