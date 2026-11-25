
import csv
import json
import logging
import math
from pathlib import Path

from app.services.rag_service import RAGService


BACKEND_DIR = Path(__file__).resolve().parents[1]

GOLDEN_PATH = (
    BACKEND_DIR / "data" / "evaluation" / "golden_dataset.jsonl"
)
MAPPING_PATH = (
    BACKEND_DIR / "data" / "evaluation" / "document_id_mapping_report.json"
)

OUTPUT_DIR = BACKEND_DIR / "data" / "evaluation"

TOP_K = 5

STRATEGIES = [
    "basic_vector",
    "semantic_vector",
    "hybrid",
    "hybrid_reranked",
]


def load_golden_dataset():
    records = []

    with GOLDEN_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    # Load old-to-new document ID mapping.
    with MAPPING_PATH.open("r", encoding="utf-8") as file:
        mapping_report = json.load(file)

    id_mappings = mapping_report.get("mappings", {})

    if mapping_report.get("unmatched_count") != 0:
        raise ValueError(
            "Document ID mapping contains unmatched IDs. "
            "Resolve them before evaluation."
        )

    answerable = [
        record for record in records
        if record.get("answerable") is True
    ]

    mapped_count = 0

    for record in answerable:
        relevant_ids = record.get("relevant_document_ids", [])

        if not relevant_ids:
            raise ValueError(
                f"{record['id']} has no relevant document IDs."
            )

        new_relevant_ids = []

        for old_id in relevant_ids:
            mapping = id_mappings.get(old_id)

            if not mapping or not mapping.get("new_document_id"):
                raise ValueError(
                    f"No valid mapping found for document ID "
                    f"{old_id} in question {record['id']}."
                )

            new_relevant_ids.append(mapping["new_document_id"])
            mapped_count += 1

        # Update only the in-memory record.
        record["relevant_document_ids"] = new_relevant_ids

    print(
        f"Document ID mapping loaded successfully. "
        f"Mapped {mapped_count} relevant ID references "
        f"across {len(answerable)} answerable questions."
    )

    return answerable

def get_retrieved_document_ids(results):
    """
    Extract unique document IDs in retrieval ranking order.
    Multiple chunks from the same document count only once.
    """
    document_ids = []
    seen = set()

    for item in results:
        metadata = item.get("metadata") or {}

        document_id = (
            metadata.get("document_id")
            or item.get("document_id")
        )

        if document_id and document_id not in seen:
            document_ids.append(str(document_id))
            seen.add(str(document_id))

    return document_ids


def calculate_metrics(retrieved_ids, relevant_ids, k=5):
    relevant = set(str(item) for item in relevant_ids)
    retrieved = retrieved_ids[:k]

    relevance = [
        1 if document_id in relevant else 0
        for document_id in retrieved
    ]

    # Hit Rate@K
    hit_rate = 1.0 if any(relevance) else 0.0

    # Recall@K
    recall = sum(relevance) / len(relevant)

    # Mean Reciprocal Rank contribution
    reciprocal_rank = 0.0

    for rank, is_relevant in enumerate(relevance, start=1):
        if is_relevant:
            reciprocal_rank = 1.0 / rank
            break

    # DCG@K with binary relevance
    dcg = sum(
        rel / math.log2(rank + 1)
        for rank, rel in enumerate(relevance, start=1)
    )

    # Ideal DCG@K
    ideal_relevant_count = min(len(relevant), k)

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant_count + 1)
    )

    ndcg = dcg / idcg if idcg else 0.0

    return {
        "hit_rate": hit_rate,
        "recall": recall,
        "mrr": reciprocal_rank,
        "ndcg": ndcg,
    }


def main():
    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(
            f"Golden dataset not found: {GOLDEN_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    questions = load_golden_dataset()

    print(f"Answerable questions loaded: {len(questions)}")
    print(f"Evaluation cutoff: top {TOP_K}")

    rag_service = RAGService()

    all_details = []
    summary = {}
    errors = []

    for strategy in STRATEGIES:
        print("\n" + "=" * 70)
        print(f"Evaluating: {strategy}")
        print("=" * 70)

        metric_values = {
            "hit_rate": [],
            "recall": [],
            "mrr": [],
            "ndcg": [],
        }

        successful_questions = 0

        for index, question_record in enumerate(questions, start=1):
            question_id = question_record["id"]
            question = question_record["question"]
            relevant_ids = question_record["relevant_document_ids"]

            try:
                service = rag_service.retrieval_services[strategy]

                response = service.search(
                    question=question,
                    top_k=TOP_K,
                )

                results = response.get("results", [])

                retrieved_ids = get_retrieved_document_ids(results)

                metrics = calculate_metrics(
                    retrieved_ids,
                    relevant_ids,
                    TOP_K,
                )

                successful_questions += 1

                for metric_name, value in metrics.items():
                    metric_values[metric_name].append(value)

                all_details.append({
                    "question_id": question_id,
                    "strategy": strategy,
                    "question": question,
                    "relevant_document_ids": relevant_ids,
                    "retrieved_document_ids": retrieved_ids,
                    "result_count": len(results),
                    **metrics,
                    "error": None,
                })

                print(
                    f"[{index}/{len(questions)}] "
                    f"{question_id}: "
                    f"Hit={metrics['hit_rate']:.2f}, "
                    f"Recall={metrics['recall']:.2f}, "
                    f"MRR={metrics['mrr']:.2f}, "
                    f"nDCG={metrics['ndcg']:.2f}"
                )

            except Exception as exc:
                error_message = (
                    f"{type(exc).__name__}: {exc}"
                )

                errors.append({
                    "question_id": question_id,
                    "strategy": strategy,
                    "error": error_message,
                })

                all_details.append({
                    "question_id": question_id,
                    "strategy": strategy,
                    "question": question,
                    "relevant_document_ids": relevant_ids,
                    "retrieved_document_ids": [],
                    "result_count": 0,
                    "hit_rate": None,
                    "recall": None,
                    "mrr": None,
                    "ndcg": None,
                    "error": error_message,
                })

                print(
                    f"[{index}/{len(questions)}] "
                    f"{question_id}: ERROR - {error_message}"
                )

        summary[strategy] = {
            "questions_expected": len(questions),
            "questions_evaluated": successful_questions,
            "complete": successful_questions == len(questions),
            "hit_rate_at_5": (
                sum(metric_values["hit_rate"])
                / len(metric_values["hit_rate"])
                if metric_values["hit_rate"] else None
            ),
            "recall_at_5": (
                sum(metric_values["recall"])
                / len(metric_values["recall"])
                if metric_values["recall"] else None
            ),
            "mrr_at_5": (
                sum(metric_values["mrr"])
                / len(metric_values["mrr"])
                if metric_values["mrr"] else None
            ),
            "ndcg_at_5": (
                sum(metric_values["ndcg"])
                / len(metric_values["ndcg"])
                if metric_values["ndcg"] else None
            ),
        }

    # Save aggregate results.
    summary_path = OUTPUT_DIR / "retrieval_evaluation_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump({
            "evaluation_cutoff": TOP_K,
            "answerable_questions": len(questions),
            "ranking_unit": "unique_document_id",
            "strategies": summary,
            "errors": errors,
            "overall_complete": (
                not errors
                and all(item["complete"] for item in summary.values())
            ),
        }, file, indent=2, ensure_ascii=False)

    # Save per-question results.
    details_path = OUTPUT_DIR / "retrieval_evaluation_details.jsonl"

    with details_path.open("w", encoding="utf-8") as file:
        for detail in all_details:
            file.write(
                json.dumps(detail, ensure_ascii=False) + "\n"
            )

    # Save a spreadsheet-friendly CSV.
    csv_path = OUTPUT_DIR / "retrieval_evaluation_details.csv"

    columns = [
        "question_id",
        "strategy",
        "hit_rate",
        "recall",
        "mrr",
        "ndcg",
        "result_count",
        "error",
    ]

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for detail in all_details:
            writer.writerow({
                column: detail.get(column)
                for column in columns
            })

    # Print aggregate comparison.
    print("\n" + "=" * 85)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 85)

    print(
        f"{'Strategy':<22}"
        f"{'Hit@5':>10}"
        f"{'Recall@5':>12}"
        f"{'MRR@5':>10}"
        f"{'nDCG@5':>10}"
        f"{'Complete':>12}"
    )

    for strategy, metrics in summary.items():
        def display(value):
            return f"{value:.4f}" if value is not None else "N/A"

        print(
            f"{strategy:<22}"
            f"{display(metrics['hit_rate_at_5']):>10}"
            f"{display(metrics['recall_at_5']):>12}"
            f"{display(metrics['mrr_at_5']):>10}"
            f"{display(metrics['ndcg_at_5']):>10}"
            f"{str(metrics['complete']):>12}"
        )

    print("\nEvaluation files created:")
    print(summary_path)
    print(details_path)
    print(csv_path)

    print(f"\nTotal retrieval errors: {len(errors)}")

    if errors:
        print(
            "WARNING: Evaluation is incomplete. "
            "Resolve errors before comparing strategies."
        )
    else:
        print("All four strategy evaluations completed successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    main()
