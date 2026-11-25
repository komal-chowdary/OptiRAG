
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.generation.answer_generator import AnswerGenerator


def main():
    generator = AnswerGenerator()

    sample_chunks = [
        {
            "text": (
                "Azure virtual networks enable Azure resources "
                "to communicate securely with each other, the "
                "internet, and on-premises networks."
            ),
            "metadata": {
                "title": "Azure Virtual Networks",
                "source": "sample_test",
            },
        }
    ]

    result = generator.generate(
        question="What do Azure virtual networks enable?",
        retrieved_chunks=sample_chunks,
    )

    print("\nGenerated answer:\n")
    print(result["answer"])
    print("\nCitations:")
    print(result["citations"])
    print("\nModel:", result["model"])


if __name__ == "__main__":
    main()
