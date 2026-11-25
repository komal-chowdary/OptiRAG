
from pathlib import Path

import pymupdf
from bs4 import BeautifulSoup


SUPPORTED_EXTENSIONS = {".html", ".htm", ".pdf", ".txt", ".md"}

# The directory containing the downloaded Microsoft documentation.
DEFAULT_DOCS_ROOT = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "microsoft_docs"
)


def build_source_metadata(
    file_path: Path,
    docs_root: Path = DEFAULT_DOCS_ROOT,
) -> dict:
    """
    Build source metadata for a document.

    Microsoft Learn URL is a candidate derived from the relative
    file path. It must be verified before being presented as a
    confirmed public citation.
    """
    file_path = file_path.resolve()
    docs_root = docs_root.resolve()

    try:
        relative_path = file_path.relative_to(docs_root)
    except ValueError:
        return {
            "source": str(file_path),
            "source_url": None,
            "source_url_status": "local_only",
            "relative_path": None,
        }

    relative_path_string = relative_path.as_posix()

    # Only map the two downloaded documentation sections.
    if relative_path.parts[0].lower() not in {"networking", "storage"}:
        return {
            "source": str(file_path),
            "source_url": None,
            "source_url_status": "local_only",
            "relative_path": relative_path_string,
        }

    if relative_path.suffix.lower() != ".md":
        return {
            "source": str(file_path),
            "source_url": None,
            "source_url_status": "local_only",
            "relative_path": relative_path_string,
        }

    # Remove the Markdown extension to form a candidate Learn path.
    url_path = relative_path.with_suffix("").as_posix()

    return {
        "source": str(file_path),
        "source_url": (
            f"https://learn.microsoft.com/en-us/azure/{url_path}"
        ),
        "source_url_status": "candidate_unverified",
        "relative_path": relative_path_string,
    }


def load_document(file_path: Path) -> dict:
    """Load a document and return its text and source metadata."""
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = file_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension}")

    if extension == ".pdf":
        with pymupdf.open(file_path) as pdf:
            text = "\n".join(page.get_text() for page in pdf)

    else:
        text = file_path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        if extension in {".html", ".htm"}:
            soup = BeautifulSoup(text, "html.parser")

            for element in soup(["script", "style", "noscript"]):
                element.decompose()

            text = soup.get_text(separator="\n")

    source_metadata = build_source_metadata(file_path)

    return {
        "title": file_path.stem,
        "text": text,
        "file_type": extension.lstrip("."),
        **source_metadata,
    }
