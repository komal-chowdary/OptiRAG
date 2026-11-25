
import re
from typing import Any

import yaml


def extract_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """
    Extract YAML metadata from the beginning of a Markdown document.

    Returns:
        metadata: Parsed YAML fields.
        content: Markdown content without the YAML block.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if not text.startswith("---\n"):
        return {}, text

    match = re.match(r"\A---\s*\n(.*?)\n---\s*(?:\n|$)", text, re.DOTALL)

    if not match:
        return {}, text

    yaml_content = match.group(1)
    document_content = text[match.end():]

    try:
        metadata = yaml.safe_load(yaml_content) or {}
    except yaml.YAMLError:
        # Keep the original content if the YAML block is malformed.
        return {}, text

    if not isinstance(metadata, dict):
        metadata = {}

    return metadata, document_content


def clean_text(text: str) -> str:
    """
    Normalize whitespace while preserving Markdown structure,
    headings, tables, links, lists, and code examples.
    """
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove HTML comments.
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # Remove trailing whitespace from each line.
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # Normalize spaces and tabs without flattening indentation
    # inside code blocks.
    lines = []
    in_code_block = False

    for line in text.split("\n"):
        if line.strip().startswith("```") or line.strip().startswith("~~~"):
            in_code_block = not in_code_block
            lines.append(line.strip())
            continue

        if not in_code_block:
            line = re.sub(r"[ \t]+", " ", line)
            line = line.strip()

        lines.append(line)

    text = "\n".join(lines)

    # Collapse excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
