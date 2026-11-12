import re
from collections.abc import Iterator
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .loader import Document

# Fenced code blocks hold shell comments ("# Check the config") and other
# lines that look like Markdown headings but are not.
CODE_FENCES = ("```", "~~~")

# MkDocs pages often set the page title with raw HTML instead of "# ".
HTML_H1 = re.compile(r"^\s*<h1[^>]*>(.*?)</h1>\s*$", re.IGNORECASE)

# ID key of the chunk holding the text before the first "##" section.
INTRO_KEY = "intro"


@dataclass
class Chunk:
    id: str
    source: str
    title: str
    text: str

    @property
    def embedding_text(self) -> str:
        """
        Text to embed: the "Document > Section" title followed by the chunk text.

        Parts of a split section do not repeat its "##" header, and no chunk
        contains the document title, so without this a chunk loses the topic
        it belongs to (e.g. a table under "Доступные ответы" in custom-response.md).
        The stored and cited text stays chunk.text.
        """
        return f"{self.title}\n\n{self.text}"


def iter_prose_lines(lines: list[str]) -> Iterator[tuple[int, str]]:
    """
    Yield lines that are outside fenced code blocks.

    Fence delimiter lines themselves are skipped too.

    Args:
        lines: Document lines.

    Yields:
        (line index, line) pairs for lines outside code blocks.
    """
    in_code = False
    for index, line in enumerate(lines):
        if line.lstrip().startswith(CODE_FENCES):
            in_code = not in_code
            continue
        if not in_code:
            yield index, line


def find_document_title(lines: list[str], fallback: str) -> str:
    """
    Find the document title: the first non-empty "# " or <h1> heading.

    Headings inside code blocks are ignored, and so are empty <h1> tags
    (MkDocs pages use them to hide the rendered title).

    Args:
        lines: Document lines.
        fallback: Title to use when the document has no level-1 heading.

    Returns:
        The document title.
    """
    for _, line in iter_prose_lines(lines):
        if line.startswith("# "):
            title = line[2:].strip()
        elif match := HTML_H1.match(line):
            title = re.sub(r"<[^>]+>", "", match.group(1)).strip()
        else:
            continue
        if title:
            return title
    return fallback


def extract_intro(lines: list[str]) -> str:
    """
    Return the document's intro: text before the first "##" section.

    YAML front matter is dropped. The intro counts as empty when it holds
    nothing but the title heading, blank lines and horizontal rules.

    Args:
        lines: Document lines before the first "##" header.

    Returns:
        The intro text, or "" when there is no prose before the first section.
    """
    if lines and lines[0].strip() == "---":
        closing = next((i for i, line in enumerate(lines[1:], 1) if line.strip() == "---"), None)
        if closing is not None:
            lines = lines[closing + 1:]

    def is_content(line: str) -> bool:
        stripped = line.strip()
        return bool(stripped) and stripped != "---" and not stripped.startswith("# ") and not HTML_H1.match(stripped)

    if not any(is_content(line) for line in lines):
        return ""
    return "\n".join(lines).strip()


def chunk_documents(documents: list[Document]) -> list[Chunk]:
    """
    Split documents into chunks by ## headers, then further split large sections.

    Strategy:
    1. Split document by ## headers into logical sections
       (lines inside fenced code blocks are never treated as headers)
    2. Each section includes its ## header
    3. Preserve the document title ("# " or <h1>) as context
    4. Keep the intro before the first ## as its own chunk titled after the document
    5. If section > 1000 chars, split further with RecursiveCharacterTextSplitter
    6. Use chunk_overlap=200 for additional splitting
    7. Generate stable IDs: source + section_title (or "intro") + part_number

    Args:
        documents: List of Document objects to chunk.

    Returns:
        List of Chunk objects.
    """
    chunks: list[Chunk] = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    for doc in documents:
        lines = doc.text.split('\n')

        main_title = find_document_title(lines, fallback=doc.filename)

        # Positions of ## headers, ignoring look-alikes inside code blocks
        header_positions = [i for i, line in iter_prose_lines(lines) if line.startswith('## ')]

        # Sections as (ID key, chunk title, text)
        sections: list[tuple[str, str, str]] = []
        if not header_positions:
            # No ## headers, treat whole document as one section
            sections.append((doc.filename, f"{main_title} > {doc.filename}", doc.text))
        else:
            intro = extract_intro(lines[:header_positions[0]])
            if intro:
                sections.append((INTRO_KEY, main_title, intro))

            for i, pos in enumerate(header_positions):
                section_title = lines[pos][3:].strip()
                end_pos = header_positions[i + 1] if i + 1 < len(header_positions) else len(lines)
                section_text = '\n'.join(lines[pos:end_pos])
                sections.append((section_title, f"{main_title} > {section_title}", section_text))

        for id_key, title, section_text in sections:
            # Small sections stay whole; large ones are split further
            parts = [section_text] if len(section_text) <= 1000 else text_splitter.split_text(section_text)
            for part_idx, part_text in enumerate(parts):
                chunks.append(Chunk(
                    id=f"{doc.filename}::{id_key}::{part_idx}",
                    source=doc.filename,
                    title=title,
                    text=part_text,
                ))

    return chunks


if __name__ == "__main__":
    # Example usage
    from .loader import load_documents

    documents = load_documents()
    chunks = chunk_documents(documents)

    print(f"Created {len(chunks)} chunks from {len(documents)} documents:")
    for chunk in chunks:
        print(f"  [{chunk.id}] {chunk.title} ({len(chunk.text)} chars)")
