from dataclasses import dataclass
from pathlib import Path

# Resolved from this file, so loading works regardless of the current
# working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"


@dataclass
class Document:
    path: Path
    text: str

    @property
    def filename(self) -> str:
        return self.path.name


def load_documents(docs_dir: str | Path = DOCS_DIR) -> list[Document]:
    """
    Recursively find all .md files in the specified directory and load their contents.

    Args:
        docs_dir: Path to the directory containing markdown files.
            Defaults to the project's docs/ directory.

    Returns:
        A list of Document objects containing the file path and text content.
    """
    docs_path = Path(docs_dir)
    documents: list[Document] = []

    for md_file in docs_path.rglob("*.md"):
        try:
            text = md_file.read_text(encoding="utf-8")
            documents.append(Document(path=md_file, text=text))
        except Exception as e:
            print(f"Error reading {md_file}: {e}")

    return documents


if __name__ == "__main__":
    # Example usage
    docs = load_documents()
    print(f"Loaded {len(docs)} documents:")
    for doc in docs:
        print(f"  - {doc.path} ({len(doc.text)} characters)")
