from pathlib import Path

from rag_system.chunker import chunk_documents
from rag_system.loader import Document


def make_doc(text: str, name: str = "guide.md") -> Document:
    return Document(path=Path(name), text=text)


def test_splits_by_second_level_headers_with_breadcrumb_titles():
    doc = make_doc("# Guide\n\nIntro.\n\n## Install\nRun it.\n\n## Configure\nEdit it.\n")

    chunks = chunk_documents([doc])

    assert [c.title for c in chunks] == ["Guide", "Guide > Install", "Guide > Configure"]
    assert chunks[1].text.startswith("## Install")
    assert all(c.source == "guide.md" for c in chunks)


def test_embedding_text_prefixes_breadcrumb_title():
    doc = make_doc("# Alerting\n## Email\nSMTP settings.\n")

    chunk = chunk_documents([doc])[0]

    assert chunk.embedding_text == "Alerting > Email\n\n## Email\nSMTP settings.\n"


def test_ids_are_stable_and_unique():
    doc = make_doc("# Guide\n## A\nx\n## B\ny\n")

    first = [c.id for c in chunk_documents([doc])]
    second = [c.id for c in chunk_documents([doc])]

    assert first == second == ["guide.md::A::0", "guide.md::B::0"]


def test_long_section_is_split_into_numbered_parts():
    paragraph = "Слово " * 60  # ~360 chars
    doc = make_doc("# Guide\n## Big\n" + "\n\n".join([paragraph] * 6))

    chunks = chunk_documents([doc])

    assert len(chunks) > 1
    assert all(len(c.text) <= 1000 for c in chunks)
    assert [c.id for c in chunks] == [f"guide.md::Big::{i}" for i in range(len(chunks))]
    assert {c.title for c in chunks} == {"Guide > Big"}


def test_document_without_sections_is_one_chunk_named_after_file():
    doc = make_doc("# Notes\nJust text.", name="notes.md")

    chunks = chunk_documents([doc])

    assert len(chunks) == 1
    assert chunks[0].id == "notes.md::notes.md::0"
    assert chunks[0].text == doc.text


def test_headings_inside_code_blocks_are_ignored():
    doc = make_doc(
        "Intro.\n\n"
        "```bash\n# Check the version\nfastapi --version\n## not a section\n```\n\n"
        "# App config\n\n"
        "## Sections\nText.\n\n"
        "~~~\n## also not a section\n~~~\n"
    )

    chunks = chunk_documents([doc])

    assert [c.title for c in chunks] == ["App config", "App config > Sections"]
    assert "## also not a section" in chunks[1].text


def test_html_h1_is_used_as_document_title_and_empty_h1_is_skipped():
    mkdocs_page = make_doc("<h1>Установка</h1>\n\n## Требования\nPython 3.10+.\n")
    hidden_title = make_doc('<h1 class="display-none"></h1>\n\n# FastAPI\n\n## Обзор\nТекст.\n', name="index.md")

    assert chunk_documents([mkdocs_page])[0].title == "Установка > Требования"
    assert chunk_documents([hidden_title])[0].title == "FastAPI > Обзор"


def test_intro_before_first_section_is_kept_without_front_matter():
    doc = make_doc(
        "---\nhide:\n  - navigation\n---\n\n"
        "<h1>Быстрый старт</h1>\n\nМинимальная установка: Python и один пакет.\n\n"
        "## Установка\nШаги.\n"
    )

    intro, section = chunk_documents([doc])

    assert intro.id == "guide.md::intro::0"
    assert intro.title == "Быстрый старт"
    assert "Минимальная установка" in intro.text
    assert "hide:" not in intro.text
    assert section.title == "Быстрый старт > Установка"


def test_title_only_intro_does_not_produce_a_chunk():
    doc = make_doc('<h1 class="display-none"></h1>\n\n# Guide\n\n---\n\n## A\nText.\n')

    assert [c.id for c in chunk_documents([doc])] == ["guide.md::A::0"]
