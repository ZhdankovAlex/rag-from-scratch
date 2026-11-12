"""
Build the demo corpus in docs/ from a checkout of the FastAPI repository.

The corpus is a fixed selection of pages from the Russian FastAPI documentation
(https://github.com/fastapi/fastapi, docs/ru/docs, MIT license). Pages are copied
into a flat docs/ directory under unique names and lightly cleaned of MkDocs-only
markup that means nothing outside the FastAPI site:

* heading anchors such as ``## Шаги { #steps }`` lose the ``{ #steps }`` part;
* code include directives such as ``{* ../../docs_src/body/tutorial001_py310.py *}``
  are dropped, because the referenced files are not part of the corpus;
* admonition fences (``/// tip | Совет`` ... ``///``) become a plain ``Совет:`` line;
  code tab fences (``//// tab | Python 3.10+`` ... ``////``) are dropped.

Usage::

    git clone --depth 1 --filter=blob:none --sparse https://github.com/fastapi/fastapi.git
    git -C fastapi sparse-checkout set docs/ru
    python scripts/prepare_corpus.py fastapi
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"

# Source path under docs/ru/docs -> file name in docs/. Names must be unique,
# because retrieval labels in eval/ refer to documents by file name.
PAGES: dict[str, str] = {
    "async.md": "async.md",
    "python-types.md": "python-types.md",
    "fastapi-cli.md": "fastapi-cli.md",
    "tutorial/first-steps.md": "first-steps.md",
    "tutorial/path-params.md": "path-params.md",
    "tutorial/query-params.md": "query-params.md",
    "tutorial/query-params-str-validations.md": "query-params-str-validations.md",
    "tutorial/path-params-numeric-validations.md": "path-params-numeric-validations.md",
    "tutorial/body.md": "body.md",
    "tutorial/body-nested-models.md": "body-nested-models.md",
    "tutorial/body-updates.md": "body-updates.md",
    "tutorial/extra-models.md": "extra-models.md",
    "tutorial/response-model.md": "response-model.md",
    "tutorial/response-status-code.md": "response-status-code.md",
    "tutorial/request-files.md": "request-files.md",
    "tutorial/request-form-models.md": "request-form-models.md",
    "tutorial/handling-errors.md": "handling-errors.md",
    "tutorial/path-operation-configuration.md": "path-operation-configuration.md",
    "tutorial/schema-extra-example.md": "schema-extra-example.md",
    "tutorial/dependencies/index.md": "dependencies.md",
    "tutorial/dependencies/classes-as-dependencies.md": "classes-as-dependencies.md",
    "tutorial/dependencies/sub-dependencies.md": "sub-dependencies.md",
    "tutorial/dependencies/dependencies-with-yield.md": "dependencies-with-yield.md",
    "tutorial/security/first-steps.md": "security-first-steps.md",
    "tutorial/security/simple-oauth2.md": "simple-oauth2.md",
    "tutorial/security/oauth2-jwt.md": "oauth2-jwt.md",
    "tutorial/middleware.md": "middleware.md",
    "tutorial/cors.md": "cors.md",
    "tutorial/sql-databases.md": "sql-databases.md",
    "tutorial/bigger-applications.md": "bigger-applications.md",
    "tutorial/background-tasks.md": "background-tasks.md",
    "tutorial/metadata.md": "metadata.md",
    "tutorial/testing.md": "testing.md",
    "advanced/settings.md": "settings.md",
    "advanced/custom-response.md": "custom-response.md",
    "advanced/websockets.md": "websockets.md",
    "advanced/events.md": "events.md",
    "advanced/behind-a-proxy.md": "behind-a-proxy.md",
    "advanced/middleware.md": "advanced-middleware.md",
    "advanced/security/http-basic-auth.md": "http-basic-auth.md",
    "deployment/concepts.md": "deployment-concepts.md",
    "deployment/docker.md": "docker.md",
    "deployment/https.md": "https.md",
    "deployment/server-workers.md": "server-workers.md",
}

HEADING_ANCHOR = re.compile(r"^(#{1,6} .*?)\s*\{ #[^}]*\}\s*$")
INCLUDE_DIRECTIVE = re.compile(r"^\s*\{\*.*\*\}\s*$")
ADMONITION_OPEN = re.compile(r"^\s*/{3,}\s*(?P<kind>\w+)\s*(?:\|\s*(?P<title>.+?))?\s*$")
ADMONITION_CLOSE = re.compile(r"^\s*/{3,}\s*$")


def clean_markdown(text: str) -> str:
    """
    Strip FastAPI-site-specific MkDocs markup from a page.

    Args:
        text: Raw markdown of a FastAPI documentation page.

    Returns:
        The page with heading anchors, code include directives and admonition
        fences removed (an admonition's title, but not a code tab's, is kept as a ``Title:`` line).
    """
    lines: list[str] = []
    in_code = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
            lines.append(line)
            continue
        if in_code:
            lines.append(line)
            continue
        if INCLUDE_DIRECTIVE.match(line) or ADMONITION_CLOSE.match(line):
            continue
        if match := ADMONITION_OPEN.match(line):
            if match.group("title") and match.group("kind") != "tab":
                lines.append(f"{match.group('title')}:")
            continue
        if match := HEADING_ANCHOR.match(line):
            line = match.group(1)
        lines.append(line)
    # Removed lines can leave runs of blank lines behind; collapse them.
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"


def prepare_corpus(fastapi_repo: Path, docs_dir: Path = DOCS_DIR) -> int:
    """
    Copy and clean the selected FastAPI pages into the corpus directory.

    Existing ``*.md`` files in ``docs_dir`` are removed first, so the corpus
    always matches ``PAGES`` exactly. The FastAPI license is copied next to the
    pages for attribution.

    Args:
        fastapi_repo: Root of a FastAPI checkout that contains ``docs/ru/docs``.
        docs_dir: Output directory for the corpus.

    Returns:
        The number of pages written.

    Raises:
        FileNotFoundError: If the checkout lacks a selected page or the LICENSE.
    """
    source_dir = fastapi_repo / "docs" / "ru" / "docs"
    missing = [page for page in PAGES if not (source_dir / page).is_file()]
    if missing:
        raise FileNotFoundError(f"Pages not found under {source_dir}: {', '.join(missing)}")

    docs_dir.mkdir(parents=True, exist_ok=True)
    for old in docs_dir.glob("*.md"):
        old.unlink()
    for source, target in PAGES.items():
        text = (source_dir / source).read_text(encoding="utf-8")
        (docs_dir / target).write_text(clean_markdown(text), encoding="utf-8", newline="\n")
    shutil.copyfile(fastapi_repo / "LICENSE", docs_dir / "LICENSE")
    return len(PAGES)


def main() -> int:
    """Parse the command line and build the corpus."""
    parser = argparse.ArgumentParser(description="Build docs/ from a FastAPI checkout.")
    parser.add_argument("fastapi_repo", type=Path, help="path to a clone of github.com/fastapi/fastapi")
    args = parser.parse_args()
    try:
        count = prepare_corpus(args.fastapi_repo)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    print(f"Wrote {count} pages to {DOCS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
