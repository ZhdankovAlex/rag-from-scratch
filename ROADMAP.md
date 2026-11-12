# Roadmap

The pipeline is complete: indexing, search, answers from a local LLM (Ollama)
with links to sources, the CLI, search evaluation on a held-out set, offline tests
and a linter in CI. The metrics are in the README, section "Search quality".

---

## 1. Evaluate answers, not only search
`eval` currently measures search only.
- [ ] A question set with reference facts (or links to the right section).
- [ ] Metrics: share of answers with citations; share of statements backed
  by citations; a correct refusal on the "negative" questions (6 are already in test).
- [ ] Automatic checking with an LLM judge, plus spot checks of the judge by hand.
- [ ] `python main.py eval --answers`. On a CPU an answer takes about a minute, so a run
  over the whole set is a separate long command, not part of CI.

## 2. Search
- [ ] Hybrid search (BM25 + vectors, merged with RRF) and a cross-encoder reranker
  were already tried on a different, private document set and gave no improvement,
  so they are not in the code. These docs have many exact identifiers (`Depends`,
  `UploadFile`, `root_path`) where keyword search may help: re-measure on test.
- [ ] Go through the dev/test misses (`python main.py eval -v`): most of them are
  neighbouring pages on one topic (`security-first-steps.md` vs
  `simple-oauth2.md`), not model errors.

## 3. The documents
- [ ] There are 44 of about 120 Russian FastAPI pages now. Add the rest
  to `scripts/prepare_corpus.py` and measure how the metrics change as the
  set grows.
- [ ] FastAPI code blocks are included with `{* … *}` directives from `docs_src/` and are
  dropped when the set is prepared. Put the code itself in their place, so that
  "how do I write…" questions find the example too.

## 4. Engineering housekeeping
- [ ] `ruff format` in a separate commit: a large mechanical diff.
- [ ] `mypy` in CI.
- [ ] `pyproject.toml` instead of `requirements*.txt`, and a `rag` entry point instead of
  `python main.py`.

## 5. Usability
- [ ] Stream the answer in the CLI as well (the web UI already does): on a CPU an
  answer takes about a minute.
- [ ] Conversation history in `chat` for follow-up questions.
