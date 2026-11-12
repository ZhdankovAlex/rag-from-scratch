import json

import pytest

from rag_system.evaluate import QUESTIONS_FILE, SEARCH_DEPTH, EvalQuestion, evaluate_retrieval, load_questions


class RankedStore:
    """Returns a fixed ranking of source files per query."""

    def __init__(self, rankings: dict[str, list[str]]):
        self.rankings = rankings

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        return [{"metadata": {"source": source}} for source in self.rankings[query][:top_k]]


def test_ranks_hit_at_k_and_mrr():
    store = RankedStore({
        "q1": ["a.md", "b.md"],             # rank 1
        "q2": ["x.md", "y.md", "b.md"],     # rank 3
        "q3": ["x.md"] * SEARCH_DEPTH,      # not found
        "q4": ["x.md", "c.md", "a.md"],     # rank 2: either expected file counts
    })
    questions = [
        EvalQuestion("q1", ["a.md"]),
        EvalQuestion("q2", ["b.md"]),
        EvalQuestion("q3", ["a.md"]),
        EvalQuestion("q4", ["a.md", "c.md"]),
    ]

    report = evaluate_retrieval(store, questions, ks=(1, 3))

    assert [r.rank for r in report.results] == [1, 3, None, 2]
    assert [r.top_source for r in report.results] == ["a.md", "x.md", "x.md", "x.md"]
    assert report.hit_at == {1: 0.25, 3: 0.75}
    assert report.mrr == pytest.approx((1 + 1 / 3 + 0 + 1 / 2) / 4)


def test_negative_questions_are_not_scored():
    store = RankedStore({"q1": ["a.md"], "oracle": ["x.md"]})
    questions = [EvalQuestion("q1", ["a.md"]), EvalQuestion("oracle", [])]

    report = evaluate_retrieval(store, questions, ks=(1,))

    assert [r.question.question for r in report.results] == ["q1"]
    assert report.hit_at == {1: 1.0}
    assert report.mrr == 1.0


def test_only_negative_questions_is_an_error():
    with pytest.raises(ValueError):
        evaluate_retrieval(RankedStore({}), [EvalQuestion("oracle", [])])


def test_load_questions_filters_by_split(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps([
        {"split": "dev", "question": "d", "expected": ["a.md"]},
        {"split": "test", "question": "t", "expected": ["b.md"]},
        {"question": "no split field", "expected": ["c.md"]},
    ]), encoding="utf-8")

    assert [q.question for q in load_questions(path, split="test")] == ["t"]
    assert [q.question for q in load_questions(path, split="dev")] == ["d", "no split field"]
    assert len(load_questions(path)) == 3


def test_shipped_question_set_is_consistent():
    questions = load_questions(QUESTIONS_FILE)
    docs = {p.name for p in (QUESTIONS_FILE.parent.parent / "docs").glob("*.md")}

    assert {q.split for q in questions} == {"dev", "test"}
    assert len({q.question for q in questions}) == len(questions)  # no duplicates
    assert all(set(q.expected) <= docs for q in questions)  # expected files exist
    assert any(q.negative for q in questions if q.split == "test")
