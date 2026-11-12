import json
from dataclasses import dataclass
from pathlib import Path

from .loader import PROJECT_ROOT
from .vector_store import VectorStore

QUESTIONS_FILE = PROJECT_ROOT / "eval" / "retrieval_questions.json"

# How deep to look for the first relevant chunk; a file found below this
# depth counts as a miss (reciprocal rank 0).
SEARCH_DEPTH = 20


@dataclass
class EvalQuestion:
    """
    A question with the docs files that answer it.

    Attributes:
        question: Question text, phrased the way a user would ask.
        expected: File names where the answer is described; retrieving a chunk
            from any of them counts as a hit. Empty for a negative question
            (the docs do not answer it), kept for evaluating refusals.
        split: "dev" for questions already used to make retrieval decisions,
            "test" for held-out ones that give an unbiased estimate.
    """

    question: str
    expected: list[str]
    split: str = "dev"

    @property
    def negative(self) -> bool:
        """True when the docs do not answer the question."""
        return not self.expected


@dataclass
class QuestionResult:
    """
    Retrieval outcome for one question.

    Attributes:
        question: The evaluated question.
        rank: 1-based rank of the first chunk from an expected file,
            or None if none was found within SEARCH_DEPTH.
        top_source: File of the top-ranked chunk.
    """

    question: EvalQuestion
    rank: int | None
    top_source: str


@dataclass
class EvalReport:
    """
    Aggregated retrieval metrics over a question set.

    Attributes:
        results: Per-question outcomes, in input order.
        hit_at: Share of questions with a relevant file in the top k, keyed by k.
        mrr: Mean reciprocal rank of the first relevant file.
    """

    results: list[QuestionResult]
    hit_at: dict[int, float]
    mrr: float


def load_questions(path: Path = QUESTIONS_FILE, split: str | None = None) -> list[EvalQuestion]:
    """
    Load the evaluation question set.

    Args:
        path: JSON file with a list of {"question": ..., "expected": [...], "split": ...}.
        split: Keep only "dev" or "test" questions; all when None.

    Returns:
        Questions in file order.
    """
    items = json.loads(path.read_text(encoding="utf-8"))
    questions = [
        EvalQuestion(question=item["question"], expected=item["expected"], split=item.get("split", "dev"))
        for item in items
    ]
    return [q for q in questions if split is None or q.split == split]


def evaluate_retrieval(
    store: VectorStore,
    questions: list[EvalQuestion],
    ks: tuple[int, ...] = (1, 3, 5),
) -> EvalReport:
    """
    Measure how well retrieval finds the docs file that answers each question.

    Relevance is judged at file level: a question is answered at rank r if
    the r-th retrieved chunk is the first one from any expected file.
    Negative questions have no relevant file and are left out.

    Args:
        store: Indexed vector store to search.
        questions: Questions with expected files.
        ks: Cut-offs for hit@k.

    Returns:
        Per-question ranks with hit@k and MRR.

    Raises:
        ValueError: No positive questions to evaluate.
    """
    positive = [question for question in questions if not question.negative]
    if not positive:
        raise ValueError("no questions with expected files to evaluate")

    results: list[QuestionResult] = []
    for question in positive:
        sources = [r["metadata"]["source"] for r in store.search(question.question, top_k=SEARCH_DEPTH)]
        rank = next(
            (position for position, source in enumerate(sources, 1) if source in question.expected),
            None,
        )
        results.append(QuestionResult(question=question, rank=rank, top_source=sources[0] if sources else ""))

    total = len(results)
    hit_at = {k: sum(1 for r in results if r.rank is not None and r.rank <= k) / total for k in ks}
    mrr = sum(1 / r.rank for r in results if r.rank is not None) / total
    return EvalReport(results=results, hit_at=hit_at, mrr=mrr)


if __name__ == "__main__":
    # Example usage: needs an indexed vector store (python main.py index)
    report = evaluate_retrieval(VectorStore(), load_questions())
    print(f"hit@k: {report.hit_at}, MRR: {report.mrr:.3f}")
