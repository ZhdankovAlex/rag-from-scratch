"""
Answer types returned by the LLM step: the answer text and the sources it cites.
"""
from dataclasses import dataclass

# Exact reply when the docs do not answer the question: an answer without
# sources is expected only in this case.
NO_ANSWER_TEXT = "В найденной документации нет ответа на этот вопрос."


@dataclass
class Source:
    """
    A retrieved chunk that the answer cites.

    Attributes:
        number: Citation marker number used in the answer text, e.g. 1 for "[1]".
        file: Markdown file the chunk came from.
        title: Chunk breadcrumb title ("Document > Section").
        chunk_id: ID of the cited chunk in the vector store ("" when unknown).
    """

    number: int
    file: str
    title: str
    chunk_id: str = ""


@dataclass
class Answer:
    """
    The model's answer to a question with the sources it relied on.

    Attributes:
        text: Answer text with "[n]" citation markers.
        sources: Cited chunks, ordered by marker number.
        retrieved: Number of chunks that were passed to the model.
    """

    text: str
    sources: list[Source]
    retrieved: int

    @property
    def unsupported(self) -> bool:
        """
        True when the model answered without citing any retrieved chunk.

        Such an answer may come from the model's general knowledge rather than
        the docs. The "not in the docs" reply is expected to have no sources
        and does not count.
        """
        return self.retrieved > 0 and not self.sources and NO_ANSWER_TEXT.rstrip(".") not in self.text
