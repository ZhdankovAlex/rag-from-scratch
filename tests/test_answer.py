from rag_system.answer import NO_ANSWER_TEXT, Answer, Source


def test_answer_without_sources_is_unsupported_unless_it_is_the_no_answer_reply():
    cited = Answer(text="Use Uvicorn [1].", sources=[Source(1, "server-workers.md", "Workers")], retrieved=5)
    uncited = Answer(text="Use Gunicorn.", sources=[], retrieved=5)
    no_answer = Answer(text=NO_ANSWER_TEXT, sources=[], retrieved=5)
    empty_index = Answer(text="Индекс пуст", sources=[], retrieved=0)

    assert not cited.unsupported
    assert uncited.unsupported
    assert not no_answer.unsupported
    assert not empty_index.unsupported
