"""Tests for the Streamlit chat page. No database, no embedding model and no Groq calls.

The services are replaced with fakes. The page itself is run with Streamlit's AppTest, which
executes app/streamlit_app.py exactly as `streamlit run` would, without a browser.
"""
import json

import pytest
from streamlit.testing.v1 import AppTest

import app.services.answer_service as answer_service
import app.services.retrieval_service as retrieval_service
from app import streamlit_app
from app.services.answer_service import AnswerResult, Citation
from app.services.retrieval_service import RetrievalError, RetrievedChunk

APP_FILE = "app/streamlit_app.py"
FILE_LABEL_WORDS = {5: {"GLUTA"}}
ANSWER_SERVICE_GENERATE = answer_service.generate_answer   # the real function, kept before any patching
SECRET = "gsk_thisIsAFakeGroqKeyForTesting1234567890"
DB_SECRET = "postgresql://pharmaassist_app:SuperSecretPassword@localhost/pharmaassist"

CHUNKS = [RetrievedChunk(chunk_id=2, document_id=1, file_name="CZARLIV_SYRUP.pdf", document_type="pdf",
                         source_type="product_leaflet", page_number=1,
                         content="Storage :\nStore below 25 degree C at\nroom temperature",
                         product_name="CZARLIV SYRUP", similarity=0.6)]
ANSWER = AnswerResult(
    answer="The leaflet states: \"Store below 25 degree C at room temperature\" [S1].",
    citations=[Citation(chunk_id=2, document="CZARLIV_SYRUP.pdf", page=1, location="page 1")],
    abstained=False, sources_used=1)
ABSTENTION = AnswerResult(answer=answer_service.NO_EVIDENCE_MESSAGE, abstained=True, sources_used=0,
                          missing_information=["Side effects"], failure_reason="insufficient_evidence")


class FakeConnection:
    def __init__(self):
        self.closed = False

    def execute(self, sql):
        self.sql = sql
        return self

    def fetchall(self):
        return [("CZARLIV", "CZARLIV SYRUP"), ("REPOB", "REPOB")]

    def close(self):
        self.closed = True


@pytest.fixture
def backend(monkeypatch):
    """Fake retrieval and answer services that record what they receive."""
    calls = {"search": [], "answer": [], "connections": []}

    def open_connection():
        connection = FakeConnection()
        calls["connections"].append(connection)
        return connection

    def search(connection, question, top_k, min_similarity):
        calls["search"].append({"question": question, "top_k": top_k, "min_similarity": min_similarity})
        return calls.get("chunks", CHUNKS)

    def answer(question, chunks, known_products, file_label_words):
        calls["answer"].append({"question": question, "chunks": chunks, "known_products": known_products,
                                "file_label_words": file_label_words})
        return calls.get("result", ANSWER)

    monkeypatch.setattr(retrieval_service, "open_retrieval_connection", open_connection)
    monkeypatch.setattr(retrieval_service, "get_retrieval_settings", lambda: (5, None))
    monkeypatch.setattr(retrieval_service, "search_chunks", search)
    monkeypatch.setattr(answer_service, "generate_answer", answer)
    monkeypatch.setattr(answer_service, "load_file_label_words", lambda connection: FILE_LABEL_WORDS)
    return calls


@pytest.fixture(autouse=True)
def no_real_groq(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("tests must never create a real Groq client")
    monkeypatch.setattr(answer_service, "create_groq_client", refuse)


def run_page(question=None):
    page = AppTest.from_file(APP_FILE, default_timeout=30)
    page.run()
    if question is not None:
        page.chat_input[0].set_value(question).run()
    assert not page.exception
    return page


def page_text(page) -> str:
    elements = list(page.markdown) + list(page.caption) + list(page.warning) + list(page.error) + list(page.title)
    return "\n".join(str(element.value) for element in elements)


# ---------- the pipeline behind the page ----------

def test_question_reaches_retrieval_and_chunks_reach_the_answer_service(backend):
    message = streamlit_app.answer_question("  What are the storage instructions?  ")
    assert backend["search"] == [{"question": "What are the storage instructions?", "top_k": 5, "min_similarity": None}]
    assert backend["answer"][0]["chunks"] == CHUNKS
    assert backend["answer"][0]["known_products"] == ["CZARLIV", "CZARLIV SYRUP", "REPOB", "REPOB"]
    assert backend["answer"][0]["file_label_words"] == FILE_LABEL_WORDS   # loaded from the database once
    assert message["kind"] == "answer" and message["content"] == ANSWER.answer
    assert message["citations"] == [{"chunk_id": 2, "document": "CZARLIV_SYRUP.pdf", "page": 1,
                                     "location": "page 1", "source_url": None}]
    assert backend["connections"][0].closed


def test_empty_retrieval_is_an_abstention_without_a_model_call(backend, monkeypatch):
    monkeypatch.setattr(answer_service, "generate_answer", ANSWER_SERVICE_GENERATE)   # the real function
    backend["chunks"] = []
    message = streamlit_app.answer_question("What is CZARLIV?")
    assert message["kind"] == "abstention" and message["content"] == answer_service.NO_EVIDENCE_MESSAGE
    assert message["citations"] == []


def test_grounding_refusal_is_shown_as_an_abstention(backend):
    backend["result"] = AnswerResult(answer=answer_service.REJECTED_MESSAGE, abstained=True, sources_used=0,
                                     failure_reason="grounding_check_failed")
    message = streamlit_app.answer_question("Strength of Kutki?")
    assert message["kind"] == "abstention" and message["content"] == answer_service.REJECTED_MESSAGE


def test_groq_failure_is_shown_as_an_error(backend):
    backend["result"] = AnswerResult(answer=answer_service.UNAVAILABLE_MESSAGE, abstained=True, sources_used=0,
                                     failure_reason="model_unavailable")
    message = streamlit_app.answer_question("Storage?")
    assert message["kind"] == "error" and message["content"] == answer_service.UNAVAILABLE_MESSAGE


@pytest.mark.parametrize("stage, error, expected", [
    ("connect", RuntimeError(f"DATABASE_URL={DB_SECRET} failed"), streamlit_app.DATABASE_UNAVAILABLE),
    ("search", RetrievalError(f"Knowledge base search failed {DB_SECRET}"), streamlit_app.DATABASE_UNAVAILABLE),
    ("answer", answer_service.ConfigurationError(f"GROQ_API_KEY {SECRET} is not set"), streamlit_app.NOT_CONFIGURED),
    ("answer", KeyError(f"unexpected {SECRET}"), streamlit_app.UNEXPECTED_ERROR),
])
def test_errors_show_fixed_messages_without_secrets(backend, monkeypatch, stage, error, expected):
    def fail(*args, **kwargs):
        raise error
    target = {"connect": (retrieval_service, "open_retrieval_connection"),
              "search": (retrieval_service, "search_chunks"),
              "answer": (answer_service, "generate_answer")}[stage]
    monkeypatch.setattr(*target, fail)
    message = streamlit_app.answer_question("Storage?")
    assert message["kind"] == "error" and message["content"] == expected
    dumped = json.dumps(message)
    assert SECRET not in dumped and "SuperSecretPassword" not in dumped and "Traceback" not in dumped
    for connection in backend["connections"]:
        assert connection.closed


def test_overlong_question_is_refused_before_retrieval(backend):
    message = streamlit_app.answer_question("x" * (answer_service.MAX_QUESTION_CHARS + 1))
    assert message["kind"] == "error" and message["content"] == streamlit_app.QUESTION_TOO_LONG
    assert backend["search"] == [] and backend["connections"] == []


def test_citation_formatting_never_invents_fields():
    assert streamlit_app.format_citation({"chunk_id": 2, "document": "A.pdf", "page": 1, "location": "page 1",
                                          "source_url": None}) == "A.pdf · page 1 · chunk 2"
    docx = {"chunk_id": 7, "document": "B.docx", "page": None,
            "location": "paragraphs [3, 4], section 'Storage'", "source_url": "https://example.com/b"}
    assert streamlit_app.format_citation(docx) == (
        "B.docx · paragraphs [3, 4], section 'Storage' · chunk 7 (https://example.com/b)")
    assert streamlit_app.format_citation({"chunk_id": 9, "document": "C.pdf", "page": None, "location": "",
                                          "source_url": None}) == "C.pdf · chunk 9"


# ---------- the page itself (Streamlit AppTest) ----------

def test_page_shows_title_and_description(backend):
    page = run_page()
    assert page.title[0].value == "PharmaAssist"
    assert "approved product documents" in page.caption[0].value
    assert len(page.chat_input) == 1 and backend["search"] == []


def test_page_shows_answer_and_citations(backend):
    page = run_page("What are the storage instructions?")
    assert [message.name for message in page.chat_message] == ["user", "assistant"]
    text = page_text(page)
    assert "What are the storage instructions?" in text
    assert ANSWER.answer in text
    assert "Sources" in text and "CZARLIV_SYRUP.pdf · page 1 · chunk 2" in text
    assert backend["search"][0]["question"] == "What are the storage instructions?"


def test_page_shows_abstention_as_a_warning(backend):
    backend["result"] = ABSTENTION
    page = run_page("What are the side effects?")
    assert page.warning[0].value == answer_service.NO_EVIDENCE_MESSAGE
    assert "Side effects" in page_text(page)
    assert "Sources" not in page_text(page)


@pytest.mark.parametrize("empty", ["", "   \n "])
def test_empty_refusal_shows_the_safe_message_and_keeps_not_covered_items(backend, empty):
    # Observed live (side-effects question): the model refused with an empty answer.
    backend["result"] = AnswerResult(answer=empty, abstained=True, sources_used=0,
                                     missing_information=["Side effects information"],
                                     failure_reason="insufficient_evidence")
    message = streamlit_app.answer_question("What are the side effects of CZARLIV syrup?")
    assert message["kind"] == "abstention" and message["content"] == answer_service.NO_EVIDENCE_MESSAGE
    page = run_page("What are the side effects of CZARLIV syrup?")
    assert page.warning[0].value == answer_service.NO_EVIDENCE_MESSAGE
    assert "Side effects information" in page_text(page)


def test_non_empty_refusal_from_the_model_is_preserved(backend):
    refusal = "The approved leaflet does not contain information about side effects."
    backend["result"] = AnswerResult(answer=refusal, abstained=True, sources_used=0,
                                     missing_information=["Side effects"], failure_reason="insufficient_evidence")
    page = run_page("What are the side effects of CZARLIV syrup?")
    assert page.warning[0].value == refusal


def test_page_shows_errors_without_secrets(backend, monkeypatch):
    def fail():
        raise RuntimeError(f"DATABASE_URL={DB_SECRET} GROQ_API_KEY={SECRET}")
    monkeypatch.setattr(retrieval_service, "open_retrieval_connection", fail)
    page = run_page("Storage?")
    assert page.error[0].value == streamlit_app.DATABASE_UNAVAILABLE
    text = page_text(page)
    assert SECRET not in text and "SuperSecretPassword" not in text and "DATABASE_URL" not in text


def test_chat_history_is_kept_during_the_session(backend):
    page = run_page("First question?")
    page.chat_input[0].set_value("Second question?").run()
    assert not page.exception
    assert [message.name for message in page.chat_message] == ["user", "assistant", "user", "assistant"]
    assert [call["question"] for call in backend["search"]] == ["First question?", "Second question?"]


def test_end_to_end_with_the_real_answer_service_and_a_fake_groq_client(backend, monkeypatch):
    """Real generate_answer (prompt, schema, Pydantic, grounding checks); only Groq itself is faked."""
    reply = json.dumps({
        "answer": "The leaflet states: \"Store below 25 degree C at room temperature\" [S1].",
        "abstained": False, "cited_source_ids": ["S1"],
        "facts": [{"statement": "Storage", "source_id": "S1",
                   "quote": "Store below 25 degree C at\nroom temperature"}],
        "missing_information": [], "conflicts": []})

    class FakeGroq:
        def __init__(self):
            self.chat, self.completions, self.calls = self, self, 0

        def create(self, **kwargs):
            self.calls += 1
            from groq.types.chat import ChatCompletion
            return ChatCompletion.model_validate({
                "id": "c1", "object": "chat.completion", "created": 1, "model": "openai/gpt-oss-120b",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": reply}}]})

    fake = FakeGroq()
    monkeypatch.setattr(answer_service, "generate_answer", ANSWER_SERVICE_GENERATE)
    monkeypatch.setattr(answer_service, "get_groq_settings", lambda: (SECRET, "openai/gpt-oss-120b", 30.0))
    monkeypatch.setattr(answer_service, "create_groq_client", lambda api_key, timeout: fake)
    page = run_page("What are the storage instructions?")
    text = page_text(page)
    assert "Store below 25 degree C at room temperature" in text
    assert "CZARLIV_SYRUP.pdf · page 1 · chunk 2" in text
    assert fake.calls == 1 and SECRET not in text
