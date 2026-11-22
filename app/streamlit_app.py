"""PharmaAssist chat interface (Streamlit).

Run from the project folder:  streamlit run app/streamlit_app.py

This page only displays results. The existing services do the work:
open_retrieval_connection() -> search_chunks() -> generate_answer() (Groq + grounding checks).
Users never see stack traces, environment variables or keys: errors are shown as fixed messages.
"""
import logging
import sys
from pathlib import Path

# Streamlit runs this file as a script, so the project folder must be importable.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import psycopg  # noqa: E402
import streamlit as st  # noqa: E402

from app.services import answer_service, retrieval_service  # noqa: E402

logger = logging.getLogger(__name__)

DATABASE_UNAVAILABLE = "The approved knowledge base is temporarily unavailable. Please try again later."
NOT_CONFIGURED = "The answer service is not configured yet. Please contact the administrator."
UNEXPECTED_ERROR = "Something went wrong while answering your question. Please try again later."
QUESTION_TOO_LONG = f"Please shorten your question to at most {answer_service.MAX_QUESTION_CHARS} characters."


def load_known_products(connection) -> list:
    """Product names and ids, so questions about products without approved documents are refused."""
    rows = connection.execute("SELECT product_id, product_name FROM products").fetchall()
    return [value for row in rows for value in row]


def error_message(text: str) -> dict:
    return {"role": "assistant", "kind": "error", "content": text,
            "citations": [], "missing_information": [], "conflicts": []}


def result_message(result: answer_service.AnswerResult) -> dict:
    """Turn an AnswerResult into a chat message. Only data from the result is used."""
    if result.failure_reason == "model_unavailable":
        kind = "error"
    elif result.abstained:
        kind = "abstention"
    else:
        kind = "answer"
    content = result.answer
    if result.abstained and not content.strip():
        content = answer_service.NO_EVIDENCE_MESSAGE   # the model may refuse with an empty answer
    return {"role": "assistant", "kind": kind, "content": content,
            "citations": [citation.model_dump() for citation in result.citations],
            "missing_information": result.missing_information, "conflicts": result.conflicts}


def answer_question(question: str) -> dict:
    """Answer one question with the existing services and return the assistant's chat message."""
    question = question.strip()
    if len(question) > answer_service.MAX_QUESTION_CHARS:
        return error_message(QUESTION_TOO_LONG)

    try:
        connection = retrieval_service.open_retrieval_connection()
    except Exception as error:   # missing DATABASE_URL, server down, wrong role
        logger.warning("Database connection failed: %s", type(error).__name__)
        return error_message(DATABASE_UNAVAILABLE)

    try:
        top_k, min_similarity = retrieval_service.get_retrieval_settings()
        known_products = load_known_products(connection)
        file_label_words = answer_service.load_file_label_words(connection)
        chunks = retrieval_service.search_chunks(connection, question, top_k=top_k, min_similarity=min_similarity)
        result = answer_service.generate_answer(question, chunks, known_products=known_products,
                                                file_label_words=file_label_words)
    except answer_service.ConfigurationError as error:
        logger.warning("Answer service not configured: %s", type(error).__name__)
        return error_message(NOT_CONFIGURED)
    except (retrieval_service.RetrievalError, psycopg.Error) as error:
        logger.warning("Knowledge base search failed: %s", type(error).__name__)
        return error_message(DATABASE_UNAVAILABLE)
    except Exception as error:
        logger.warning("Unexpected error while answering: %s", type(error).__name__)
        return error_message(UNEXPECTED_ERROR)
    finally:
        connection.close()
    return result_message(result)


def format_citation(citation: dict) -> str:
    """'CZARLIV_SYRUP.pdf · page 1 · chunk 2', plus the source URL when there is one."""
    parts = [citation["document"]]
    if citation.get("location"):
        parts.append(citation["location"])    # "page 1", or DOCX paragraphs / section
    elif citation.get("page") is not None:
        parts.append(f"page {citation['page']}")
    parts.append(f"chunk {citation['chunk_id']}")
    text = " · ".join(parts)
    if citation.get("source_url"):
        text += f" ({citation['source_url']})"
    return text


def show_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
            return
        if message["kind"] == "answer":
            st.markdown(message["content"])
        elif message["kind"] == "abstention":
            st.warning(message["content"])
        else:
            st.error(message["content"])
        if message["missing_information"]:
            st.markdown("**Not covered by the approved documents:**")
            for item in message["missing_information"]:
                st.markdown(f"- {item}")
        if message["conflicts"]:
            st.markdown("**The approved sources disagree:**")
            for item in message["conflicts"]:
                st.markdown(f"- {item}")
        if message["citations"]:
            st.caption("Sources")
            for citation in message["citations"]:
                st.caption(format_citation(citation))


def main() -> None:
    st.set_page_config(page_title="PharmaAssist")
    st.title("PharmaAssist")
    st.caption("Answers questions using only approved product documents and shows the sources. "
               "It does not give medical advice: please consult a qualified healthcare professional "
               "or pharmacist.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        show_message(message)

    question = st.chat_input("Ask about an approved product")
    if question and question.strip():
        user_message = {"role": "user", "content": question}
        st.session_state.messages.append(user_message)
        show_message(user_message)
        with st.spinner("Searching approved documents and checking the answer..."):
            reply = answer_question(question)
        st.session_state.messages.append(reply)
        show_message(reply)


if __name__ == "__main__":
    main()
