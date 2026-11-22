"""Tests for semantic retrieval.

Integration tests use ONLY pharmaassist_test (schema created and rolled back by the `db` fixture)
and the application stand-in role, which can read nothing except products and the production view.
They use the real BGE model, so relevance is tested with real embeddings.
"""
from unittest.mock import MagicMock

import numpy as np
import psycopg
import pytest

import app.services.retrieval_service as retrieval_service
from app.services.embedding_service import embed_chunks
from app.services.retrieval_service import (
    DEFAULT_TOP_K,
    SEARCH_SQL,
    RetrievalError,
    RetrievedChunk,
    get_retrieval_settings,
    search_chunks,
)
from test_database_schema import APP, REVIEWER, as_role, db  # noqa: F401  (db is a pytest fixture)

STORAGE_TEXT = "Storage: Store below 25 degree C at room temperature."
COMPOSITION_TEXT = "COMPOSITION: Picrorhiza Kurroa (Kutki) 200 mg, Terminalia Chebula (Haritaki) 200 mg."
DOSAGE_TEXT = "Dosage: 10 ml twice a day or as directed by the physician."


def add_document(db, file_hash, texts, document_type="pdf", status="approved", product_id=None, superseded=False):
    """Store a document with real BGE embeddings (as the table owner), then review it as the reviewer."""
    document_id = db.execute(
        "INSERT INTO documents (file_name, file_hash, file_size_bytes, storage_uri, document_type, source_type, product_id) "
        "VALUES (%s, %s, 1, %s, %s, %s, %s) RETURNING id",
        (f"doc_{file_hash[:4]}.{document_type}", file_hash, f"file:///archive/{file_hash}.{document_type}",
         document_type, "product_leaflet" if document_type == "pdf" else "training_manual", product_id)).fetchone()[0]
    chunks = [{"text": text, "chunk_index": index} for index, text in enumerate(texts)]
    for chunk in embed_chunks(chunks):
        location = ((chunk["chunk_index"] + 1, None, None) if document_type == "pdf"
                    else (None, [chunk["chunk_index"]], []))
        chunk_id = db.execute(
            "INSERT INTO document_chunks (document_id, document_type, chunk_index, content, token_count, "
            "page_number, paragraph_indices, table_indices) VALUES (%s, %s, %s, %s, 10, %s, %s, %s) RETURNING id",
            (document_id, document_type, chunk["chunk_index"], chunk["text"], *location)).fetchone()[0]
        db.execute("INSERT INTO chunk_embeddings (chunk_id, embedding_model, embedding) VALUES (%s, %s, %s)",
                   (chunk_id, retrieval_service.MODEL_NAME, np.array(chunk["embedding"], dtype=np.float32)))
    if status != "pending":
        with as_role(db, REVIEWER):
            db.execute("UPDATE documents SET approval_status = %s, reviewed_by = 'tester', reviewed_at = now() "
                       "WHERE id = %s", (status, document_id))
            if superseded:
                db.execute("UPDATE documents SET superseded_at = now() WHERE id = %s", (document_id,))
    return document_id


def search_as_app(db, question, **kwargs):
    with as_role(db, APP):
        return search_chunks(db, question, **kwargs)


# ---------- integration tests (real PostgreSQL + real BGE) ----------

def test_relevant_chunk_is_ranked_first(db):
    add_document(db, "a" * 64, [COMPOSITION_TEXT, STORAGE_TEXT, DOSAGE_TEXT])
    results = search_as_app(db, "How should the syrup be stored?")
    assert results[0].content == STORAGE_TEXT
    results = search_as_app(db, "What is the dose?")
    assert results[0].content == DOSAGE_TEXT


def test_results_are_sorted_by_similarity(db):
    add_document(db, "a" * 64, [COMPOSITION_TEXT, STORAGE_TEXT, DOSAGE_TEXT])
    results = search_as_app(db, "Which ingredients does it contain?")
    similarities = [chunk.similarity for chunk in results]
    assert similarities == sorted(similarities, reverse=True)
    assert all(-1.0 <= s <= 1.0 for s in similarities)


def test_top_k_limits_the_number_of_results(db):
    add_document(db, "a" * 64, [f"Fictional product fact number {i}." for i in range(8)])
    assert len(search_as_app(db, "product fact")) == DEFAULT_TOP_K
    assert len(search_as_app(db, "product fact", top_k=2)) == 2
    assert len(search_as_app(db, "product fact", top_k=20)) == 8


def test_irrelevant_question_scores_lower_than_relevant_question(db):
    """Vector search always returns the nearest chunks, even for unrelated questions.

    This is why a threshold (chosen by evaluation) and abstention by the answer step are needed.
    """
    add_document(db, "a" * 64, [STORAGE_TEXT])
    relevant = search_as_app(db, "What is the storage temperature?")[0].similarity
    irrelevant = search_as_app(db, "Who won the football world cup?")[0].similarity
    assert irrelevant < relevant


def test_min_similarity_removes_weak_matches(db):
    add_document(db, "a" * 64, [STORAGE_TEXT])
    best = search_as_app(db, "Who won the football world cup?")[0].similarity
    assert search_as_app(db, "Who won the football world cup?", min_similarity=best + 0.01) == []
    assert len(search_as_app(db, "Who won the football world cup?", min_similarity=best - 0.01)) == 1


def test_empty_result_when_nothing_is_approved(db):
    assert search_as_app(db, "How should the syrup be stored?") == []
    add_document(db, "a" * 64, [STORAGE_TEXT], status="pending")
    assert search_as_app(db, "How should the syrup be stored?") == []


def test_only_approved_current_documents_are_returned(db):
    approved = add_document(db, "1" * 64, [DOSAGE_TEXT], status="approved")
    add_document(db, "2" * 64, [STORAGE_TEXT], status="pending")
    add_document(db, "3" * 64, [STORAGE_TEXT + " Pending copy B."], status="rejected")
    add_document(db, "4" * 64, [STORAGE_TEXT + " Old version."], status="approved", superseded=True)
    results = search_as_app(db, "How should the syrup be stored?", top_k=20)
    assert {chunk.document_id for chunk in results} == {approved}
    assert [chunk.content for chunk in results] == [DOSAGE_TEXT]


def test_source_metadata_is_returned(db):
    db.execute("INSERT INTO products VALUES ('CZARLIV', 'CZARLIV SYRUP')")
    pdf_id = add_document(db, "a" * 64, [COMPOSITION_TEXT, STORAGE_TEXT], product_id="CZARLIV")
    docx_id = add_document(db, "b" * 64, ["General probiotic information for training."], document_type="docx")

    results = search_as_app(db, "How should the syrup be stored?", top_k=20)
    storage = next(chunk for chunk in results if chunk.content == STORAGE_TEXT)
    assert (storage.document_id, storage.file_name, storage.document_type) == (pdf_id, f"doc_aaaa.pdf", "pdf")
    assert (storage.page_number, storage.product_name, storage.source_type) == (2, "CZARLIV SYRUP", "product_leaflet")
    assert storage.location() == "page 2"
    assert storage.chunk_id > 0

    training = next(chunk for chunk in results if chunk.document_id == docx_id)
    assert training.page_number is None and training.paragraph_indices == [0] and training.table_indices == []
    assert training.product_name is None
    assert training.location() == "paragraphs [0]"


def test_other_embedding_models_are_ignored(db):
    document_id = add_document(db, "a" * 64, [STORAGE_TEXT])
    chunk_id = db.execute("SELECT id FROM document_chunks WHERE document_id = %s", (document_id,)).fetchone()[0]
    other = np.zeros(384, dtype=np.float32)
    other[0] = 1.0
    db.execute("INSERT INTO chunk_embeddings (chunk_id, embedding_model, embedding) VALUES (%s, 'other-model', %s)",
               (chunk_id, other))
    results = search_as_app(db, "storage", top_k=20)
    assert len(results) == 1


def test_application_role_cannot_bypass_the_view(db):
    add_document(db, "a" * 64, [STORAGE_TEXT], status="pending")
    with as_role(db, APP):
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            with db.transaction():
                db.execute("SELECT content FROM document_chunks").fetchall()


# ---------- unit tests (no database) ----------

def test_search_sql_reads_only_the_production_view():
    sql = SEARCH_SQL.lower()
    assert "from production_chunk_embeddings" in sql
    for raw_table in ("documents", "document_chunks", "chunk_embeddings", "ingestion_runs"):
        assert f"from {raw_table}" not in sql and f"join {raw_table}" not in sql
    assert "%(query)s" in sql and "%(model)s" in sql and "%(top_k)s" in sql


def test_database_failure_raises_retrieval_error(monkeypatch):
    monkeypatch.setattr(retrieval_service, "embed_query", lambda question: [0.0] * 384)
    connection = MagicMock()
    connection.execute.side_effect = psycopg.OperationalError("server closed the connection")
    with pytest.raises(RetrievalError, match="Knowledge base search failed"):
        search_chunks(connection, "How should it be stored?")


def test_empty_question_is_rejected():
    with pytest.raises(ValueError, match="question cannot be empty"):
        search_chunks(MagicMock(), "   ")


@pytest.mark.parametrize("top_k, min_similarity", [(0, None), (21, None), (5, 1.5), (5, -2.0)])
def test_invalid_settings_are_rejected(top_k, min_similarity):
    with pytest.raises(ValueError):
        search_chunks(MagicMock(), "question", top_k=top_k, min_similarity=min_similarity)


def test_settings_come_from_the_environment(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_TOP_K", raising=False)
    monkeypatch.delenv("RETRIEVAL_MIN_SIMILARITY", raising=False)
    assert get_retrieval_settings() == (5, None)
    monkeypatch.setenv("RETRIEVAL_TOP_K", "3")
    monkeypatch.setenv("RETRIEVAL_MIN_SIMILARITY", "")
    assert get_retrieval_settings() == (3, None)
    monkeypatch.setenv("RETRIEVAL_MIN_SIMILARITY", "0.6")
    assert get_retrieval_settings() == (3, 0.6)


def test_open_retrieval_connection_refuses_other_roles(monkeypatch):
    fake = MagicMock()
    fake.execute.return_value.fetchone.return_value = ("pharmaassist_reviewer",)
    monkeypatch.setattr(retrieval_service, "get_connection", lambda variable: fake)
    with pytest.raises(RuntimeError, match="must run as pharmaassist_app"):
        retrieval_service.open_retrieval_connection()
    fake.close.assert_called_once()


def test_docx_location_text():
    chunk = RetrievedChunk(chunk_id=1, document_id=1, file_name="m.docx", document_type="docx",
                           source_type="training_manual", paragraph_indices=[67, 69], table_indices=[0],
                           section_heading="Composition", content="x", similarity=0.5)
    assert chunk.location() == "paragraphs [67, 69], tables [0], section 'Composition'"
