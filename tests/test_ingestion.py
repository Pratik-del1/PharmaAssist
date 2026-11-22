"""Integration tests for document ingestion and review, using ONLY the pharmaassist_test database.

The schema is created inside a rolled-back transaction by the `db` fixture from
test_database_schema.py, and the application and reviewer roles are the test stand-ins.
Archived files go to pytest's temporary folder, never to data/archive.
"""
import numpy as np
import pymupdf
import pytest
from docx import Document

import app.services.ingestion_service as ingestion_service
from app.database.connection import require_current_user
from app.services.embedding_service import EMBEDDING_DIMENSION, MODEL_NAME
from app.services.ingestion_service import ingest_file
from app.services.review_service import get_chunks, get_document, list_documents, set_review_status, supersede
from app.services.vector_store import compute_file_hash, save_document
from test_database_schema import APP, REVIEWER, as_role, db  # noqa: F401  (db is a pytest fixture)


def create_pdf(path, page_texts):
    document = pymupdf.open()
    for text in page_texts:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def create_docx(path):
    document = Document()
    document.add_heading("Composition", level=1)
    document.add_paragraph("Each capsule contains:")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Ingredient"
    table.cell(0, 1).text = "Strength"
    table.cell(1, 0).text = "Ingredient A"
    table.cell(1, 1).text = "100 mg"
    document.save(path)


@pytest.fixture
def leaflet(tmp_path):
    path = tmp_path / "leaflet.pdf"
    create_pdf(path, ["DemoCompound Alpha syrup leaflet.", "Store below 25 degree C."])
    return path


def count(db, table):
    return db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def test_pdf_is_ingested_as_pending_with_chunks_and_embeddings(db, leaflet, tmp_path):
    db.execute("INSERT INTO products VALUES ('DEMO-001', 'DemoCompound Alpha')")
    with as_role(db, APP):
        result = ingest_file(db, leaflet, "product_leaflet", product_id="DEMO-001", archive_dir=tmp_path / "archive")

    assert result["status"] == "succeeded"
    assert result["file_hash"] == compute_file_hash(leaflet)
    document = db.execute("SELECT approval_status, document_type, product_id, file_size_bytes, storage_uri, "
                          "extraction_metadata FROM documents WHERE id = %s", (result["document_id"],)).fetchone()
    status, document_type, product_id, size, storage_uri, metadata = document
    assert (status, document_type, product_id) == ("pending", "pdf", "DEMO-001")
    assert size == leaflet.stat().st_size
    assert metadata["page_count"] == 2 and metadata["chunk_size"] == 800 and metadata["embedding_model"] == MODEL_NAME

    archived = tmp_path / "archive" / f"{result['file_hash']}.pdf"
    assert storage_uri == archived.resolve().as_uri()
    assert compute_file_hash(archived) == result["file_hash"]

    chunks = db.execute("SELECT c.page_number, e.embedding FROM document_chunks c "
                        "JOIN chunk_embeddings e ON e.chunk_id = c.id ORDER BY c.chunk_index").fetchall()
    assert [page for page, _ in chunks] == [1, 2]
    for _, embedding in chunks:
        assert len(embedding) == EMBEDDING_DIMENSION
        assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-5)
    assert db.execute("SELECT status, chunk_count FROM ingestion_runs").fetchall() == [("succeeded", 2)]


def test_docx_is_ingested_with_paragraph_and_table_locations(db, tmp_path):
    path = tmp_path / "manual.docx"
    create_docx(path)
    with as_role(db, APP):
        result = ingest_file(db, path, "training_manual", archive_dir=tmp_path / "archive")
    rows = db.execute("SELECT document_type, paragraph_indices, table_indices, section_heading, content "
                      "FROM document_chunks").fetchall()
    assert result["status"] == "succeeded"
    assert rows == [("docx", [0, 1], [0], "Composition",
                     "Composition\nEach capsule contains:\nIngredient: Ingredient A | Strength: 100 mg")]


def test_duplicate_is_detected_by_hash_before_any_work(db, leaflet, tmp_path, monkeypatch):
    with as_role(db, APP):
        first = ingest_file(db, leaflet, "product_leaflet", archive_dir=tmp_path / "archive")

        def must_not_be_called(*args, **kwargs):
            raise AssertionError("a duplicate must not be extracted or embedded again")
        monkeypatch.setattr(ingestion_service, "extract_and_chunk", must_not_be_called)
        monkeypatch.setattr(ingestion_service, "embed_chunks", must_not_be_called)

        renamed_copy = tmp_path / "renamed_copy.pdf"
        renamed_copy.write_bytes(leaflet.read_bytes())
        second = ingest_file(db, renamed_copy, "product_leaflet", archive_dir=tmp_path / "archive")

    assert second == {"status": "skipped_duplicate", "document_id": first["document_id"], "file_hash": first["file_hash"]}
    assert count(db, "documents") == 1
    assert count(db, "document_chunks") == 2
    assert count(db, "chunk_embeddings") == 2
    assert db.execute("SELECT status, document_id FROM ingestion_runs ORDER BY id").fetchall() == [
        ("succeeded", first["document_id"]), ("skipped_duplicate", first["document_id"])]


def test_failed_ingestion_is_rolled_back_but_logged(db, leaflet, tmp_path, monkeypatch):
    # A token count over 512 is rejected by the database in the middle of the document transaction.
    monkeypatch.setattr(ingestion_service, "count_tokens", lambda text: 999)
    with as_role(db, APP):
        with pytest.raises(Exception, match="token_count"):
            ingest_file(db, leaflet, "product_leaflet", archive_dir=tmp_path / "archive")

    assert count(db, "documents") == 0
    assert count(db, "document_chunks") == 0
    assert count(db, "chunk_embeddings") == 0
    status, file_hash, error = db.execute("SELECT status, file_hash, error_message FROM ingestion_runs").fetchone()
    assert status == "failed"
    assert file_hash == compute_file_hash(leaflet)
    assert "token_count" in error


def test_missing_file_is_logged_without_hash(db, tmp_path):
    with as_role(db, APP):
        with pytest.raises(FileNotFoundError):
            ingest_file(db, tmp_path / "missing.pdf", "product_leaflet", archive_dir=tmp_path / "archive")
    assert db.execute("SELECT file_name, file_hash, status FROM ingestion_runs").fetchall() == [
        ("missing.pdf", None, "failed")]


def test_unsupported_file_type_is_logged(db, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Some notes")
    with as_role(db, APP):
        with pytest.raises(ValueError, match="Unsupported file type"):
            ingest_file(db, path, "product_leaflet", archive_dir=tmp_path / "archive")
    status, file_hash = db.execute("SELECT status, file_hash FROM ingestion_runs").fetchone()
    assert status == "failed" and file_hash == compute_file_hash(path)


def test_unknown_chunk_keys_are_rejected_before_storing(db):
    document = {"file_name": "x.pdf", "file_hash": "a" * 64, "file_size_bytes": 1, "storage_uri": "file:///x.pdf",
                "document_type": "pdf", "source_type": "product_leaflet", "source_url": None, "retrieved_at": None,
                "product_id": None, "extraction_metadata": {}}
    chunk = {"text": "x", "chunk_index": 0, "page_number": 1, "product_id": None, "file_name": "x.pdf",
             "token_count": 3, "embedding": [0.0] * EMBEDDING_DIMENSION, "unexpected": "value"}
    with as_role(db, APP):
        with pytest.raises(ValueError, match="unexpected keys"):
            save_document(db, document, [chunk], MODEL_NAME, started_at=None)
    assert count(db, "documents") == 0


def test_require_current_user_stops_on_the_wrong_role(db):
    with as_role(db, REVIEWER):
        require_current_user(db, REVIEWER)
        with pytest.raises(RuntimeError, match="must run as"):
            require_current_user(db, APP)


def test_reviewer_can_inspect_approve_and_supersede(db, leaflet, tmp_path):
    with as_role(db, APP):
        document_id = ingest_file(db, leaflet, "product_leaflet", archive_dir=tmp_path / "archive")["document_id"]

    with as_role(db, REVIEWER):
        assert [d["id"] for d in list_documents(db)] == [document_id]
        assert get_document(db, document_id)["approval_status"] == "pending"
        assert [c["page_number"] for c in get_chunks(db, document_id)] == [1, 2]
        assert supersede(db, document_id) is False             # not approved yet
        assert set_review_status(db, document_id, "approved", "Test Reviewer") is True
        assert set_review_status(db, document_id, "rejected", "Test Reviewer") is False   # no longer pending

    with as_role(db, APP):
        assert db.execute("SELECT count(*) FROM production_chunk_embeddings").fetchone()[0] == 2

    with as_role(db, REVIEWER):
        assert supersede(db, document_id) is True
    with as_role(db, APP):
        assert db.execute("SELECT count(*) FROM production_chunk_embeddings").fetchone()[0] == 0


def test_review_requires_a_reviewer_name(db):
    with pytest.raises(ValueError, match="reviewer name"):
        set_review_status(db, 1, "approved", "  ")
