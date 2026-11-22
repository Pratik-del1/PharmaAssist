"""Integration tests for app/database/schema.sql and permissions.sql.

These tests use the real PostgreSQL server, but ONLY the pharmaassist_test database.
Every test creates the schema inside one transaction and rolls it back at the end,
so nothing is left behind in the database.

The application, reviewer and view-owner perspectives are tested with SET ROLE to the
non-login stand-in roles pharmaassist_test_app, pharmaassist_test_reviewer and
pharmaassist_test_view_owner.
"""
import os
import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.types.json import Jsonb

from app.database.connection import get_connection, get_database_url

TEST_DATABASE = "pharmaassist_test"
TEST_USER = "pharmaassist_test"
APP = "pharmaassist_test_app"
REVIEWER = "pharmaassist_test_reviewer"
VIEW_OWNER = "pharmaassist_test_view_owner"
MODEL = "BAAI/bge-small-en-v1.5"

DATABASE_DIR = Path(__file__).resolve().parents[1] / "app" / "database"


def read_sql(file_name, replacements=()):
    """Read a SQL file without its BEGIN/COMMIT lines, so it can run inside the test transaction."""
    sql = (DATABASE_DIR / file_name).read_text()
    sql = re.sub(r"^(BEGIN|COMMIT);\s*$", "", sql, flags=re.MULTILINE)
    for old, new in replacements:
        sql = sql.replace(old, new)
    return sql


SCHEMA_SQL = read_sql("schema.sql")
# The real role names are replaced with the test stand-ins.
PERMISSIONS_SQL = read_sql("permissions.sql", [
    ("pharmaassist_view_owner", VIEW_OWNER),
    ("pharmaassist_reviewer", REVIEWER),
    ("pharmaassist_app", APP),
])


def require_test_database_url():
    """Return TEST_DATABASE_URL, or skip if it is missing or points at another database."""
    try:
        url = get_database_url("TEST_DATABASE_URL")
    except RuntimeError:
        pytest.skip("TEST_DATABASE_URL is not set")
    if conninfo_to_dict(url).get("dbname") != TEST_DATABASE:
        pytest.skip(f"TEST_DATABASE_URL does not point at {TEST_DATABASE}; refusing to run")
    return url


@pytest.fixture
def db():
    """A connection to pharmaassist_test with the schema created inside a rolled-back transaction."""
    require_test_database_url()
    connection = get_connection("TEST_DATABASE_URL")
    try:
        database, user = connection.execute("SELECT current_database(), current_user").fetchone()
        if database != TEST_DATABASE or user != TEST_USER:
            pytest.fail(f"Refusing to run: connected to {database} as {user}")
        existing = connection.execute(
            "SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = 'public'").fetchone()[0]
        if existing:
            pytest.fail(f"Refusing to run: {TEST_DATABASE} already contains {existing} objects")
        connection.rollback()

        with connection.transaction(force_rollback=True):
            connection.execute(SCHEMA_SQL)
            connection.execute(PERMISSIONS_SQL)
            yield connection
    finally:
        connection.close()


@contextmanager
def as_role(connection, role):
    # If the block fails, the surrounding savepoint or transaction is rolled back, and
    # PostgreSQL undoes the SET ROLE with it. So the role is only reset after success.
    connection.execute(f"SET ROLE {role}")
    yield
    connection.execute("RESET ROLE")


def rejected(connection, sql, params=None, error=psycopg.Error, role=None):
    """Run sql inside a savepoint and return the error it raises (the test fails if it succeeds)."""
    with pytest.raises(error) as caught:
        with connection.transaction():
            if role:
                connection.execute(f"SET ROLE {role}")   # undone when the savepoint rolls back
            connection.execute(sql, params)
    return caught.value


def unit_vector(position):
    vector = np.zeros(384, dtype=np.float32)
    vector[position] = 1.0
    return vector


def insert_document(connection, file_hash="a" * 64, document_type="pdf", **columns):
    values = {
        "file_name": f"leaflet.{document_type}",
        "file_hash": file_hash,
        "file_size_bytes": 100,
        "storage_uri": f"file:///archive/{file_hash}.{document_type}",
        "document_type": document_type,
        "source_type": "product_leaflet",
    }
    values.update(columns)
    names = ", ".join(values)
    placeholders = ", ".join(f"%({name})s" for name in values)
    return connection.execute(
        f"INSERT INTO documents ({names}) VALUES ({placeholders}) RETURNING id", values).fetchone()[0]


def insert_chunk(connection, document_id, document_type="pdf", chunk_index=0,
                 content="Store below 25°C.", **location):
    if not location:
        location = {"page_number": 1} if document_type == "pdf" else {"paragraph_indices": [0], "table_indices": []}
    values = {"document_id": document_id, "document_type": document_type, "chunk_index": chunk_index,
              "content": content, "token_count": 8}
    values.update(location)
    names = ", ".join(values)
    placeholders = ", ".join(f"%({name})s" for name in values)
    return connection.execute(
        f"INSERT INTO document_chunks ({names}) VALUES ({placeholders}) RETURNING id", values).fetchone()[0]


def insert_embedding(connection, chunk_id, vector, model=MODEL):
    connection.execute(
        "INSERT INTO chunk_embeddings (chunk_id, embedding_model, embedding) VALUES (%s, %s, %s)",
        (chunk_id, model, vector))


def review(connection, document_id, status):
    with as_role(connection, REVIEWER):
        connection.execute(
            "UPDATE documents SET approval_status = %s, reviewed_by = 'test reviewer', reviewed_at = now() "
            "WHERE id = %s", (status, document_id))


def stored_document_with_embedding(connection, file_hash="a" * 64, content="Store below 25°C.", position=0):
    document_id = insert_document(connection, file_hash=file_hash)
    chunk_id = insert_chunk(connection, document_id, content=content)
    insert_embedding(connection, chunk_id, unit_vector(position))
    return document_id


def search_as_app(connection):
    query = unit_vector(0)
    with as_role(connection, APP):
        return connection.execute(
            "SELECT file_name, product_name, page_number, 1 - (embedding <=> %s) AS similarity "
            "FROM production_chunk_embeddings WHERE embedding_model = %s "
            "ORDER BY embedding <=> %s LIMIT 5", (query, MODEL, query)).fetchall()


# ---------- the guard and the objects ----------

def test_connected_to_test_database_only(db):
    assert db.execute("SELECT current_database(), current_user").fetchone() == (TEST_DATABASE, TEST_USER)


def test_all_six_objects_exist(db):
    objects = db.execute(
        "SELECT relname, relkind FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = 'public' AND relkind IN ('r', 'v') ORDER BY relname").fetchall()
    assert objects == [
        ("chunk_embeddings", "r"),
        ("document_chunks", "r"),
        ("documents", "r"),
        ("ingestion_runs", "r"),
        ("production_chunk_embeddings", "v"),
        ("products", "r"),
    ]


def test_view_owner_and_options(db):
    owner, options = db.execute(
        "SELECT pg_get_userbyid(relowner), reloptions FROM pg_class "
        "WHERE relname = 'production_chunk_embeddings'").fetchone()
    assert owner == VIEW_OWNER
    assert set(options) == {"security_barrier=true", "security_invoker=false"}


def test_view_does_not_expose_internal_columns(db):
    columns = {row[0] for row in db.execute(
        "SELECT attname FROM pg_attribute WHERE attrelid = 'production_chunk_embeddings'::regclass "
        "AND attnum > 0").fetchall()}
    assert {"content", "embedding", "source_url", "product_name", "page_number", "paragraph_indices"} <= columns
    for internal in ["storage_uri", "file_hash", "reviewed_by", "reviewed_at", "approval_status", "extraction_metadata"]:
        assert internal not in columns


def test_view_owner_grants_match_view_columns(db):
    used = set(db.execute(
        """SELECT DISTINCT d.refobjid::regclass::text, a.attname::text
           FROM pg_depend d
           JOIN pg_rewrite r ON r.oid = d.objid
           JOIN pg_attribute a ON a.attrelid = d.refobjid AND a.attnum = d.refobjsubid
           WHERE r.ev_class = 'production_chunk_embeddings'::regclass AND d.refobjsubid > 0
             AND d.refobjid <> 'production_chunk_embeddings'::regclass""").fetchall())
    granted = set(db.execute(
        "SELECT table_name::text, column_name::text FROM information_schema.column_privileges "
        "WHERE grantee = %s AND privilege_type = 'SELECT' "
        "AND table_name <> 'production_chunk_embeddings'", (VIEW_OWNER,)).fetchall())
    assert used == granted
    assert len(used) == 23


# ---------- documents ----------

def test_new_document_is_pending_with_no_reviewer(db):
    document_id = insert_document(db)
    status, reviewed_by, reviewed_at = db.execute(
        "SELECT approval_status, reviewed_by, reviewed_at FROM documents WHERE id = %s", (document_id,)).fetchone()
    assert (status, reviewed_by, reviewed_at) == ("pending", None, None)


def test_duplicate_file_hash_is_rejected(db):
    insert_document(db, file_hash="a" * 64)
    rejected(db, "INSERT INTO documents (file_name, file_hash, file_size_bytes, storage_uri, document_type, source_type) "
                 "VALUES ('copy.pdf', %s, 1, 'file:///other.pdf', 'pdf', 'product_leaflet')",
             ("a" * 64,), error=psycopg.errors.UniqueViolation)


@pytest.mark.parametrize("columns", [
    {"file_hash": "not-a-hash"},
    {"document_type": "html"},
    {"source_type": "web_page"},
    {"storage_uri": "/archive/no-scheme.pdf"},
    {"extraction_metadata": Jsonb([1, 2])},
    {"source_type": "external_article"},                                       # no source_url
    {"source_url": "http://example.com/a", "retrieved_at": "2026-01-01"},      # not https
    {"source_url": "https://example.com/a"},                                   # no retrieved_at
    {"retrieved_at": "2026-01-01"},                                            # no source_url
])
def test_invalid_documents_are_rejected(db, columns):
    values = {"file_hash": "b" * 64}
    values.update(columns)
    with pytest.raises(psycopg.errors.CheckViolation):
        with db.transaction():
            insert_document(db, **values)


def test_external_article_with_https_url_is_accepted(db):
    document_id = insert_document(db, file_hash="c" * 64, source_type="external_article",
                                  source_url="https://example.com/article", retrieved_at="2026-01-01")
    assert document_id > 0


def test_product_id_must_exist_and_be_uppercase(db):
    rejected(db, "INSERT INTO products VALUES ('czarliv', 'lowercase')", error=psycopg.errors.CheckViolation)
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        with db.transaction():
            insert_document(db, product_id="UNKNOWN")


def test_review_requires_reviewer_and_time(db):
    document_id = insert_document(db)
    rejected(db, "UPDATE documents SET approval_status = 'approved' WHERE id = %s", (document_id,),
             error=psycopg.errors.CheckViolation)
    rejected(db, "UPDATE documents SET approval_status = 'rejected', reviewed_by = 'x' WHERE id = %s", (document_id,),
             error=psycopg.errors.CheckViolation)


def test_only_approved_documents_can_be_superseded(db):
    document_id = insert_document(db)
    rejected(db, "UPDATE documents SET superseded_at = now() WHERE id = %s", (document_id,),
             error=psycopg.errors.CheckViolation)


# ---------- chunks: PDF and DOCX location rules ----------

def test_valid_pdf_and_docx_chunks_are_accepted(db):
    pdf_id = insert_document(db, file_hash="a" * 64, document_type="pdf")
    docx_id = insert_document(db, file_hash="b" * 64, document_type="docx", source_type="training_manual")
    insert_chunk(db, pdf_id, "pdf", page_number=1)
    insert_chunk(db, docx_id, "docx", chunk_index=0, paragraph_indices=[67], table_indices=[0],
                 section_heading="Composition", heading_level=1)
    insert_chunk(db, docx_id, "docx", chunk_index=1, paragraph_indices=[], table_indices=[0])  # table-only
    assert db.execute("SELECT count(*) FROM document_chunks").fetchone()[0] == 3


@pytest.mark.parametrize("location", [
    {},                                                             # PDF without page_number
    {"page_number": 1, "paragraph_indices": [0], "table_indices": []},
    {"page_number": 1, "section_heading": "Storage"},
    {"page_number": 1, "heading_level": 1, "section_heading": "Storage"},
])
def test_invalid_pdf_chunk_locations_are_rejected(db, location):
    pdf_id = insert_document(db)
    if not location:
        location = {"page_number": None}
    with pytest.raises(psycopg.errors.CheckViolation):
        with db.transaction():
            insert_chunk(db, pdf_id, "pdf", **location)


@pytest.mark.parametrize("location", [
    {"page_number": 1, "paragraph_indices": [0], "table_indices": []},   # DOCX with a page number
    {"paragraph_indices": [0]},                                         # missing table_indices
    {"table_indices": [0]},                                             # missing paragraph_indices
    {"paragraph_indices": [0], "table_indices": [], "heading_level": 1},  # level without heading
])
def test_invalid_docx_chunk_locations_are_rejected(db, location):
    docx_id = insert_document(db, document_type="docx", source_type="training_manual")
    with pytest.raises(psycopg.errors.CheckViolation):
        with db.transaction():
            insert_chunk(db, docx_id, "docx", **location)


def test_chunk_type_must_match_its_document(db):
    docx_id = insert_document(db, document_type="docx", source_type="training_manual")
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        with db.transaction():
            insert_chunk(db, docx_id, "pdf", page_number=1)


def test_chunk_must_belong_to_an_existing_document(db):
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        with db.transaction():
            insert_chunk(db, 999999, "pdf", page_number=1)


@pytest.mark.parametrize("bad", [{"token_count": 513}, {"token_count": 0}, {"content": ""}])
def test_invalid_chunk_values_are_rejected(db, bad):
    pdf_id = insert_document(db)
    values = {"document_id": pdf_id, "document_type": "pdf", "chunk_index": 0, "content": "x",
              "token_count": 1, "page_number": 1}
    values.update(bad)
    rejected(db, "INSERT INTO document_chunks (document_id, document_type, chunk_index, content, token_count, page_number) "
                 "VALUES (%(document_id)s, %(document_type)s, %(chunk_index)s, %(content)s, %(token_count)s, %(page_number)s)",
             values, error=psycopg.errors.CheckViolation)


def test_duplicate_chunk_index_is_rejected(db):
    pdf_id = insert_document(db)
    insert_chunk(db, pdf_id, "pdf", chunk_index=0)
    with pytest.raises(psycopg.errors.UniqueViolation):
        with db.transaction():
            insert_chunk(db, pdf_id, "pdf", chunk_index=0)


# ---------- embeddings ----------

def test_embedding_must_have_384_dimensions(db):
    chunk_id = insert_chunk(db, insert_document(db))
    rejected(db, "INSERT INTO chunk_embeddings (chunk_id, embedding_model, embedding) VALUES (%s, %s, %s)",
             (chunk_id, MODEL, np.ones(383, dtype=np.float32) / np.sqrt(383)), error=psycopg.errors.DataException)


def test_embedding_must_be_normalized(db):
    chunk_id = insert_chunk(db, insert_document(db))
    rejected(db, "INSERT INTO chunk_embeddings (chunk_id, embedding_model, embedding) VALUES (%s, %s, %s)",
             (chunk_id, MODEL, unit_vector(0) * 2), error=psycopg.errors.CheckViolation)
    insert_embedding(db, chunk_id, unit_vector(0))
    norm = db.execute("SELECT vector_norm(embedding) FROM chunk_embeddings").fetchone()[0]
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_duplicate_embedding_is_rejected(db):
    chunk_id = insert_chunk(db, insert_document(db))
    insert_embedding(db, chunk_id, unit_vector(0))
    with pytest.raises(psycopg.errors.UniqueViolation):
        with db.transaction():
            insert_embedding(db, chunk_id, unit_vector(0))


def test_embedding_must_belong_to_an_existing_chunk(db):
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        with db.transaction():
            insert_embedding(db, 999999, unit_vector(0))


def test_deleting_a_document_removes_its_chunks_and_embeddings(db):
    document_id = stored_document_with_embedding(db)
    db.execute("INSERT INTO ingestion_runs (file_name, file_hash, status, document_id, started_at) "
               "VALUES ('leaflet.pdf', %s, 'succeeded', %s, now())", ("a" * 64, document_id))
    db.execute("DELETE FROM documents WHERE id = %s", (document_id,))   # an administrator action
    assert db.execute("SELECT count(*) FROM document_chunks").fetchone()[0] == 0
    assert db.execute("SELECT count(*) FROM chunk_embeddings").fetchone()[0] == 0
    assert db.execute("SELECT document_id FROM ingestion_runs").fetchone()[0] is None


# ---------- ingestion log ----------

def test_ingestion_log_rules(db):
    log = "INSERT INTO ingestion_runs (file_name, file_hash, status, error_message, started_at) VALUES (%s, %s, %s, %s, now())"
    with as_role(db, APP):
        db.execute(log, ("missing.pdf", None, "failed", "file not found"))   # failure before hashing
    rejected(db, log, ("leaflet.pdf", None, "succeeded", None), error=psycopg.errors.CheckViolation, role=APP)
    rejected(db, log, ("leaflet.pdf", "a" * 64, "failed", None), error=psycopg.errors.CheckViolation, role=APP)


def test_failure_log_survives_rolled_back_document(db):
    with pytest.raises(psycopg.errors.CheckViolation):
        with db.transaction():                       # the document's own transaction
            with as_role(db, APP):
                document_id = insert_document(db, file_hash="d" * 64)
                insert_chunk(db, document_id, "pdf", token_count=999)
    with as_role(db, APP):                            # the failure is logged separately afterwards
        db.execute("INSERT INTO ingestion_runs (file_name, file_hash, status, error_message, started_at) "
                   "VALUES ('broken.pdf', %s, 'failed', 'too many tokens', now())", ("d" * 64,))
    assert db.execute("SELECT count(*) FROM documents").fetchone()[0] == 0
    assert db.execute("SELECT status FROM ingestion_runs").fetchall() == [("failed",)]


# ---------- approval filtering in the production view ----------

def test_pending_document_is_not_retrievable(db):
    stored_document_with_embedding(db)
    assert search_as_app(db) == []


def test_approved_document_is_retrievable(db):
    db.execute("INSERT INTO products VALUES ('CZARLIV', 'CZARLIV SYRUP')")
    document_id = insert_document(db, product_id="CZARLIV")
    chunk_id = insert_chunk(db, document_id)
    insert_embedding(db, chunk_id, unit_vector(0))
    review(db, document_id, "approved")
    rows = search_as_app(db)
    assert len(rows) == 1
    file_name, product_name, page_number, similarity = rows[0]
    assert (file_name, product_name, page_number) == ("leaflet.pdf", "CZARLIV SYRUP", 1)
    assert similarity == pytest.approx(1.0, abs=1e-6)


def test_rejected_document_is_not_retrievable(db):
    document_id = stored_document_with_embedding(db)
    review(db, document_id, "rejected")
    assert search_as_app(db) == []


def test_superseded_document_is_not_retrievable(db):
    document_id = stored_document_with_embedding(db)
    review(db, document_id, "approved")
    assert len(search_as_app(db)) == 1
    with as_role(db, REVIEWER):
        db.execute("UPDATE documents SET superseded_at = now() WHERE id = %s", (document_id,))
    assert search_as_app(db) == []


def test_only_approved_current_documents_are_visible_together(db):
    pending = stored_document_with_embedding(db, file_hash="1" * 64, position=1)
    approved = stored_document_with_embedding(db, file_hash="2" * 64, position=2)
    rejected_doc = stored_document_with_embedding(db, file_hash="3" * 64, position=3)
    superseded = stored_document_with_embedding(db, file_hash="4" * 64, position=4)
    review(db, approved, "approved")
    review(db, rejected_doc, "rejected")
    review(db, superseded, "approved")
    with as_role(db, REVIEWER):
        db.execute("UPDATE documents SET superseded_at = now() WHERE id = %s", (superseded,))
    with as_role(db, APP):
        visible = db.execute("SELECT document_id FROM production_chunk_embeddings").fetchall()
    assert visible == [(approved,)]
    assert pending not in {row[0] for row in visible}


# ---------- security barrier: pending text must not leak through errors ----------

PENDING_TEXT = "Store below 25°C. Keep away from light."


def prepare_leak_scenario(db):
    """One PENDING chunk with non-numeric text, plus one APPROVED chunk with numeric text.

    The approved chunk forces PostgreSQL to really scan and join the chunk table. Index scans and
    nested loops are switched off (any role may do this) so every chunk row is read.
    """
    stored_document_with_embedding(db, file_hash="5" * 64, content=PENDING_TEXT, position=0)
    approved = stored_document_with_embedding(db, file_hash="6" * 64, content="12345", position=1)
    review(db, approved, "approved")
    for setting in ("enable_indexscan", "enable_bitmapscan", "enable_nestloop"):
        db.execute(f"SET LOCAL {setting} = off")


def test_security_barrier_blocks_pending_text_leak(db):
    prepare_leak_scenario(db)
    with as_role(db, APP):
        rows = db.execute("SELECT content FROM production_chunk_embeddings WHERE content::int > 0").fetchall()
    assert rows == [("12345",)]


def test_control_view_without_barrier_leaks_pending_text(db):
    """Shows the test above is meaningful: without security_barrier the same query exposes pending text."""
    prepare_leak_scenario(db)
    db.execute("CREATE VIEW unprotected_copy WITH (security_barrier = false) AS "
               "SELECT c.id, c.content FROM document_chunks c JOIN documents d ON d.id = c.document_id "
               "WHERE d.approval_status = 'approved' AND d.superseded_at IS NULL")
    db.execute(f"GRANT SELECT ON unprotected_copy TO {APP}")
    error = rejected(db, "SELECT id FROM unprotected_copy WHERE content::int > 0",
                     error=psycopg.errors.InvalidTextRepresentation, role=APP)
    assert PENDING_TEXT in error.diag.message_primary


# ---------- application role permissions ----------

def test_app_can_insert_a_pending_document(db):
    with as_role(db, APP):
        document_id = db.execute(
            "INSERT INTO documents (file_name, file_hash, file_size_bytes, storage_uri, document_type, source_type) "
            "VALUES ('leaflet.pdf', %s, 100, 'file:///archive/a.pdf', 'pdf', 'product_leaflet') "
            "ON CONFLICT (file_hash) DO NOTHING RETURNING id", ("a" * 64,)).fetchone()[0]
        chunk_id = insert_chunk(db, document_id)
        insert_embedding(db, chunk_id, unit_vector(0))
        duplicate = db.execute("SELECT id FROM documents WHERE file_hash = %s", ("a" * 64,)).fetchone()[0]
    assert duplicate == document_id
    assert db.execute("SELECT approval_status FROM documents").fetchone()[0] == "pending"


@pytest.mark.parametrize("column, value", [
    ("approval_status", "approved"),
    ("reviewed_by", "app"),
    ("reviewed_at", "2026-01-01"),
    ("superseded_at", "2026-01-01"),
    ("ingested_at", "2000-01-01"),
])
def test_app_cannot_set_review_or_timestamp_columns(db, column, value):
    rejected(db, f"INSERT INTO documents (file_name, file_hash, file_size_bytes, storage_uri, document_type, "
                 f"source_type, {column}) VALUES ('x.pdf', %s, 1, 'file:///x.pdf', 'pdf', 'product_leaflet', %s)",
             ("e" * 64, value), error=psycopg.errors.InsufficientPrivilege, role=APP)


@pytest.mark.parametrize("sql", [
    "UPDATE documents SET approval_status = 'approved', reviewed_by = 'app', reviewed_at = now()",
    "UPDATE documents SET file_name = 'x'",
    "UPDATE document_chunks SET content = 'x'",
    "UPDATE chunk_embeddings SET embedding_model = 'x'",
    "DELETE FROM documents",
    "DELETE FROM document_chunks",
    "DELETE FROM chunk_embeddings",
    "TRUNCATE documents CASCADE",
    "INSERT INTO products VALUES ('FAKE', 'Fake product')",
])
def test_app_cannot_approve_modify_or_delete(db, sql):
    stored_document_with_embedding(db)
    rejected(db, sql, error=psycopg.errors.InsufficientPrivilege, role=APP)


@pytest.mark.parametrize("sql", [
    "SELECT * FROM documents",
    "SELECT approval_status FROM documents",
    "SELECT storage_uri FROM documents",
    "SELECT content FROM document_chunks",
    "SELECT embedding FROM chunk_embeddings",
    "SELECT * FROM ingestion_runs",
])
def test_app_cannot_read_raw_tables(db, sql):
    rejected(db, sql, error=psycopg.errors.InsufficientPrivilege, role=APP)


def test_app_can_read_only_ingestion_columns_and_products(db):
    with as_role(db, APP):
        db.execute("SELECT id, file_hash FROM documents").fetchall()
        db.execute("SELECT id FROM document_chunks").fetchall()
        db.execute("SELECT product_id, product_name FROM products").fetchall()


@pytest.mark.parametrize("sql", [
    "CREATE TABLE new_table (x int)",
    "CREATE FUNCTION leak(text) RETURNS boolean LANGUAGE sql AS 'SELECT true'",
    "CREATE TEMP TABLE temporary_table (x int)",
    "CREATE ROLE new_role",
    "INSERT INTO production_chunk_embeddings (chunk_id) VALUES (1)",
])
def test_app_cannot_create_objects_or_write_the_view(db, sql):
    rejected(db, sql, role=APP)


# ---------- reviewer role permissions ----------

def test_reviewer_can_read_pending_content_for_review(db):
    stored_document_with_embedding(db)
    with as_role(db, REVIEWER):
        rows = db.execute("SELECT d.file_name, d.approval_status, c.content FROM documents d "
                          "JOIN document_chunks c ON c.document_id = d.id").fetchall()
    assert rows == [("leaflet.pdf", "pending", "Store below 25°C.")]


@pytest.mark.parametrize("sql", [
    "UPDATE documents SET file_name = 'x'",
    "UPDATE documents SET product_id = NULL",
    "UPDATE documents SET ingested_at = now()",
    "UPDATE document_chunks SET content = 'x'",
    "UPDATE chunk_embeddings SET embedding_model = 'x'",
    "SELECT embedding FROM chunk_embeddings",
    "INSERT INTO documents (file_name) VALUES ('x')",
    "INSERT INTO products VALUES ('FAKE', 'Fake product')",
    "DELETE FROM documents",
    "CREATE TABLE new_table (x int)",
])
def test_reviewer_cannot_modify_content_or_embeddings(db, sql):
    stored_document_with_embedding(db)
    rejected(db, sql, error=psycopg.errors.InsufficientPrivilege, role=REVIEWER)


# ---------- view-owner role permissions ----------

@pytest.mark.parametrize("sql", [
    "SELECT storage_uri FROM documents",
    "SELECT file_hash FROM documents",
    "SELECT reviewed_by FROM documents",
    "SELECT * FROM ingestion_runs",
    "INSERT INTO documents (file_name) VALUES ('x')",
    "UPDATE documents SET approval_status = 'approved'",
])
def test_view_owner_is_limited_to_view_columns(db, sql):
    rejected(db, sql, error=psycopg.errors.InsufficientPrivilege, role=VIEW_OWNER)


# ---------- test-role isolation ----------

def test_test_role_does_not_inherit_stand_in_privileges(db):
    for role in (APP, REVIEWER, VIEW_OWNER):
        inherits, can_set = db.execute(
            "SELECT pg_has_role(%s, %s, 'USAGE'), pg_has_role(%s, %s, 'SET')",
            (TEST_USER, role, TEST_USER, role)).fetchone()
        assert inherits is False
        assert can_set is True


def test_test_role_cannot_connect_to_development_database():
    url = require_test_database_url()
    development = make_conninfo(**dict(conninfo_to_dict(url), dbname="pharmaassist"))
    with pytest.raises(psycopg.OperationalError, match="CONNECT privilege"):
        psycopg.connect(development, connect_timeout=5)
