from typing import Optional

import psycopg

DOCUMENT_COLUMNS = """id, file_name, document_type, source_type, product_id, approval_status,
                      reviewed_by, reviewed_at, superseded_at, ingested_at, file_hash,
                      file_size_bytes, storage_uri, source_url, retrieved_at, extraction_metadata"""


def rows_to_dicts(cursor) -> list[dict]:
    names = [column.name for column in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def list_documents(connection: psycopg.Connection, status: Optional[str] = "pending") -> list[dict]:
    if status is None:
        cursor = connection.execute(f"SELECT {DOCUMENT_COLUMNS} FROM documents ORDER BY id")
    else:
        cursor = connection.execute(
            f"SELECT {DOCUMENT_COLUMNS} FROM documents WHERE approval_status = %s ORDER BY id", (status,))
    return rows_to_dicts(cursor)


def get_document(connection: psycopg.Connection, document_id: int) -> Optional[dict]:
    rows = rows_to_dicts(connection.execute(f"SELECT {DOCUMENT_COLUMNS} FROM documents WHERE id = %s", (document_id,)))
    return rows[0] if rows else None


def get_chunks(connection: psycopg.Connection, document_id: int) -> list[dict]:
    return rows_to_dicts(connection.execute(
        """SELECT chunk_index, page_number, paragraph_indices, table_indices, section_heading,
                  heading_level, token_count, content
           FROM document_chunks WHERE document_id = %s ORDER BY chunk_index""", (document_id,)))


def set_review_status(connection: psycopg.Connection, document_id: int, status: str, reviewer: str) -> bool:
    """Approve or reject a PENDING document. Returns False if no pending document had that id."""
    if status not in ("approved", "rejected"):
        raise ValueError("status must be 'approved' or 'rejected'")
    if not reviewer.strip():
        raise ValueError("reviewer name is required")
    with connection.transaction():
        row = connection.execute(
            """UPDATE documents SET approval_status = %s, reviewed_by = %s, reviewed_at = now()
               WHERE id = %s AND approval_status = 'pending' RETURNING id""",
            (status, reviewer.strip(), document_id)).fetchone()
    return row is not None


def supersede(connection: psycopg.Connection, document_id: int) -> bool:
    """Retire an approved, current document. Returns False if there was no such document."""
    with connection.transaction():
        row = connection.execute(
            """UPDATE documents SET superseded_at = now()
               WHERE id = %s AND approval_status = 'approved' AND superseded_at IS NULL RETURNING id""",
            (document_id,)).fetchone()
    return row is not None
