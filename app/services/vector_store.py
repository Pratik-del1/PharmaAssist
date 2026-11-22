import hashlib
import shutil
import stat
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import psycopg
from psycopg.types.json import Jsonb

from app.services.embedding_service import EMBEDDING_DIMENSION

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = PROJECT_ROOT / "data" / "archive"

# Chunk keys produced by chunk_document() (PDF) and chunk_docx_document() (DOCX),
# plus the keys added during ingestion. Any other key is rejected, so nothing is lost silently.
COMMON_CHUNK_KEYS = {"text", "chunk_index", "product_id", "file_name", "token_count", "embedding"}
LOCATION_KEYS = {
    "pdf": {"page_number"},
    "docx": {"section_heading", "heading_level", "paragraph_indices", "table_indices"},
}


def compute_file_hash(file_path: Union[str, Path]) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)
    return sha256.hexdigest()


def archive_file(file_path: Union[str, Path], file_hash: str, archive_dir: Union[str, Path] = ARCHIVE_DIR) -> Path:
    """Copy the original file to <archive_dir>/<sha256>.<ext>, read-only, and return the copy's path."""
    source = Path(file_path)
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived = archive_dir / f"{file_hash}{source.suffix.lower()}"

    if not archived.exists():
        shutil.copy2(source, archived)
        archived.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)  # read-only

    # The archived copy must be exactly the file that was hashed.
    if compute_file_hash(archived) != file_hash:
        raise RuntimeError(f"Archived copy {archived} does not match SHA-256 {file_hash}")
    return archived


def find_document_id(connection: psycopg.Connection, file_hash: str) -> Optional[int]:
    row = connection.execute("SELECT id FROM documents WHERE file_hash = %s", (file_hash,)).fetchone()
    return row[0] if row else None


def log_ingestion_run(
    connection: psycopg.Connection,
    file_name: str,
    status: str,
    started_at: datetime,
    file_hash: Optional[str] = None,
    document_id: Optional[int] = None,
    chunk_count: Optional[int] = None,
    embedding_model: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    # Its own transaction, so a log entry is kept even when the document transaction failed.
    with connection.transaction():
        connection.execute(
            """INSERT INTO ingestion_runs (file_name, file_hash, status, document_id, chunk_count,
                                          embedding_model, error_message, started_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (file_name, file_hash, status, document_id, chunk_count, embedding_model, error_message, started_at),
        )


def check_chunks(document_type: str, chunks: list[dict]) -> None:
    allowed = COMMON_CHUNK_KEYS | LOCATION_KEYS[document_type]
    for chunk in chunks:
        unknown = set(chunk) - allowed
        if unknown:
            raise ValueError(f"Chunk {chunk.get('chunk_index')} has unexpected keys: {sorted(unknown)}")
        if len(chunk["embedding"]) != EMBEDDING_DIMENSION:
            raise ValueError(f"Chunk {chunk['chunk_index']} embedding has {len(chunk['embedding'])} dimensions")


def save_document(
    connection: psycopg.Connection,
    document: dict,
    chunks: list[dict],
    embedding_model: str,
    started_at: datetime,
) -> int:
    """Store a document, its chunks and embeddings, and a 'succeeded' log entry in ONE transaction.

    If anything fails, nothing from this document is stored.
    """
    document_type = document["document_type"]
    check_chunks(document_type, chunks)

    with connection.transaction():
        row = connection.execute(
            """INSERT INTO documents (file_name, file_hash, file_size_bytes, storage_uri, document_type,
                                      source_type, source_url, retrieved_at, product_id, extraction_metadata)
               VALUES (%(file_name)s, %(file_hash)s, %(file_size_bytes)s, %(storage_uri)s, %(document_type)s,
                       %(source_type)s, %(source_url)s, %(retrieved_at)s, %(product_id)s, %(extraction_metadata)s)
               ON CONFLICT (file_hash) DO NOTHING
               RETURNING id""",
            {**document, "extraction_metadata": Jsonb(document["extraction_metadata"])},
        ).fetchone()
        if row is None:
            raise ValueError(f"A document with SHA-256 {document['file_hash']} was stored in the meantime")
        document_id = row[0]

        for chunk in chunks:
            chunk_id = connection.execute(
                """INSERT INTO document_chunks (document_id, document_type, chunk_index, content, token_count,
                                                page_number, paragraph_indices, table_indices,
                                                section_heading, heading_level)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   RETURNING id""",
                (
                    document_id, document_type, chunk["chunk_index"], chunk["text"], chunk["token_count"],
                    chunk.get("page_number"), chunk.get("paragraph_indices"), chunk.get("table_indices"),
                    chunk.get("section_heading"), chunk.get("heading_level"),
                ),
            ).fetchone()[0]
            connection.execute(
                "INSERT INTO chunk_embeddings (chunk_id, embedding_model, embedding) VALUES (%s, %s, %s)",
                (chunk_id, embedding_model, np.array(chunk["embedding"], dtype=np.float32)),
            )

        connection.execute(
            """INSERT INTO ingestion_runs (file_name, file_hash, status, document_id, chunk_count,
                                          embedding_model, started_at)
               VALUES (%s, %s, 'succeeded', %s, %s, %s, %s)""",
            (document["file_name"], document["file_hash"], document_id, len(chunks), embedding_model, started_at),
        )

    return document_id
