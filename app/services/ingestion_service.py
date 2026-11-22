from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import docx
import psycopg
import pymupdf

from app.services.docx_extractor import extract_docx_text
from app.services.embedding_service import MODEL_NAME, count_tokens, embed_chunks
from app.services.pdf_extractor import extract_pdf_text
from app.services.text_processing import chunk_docx_document, chunk_document, clean_text
from app.services.vector_store import (
    ARCHIVE_DIR,
    archive_file,
    compute_file_hash,
    find_document_id,
    log_ingestion_run,
    save_document,
)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def extract_and_chunk(path: Path, product_id: Optional[str]) -> tuple:
    """Run the existing extractor and chunker for the file type. Returns (document_type, metadata, chunks)."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        extracted = extract_pdf_text(path)
        chunks = chunk_document(extracted, product_id, CHUNK_SIZE, CHUNK_OVERLAP)
        metadata = {
            "page_count": extracted["page_count"],
            "empty_pages": [page["page_number"] for page in extracted["pages"] if not clean_text(page["text"])],
            "extractor": f"pymupdf {pymupdf.VersionBind}",
        }
        return "pdf", metadata, chunks

    if suffix == ".docx":
        extracted = extract_docx_text(path)
        chunks = chunk_docx_document(extracted, product_id, CHUNK_SIZE, CHUNK_OVERLAP)
        types = [item["type"] for item in extracted["content"]]
        metadata = {
            "paragraph_count": types.count("paragraph"),
            "heading_count": types.count("heading"),
            "table_count": types.count("table"),
            "extractor": f"python-docx {docx.__version__}",
        }
        return "docx", metadata, chunks

    raise ValueError(f"Unsupported file type: {path.suffix} (only .pdf and .docx)")


def ingest_file(
    connection: psycopg.Connection,
    file_path: Union[str, Path],
    source_type: str,
    product_id: Optional[str] = None,
    source_url: Optional[str] = None,
    retrieved_at: Optional[datetime] = None,
    archive_dir: Union[str, Path] = ARCHIVE_DIR,
) -> dict:
    """Ingest one PDF or DOCX file as a PENDING document.

    1. Hash the file and stop if the same content is already stored (logged as skipped_duplicate).
    2. Extract, chunk, count tokens and embed with the existing services.
    3. Archive the original file under its SHA-256.
    4. Store the document, chunks and embeddings in one transaction.
    Any failure is logged as 'failed' in a separate transaction and then re-raised.
    """
    path = Path(file_path)
    # Use the database clock, so started_at and the log's finished_at (DEFAULT now()) come from the
    # same clock and finished_at >= started_at always holds.
    started_at = connection.execute("SELECT now()").fetchone()[0]
    file_hash = None

    try:
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        file_hash = compute_file_hash(path)

        existing_id = find_document_id(connection, file_hash)
        if existing_id is not None:
            log_ingestion_run(connection, path.name, "skipped_duplicate", started_at,
                              file_hash=file_hash, document_id=existing_id)
            return {"status": "skipped_duplicate", "document_id": existing_id, "file_hash": file_hash}

        document_type, metadata, chunks = extract_and_chunk(path, product_id)
        for chunk in chunks:
            chunk["token_count"] = count_tokens(chunk["text"])
        embedded_chunks = embed_chunks(chunks)  # raises before embedding anything if a chunk is too long

        archived = archive_file(path, file_hash, archive_dir)
        metadata.update({"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP, "embedding_model": MODEL_NAME})
        document = {
            "file_name": path.name,
            "file_hash": file_hash,
            "file_size_bytes": path.stat().st_size,
            "storage_uri": archived.resolve().as_uri(),
            "document_type": document_type,
            "source_type": source_type,
            "source_url": source_url,
            "retrieved_at": retrieved_at,
            "product_id": product_id,
            "extraction_metadata": metadata,
        }
        document_id = save_document(connection, document, embedded_chunks, MODEL_NAME, started_at)
        return {
            "status": "succeeded",
            "document_id": document_id,
            "file_hash": file_hash,
            "chunk_count": len(embedded_chunks),
            "storage_uri": document["storage_uri"],
        }

    except Exception as error:
        message = f"{type(error).__name__}: {error}"[:2000]
        log_ingestion_run(connection, path.name, "failed", started_at, file_hash=file_hash, error_message=message)
        raise
