"""Semantic retrieval of approved document chunks.

Searches ONLY the production_chunk_embeddings view (approved, non-superseded documents), using the
application database role and exact cosine similarity. The similarity threshold is optional and
off by default: no value has been validated yet; it will be chosen through evaluation.
"""
import os
from typing import Optional

import numpy as np
import psycopg
from pydantic import BaseModel

from app.database.connection import get_connection, require_current_user
from app.services.embedding_service import MODEL_NAME, embed_query

APPLICATION_ROLE = "pharmaassist_app"
DEFAULT_TOP_K = 5
MAX_TOP_K = 20

# Reads only the production view. Every value is a query parameter.
SEARCH_SQL = """
    SELECT chunk_id, document_id, file_name, document_type, source_type, source_url,
           page_number, paragraph_indices, table_indices, section_heading,
           content, product_name,
           1 - (embedding <=> %(query)s) AS similarity
    FROM production_chunk_embeddings
    WHERE embedding_model = %(model)s
    ORDER BY embedding <=> %(query)s
    LIMIT %(top_k)s
"""


class RetrievalError(RuntimeError):
    """Raised when the knowledge base cannot be searched (for example the database is unavailable)."""


class RetrievedChunk(BaseModel):
    chunk_id: int
    document_id: int
    file_name: str
    document_type: str
    source_type: str
    source_url: Optional[str] = None
    page_number: Optional[int] = None
    paragraph_indices: Optional[list[int]] = None
    table_indices: Optional[list[int]] = None
    section_heading: Optional[str] = None
    content: str
    product_name: Optional[str] = None
    similarity: float

    def location(self) -> str:
        """A readable source location: a PDF page, or DOCX paragraph and table positions."""
        if self.page_number is not None:
            return f"page {self.page_number}"
        parts = [f"paragraphs {self.paragraph_indices}"]
        if self.table_indices:
            parts.append(f"tables {self.table_indices}")
        if self.section_heading:
            parts.append(f"section '{self.section_heading}'")
        return ", ".join(parts)


def get_retrieval_settings() -> tuple:
    """Read RETRIEVAL_TOP_K and RETRIEVAL_MIN_SIMILARITY from the environment.

    RETRIEVAL_MIN_SIMILARITY empty or unset means no threshold.
    """
    top_k = int(os.getenv("RETRIEVAL_TOP_K") or DEFAULT_TOP_K)
    threshold_text = (os.getenv("RETRIEVAL_MIN_SIMILARITY") or "").strip()
    min_similarity = float(threshold_text) if threshold_text else None
    check_settings(top_k, min_similarity)
    return top_k, min_similarity


def check_settings(top_k: int, min_similarity: Optional[float]) -> None:
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")
    if min_similarity is not None and not -1.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be between -1 and 1")


def open_retrieval_connection() -> psycopg.Connection:
    """Connect with the application role. Retrieval never uses the reviewer credentials."""
    connection = get_connection("DATABASE_URL")
    connection.autocommit = True
    try:
        require_current_user(connection, APPLICATION_ROLE)
    except Exception:
        connection.close()
        raise
    return connection


def search_chunks(
    connection: psycopg.Connection,
    question: str,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: Optional[float] = None,
) -> list[RetrievedChunk]:
    """Return the approved chunks most similar to the question, best first.

    Returns an empty list when no approved content exists or nothing reaches min_similarity.
    A high similarity only means the text is about a similar topic; it is not evidence that
    any medical statement is true.
    """
    question = question.strip()
    if not question:
        raise ValueError("question cannot be empty")
    check_settings(top_k, min_similarity)

    query_vector = np.array(embed_query(question), dtype=np.float32)

    try:
        cursor = connection.execute(SEARCH_SQL, {"query": query_vector, "model": MODEL_NAME, "top_k": top_k})
        names = [column.name for column in cursor.description]
        rows = cursor.fetchall()
    except psycopg.Error as error:
        raise RetrievalError(f"Knowledge base search failed: {type(error).__name__}") from error

    chunks = [RetrievedChunk(**dict(zip(names, row))) for row in rows]
    if min_similarity is not None:
        chunks = [chunk for chunk in chunks if chunk.similarity >= min_similarity]
    return chunks
