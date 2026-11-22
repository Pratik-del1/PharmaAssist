-- PharmaAssist database schema (PostgreSQL 18, pgvector 0.8). PDF and DOCX sources.
--
-- Run as the table owner:
--   nawinkmr           in pharmaassist       (psql pharmaassist -v ON_ERROR_STOP=1 -f app/database/schema.sql)
--   pharmaassist_test  in pharmaassist_test  (done automatically by the database tests)
-- The vector extension must already exist (CREATE EXTENSION vector; run once by nawinkmr).
-- Run app/database/permissions.sql afterwards.

BEGIN;

-- Product identity only. Medical facts come from approved document chunks, never from this table.
-- A product row is NOT evidence of any pharmaceutical claim: the chatbot must still abstain when
-- no approved document supports an answer, even for a listed product (for example REPOB).
CREATE TABLE products (
    product_id    TEXT PRIMARY KEY CHECK (product_id ~ '^[A-Z0-9][A-Z0-9-]*$'),
    product_name  TEXT NOT NULL CHECK (product_name <> ''),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per unique source file.
CREATE TABLE documents (
    id                   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    file_name            TEXT NOT NULL CHECK (file_name <> ''),
    file_hash            TEXT NOT NULL UNIQUE CHECK (file_hash ~ '^[0-9a-f]{64}$'),
    file_size_bytes      BIGINT NOT NULL CHECK (file_size_bytes > 0),
    storage_uri          TEXT NOT NULL UNIQUE CHECK (storage_uri ~ '^file:///'),
    document_type        TEXT NOT NULL CHECK (document_type IN ('pdf', 'docx')),
    source_type          TEXT NOT NULL CHECK (source_type IN
                             ('product_leaflet', 'product_information', 'training_manual', 'external_article')),
    source_url           TEXT CHECK (source_url ~ '^https://[A-Za-z0-9.-]+(:[0-9]+)?(/[^[:space:]]*)?$'),
    retrieved_at         TIMESTAMPTZ,
    product_id           TEXT REFERENCES products (product_id),
    extraction_metadata  JSONB NOT NULL DEFAULT '{}' CHECK (jsonb_typeof(extraction_metadata) = 'object'),
    ingested_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Review columns. Only the reviewer role may change these; the application role cannot set them.
    approval_status      TEXT NOT NULL DEFAULT 'pending'
                             CHECK (approval_status IN ('approved', 'pending', 'rejected')),
    reviewed_by          TEXT,
    reviewed_at          TIMESTAMPTZ,
    superseded_at        TIMESTAMPTZ,

    UNIQUE (id, document_type),
    CHECK ((source_url IS NULL) = (retrieved_at IS NULL)),
    CHECK (source_type <> 'external_article' OR source_url IS NOT NULL),
    CHECK (retrieved_at IS NULL OR retrieved_at <= ingested_at),
    CHECK ((approval_status = 'pending') = (reviewed_by IS NULL)),
    CHECK ((reviewed_by IS NULL) = (reviewed_at IS NULL)),
    CHECK (superseded_at IS NULL OR approval_status = 'approved')
);

COMMENT ON COLUMN documents.product_id IS
    'Set ONLY when the whole document is about this one product. Leave NULL for general or '
    'multi-product documents (for example a training manual with general sections), so that '
    'no chunk is implied to be about a product it does not discuss.';
COMMENT ON COLUMN documents.source_url IS
    'Original HTTPS address of an external source, kept exactly as given for citations. '
    'Required for external_article documents; NULL for internal company documents.';

-- Chunk text plus its exact location in the source file.
CREATE TABLE document_chunks (
    id                 BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id        BIGINT NOT NULL,
    document_type      TEXT NOT NULL CHECK (document_type IN ('pdf', 'docx')),
    chunk_index        INTEGER NOT NULL CHECK (chunk_index >= 0),
    content            TEXT NOT NULL CHECK (content <> ''),
    token_count        INTEGER NOT NULL CHECK (token_count BETWEEN 1 AND 512),
    page_number        INTEGER CHECK (page_number >= 1),           -- PDF
    paragraph_indices  INTEGER[],                                   -- DOCX
    table_indices      INTEGER[],                                   -- DOCX
    section_heading    TEXT,                                        -- DOCX, explicit Word headings only
    heading_level      SMALLINT CHECK (heading_level BETWEEN 0 AND 9),
    UNIQUE (document_id, chunk_index),
    FOREIGN KEY (document_id, document_type)
        REFERENCES documents (id, document_type) ON DELETE CASCADE,
    CHECK (
        (document_type = 'pdf'
            AND page_number IS NOT NULL
            AND paragraph_indices IS NULL AND table_indices IS NULL
            AND section_heading IS NULL AND heading_level IS NULL)
     OR (document_type = 'docx'
            AND page_number IS NULL
            AND paragraph_indices IS NOT NULL AND table_indices IS NOT NULL)
    ),
    CHECK (heading_level IS NULL OR section_heading IS NOT NULL)
);

-- One normalized 384-dimensional vector per chunk per embedding model.
CREATE TABLE chunk_embeddings (
    chunk_id         BIGINT NOT NULL REFERENCES document_chunks (id) ON DELETE CASCADE,
    embedding_model  TEXT NOT NULL CHECK (embedding_model <> ''),
    embedding        vector(384) NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (chunk_id, embedding_model),
    CHECK (abs(vector_norm(embedding) - 1) < 0.001)
);

-- Append-only log of every ingestion attempt, including failures and duplicates.
-- file_hash is NULL only for failures that happened before the file could be hashed
-- (for example a missing or unreadable file).
CREATE TABLE ingestion_runs (
    id               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    file_name        TEXT NOT NULL CHECK (file_name <> ''),
    file_hash        TEXT CHECK (file_hash ~ '^[0-9a-f]{64}$'),
    status           TEXT NOT NULL CHECK (status IN ('succeeded', 'skipped_duplicate', 'failed')),
    document_id      BIGINT REFERENCES documents (id) ON DELETE SET NULL,
    chunk_count      INTEGER CHECK (chunk_count >= 0),
    embedding_model  TEXT,
    error_message    TEXT,
    started_at       TIMESTAMPTZ NOT NULL,
    finished_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK ((status = 'failed') = (error_message IS NOT NULL)),
    CHECK (status = 'failed' OR file_hash IS NOT NULL),
    CHECK (finished_at >= started_at)
);

-- The only way the application can read chunk text and embeddings.
--   security_invoker = false: table access is checked as the view OWNER. permissions.sql gives
--                             ownership to pharmaassist_view_owner, a non-login role that can
--                             read only the columns this view uses.
--   security_barrier = true:  the caller's own non-leakproof WHERE conditions run only after the
--                             approval filter. It does not hide row counts; see README.
CREATE VIEW production_chunk_embeddings
    WITH (security_barrier = true, security_invoker = false) AS
SELECT e.chunk_id,
       e.embedding_model,
       e.embedding,
       c.document_id,
       c.chunk_index,
       c.content,
       c.page_number,
       c.paragraph_indices,
       c.table_indices,
       c.section_heading,
       c.heading_level,
       d.file_name,
       d.document_type,
       d.source_type,
       d.source_url,
       d.retrieved_at,
       d.product_id,
       p.product_name
FROM chunk_embeddings e
JOIN document_chunks c ON c.id = e.chunk_id
JOIN documents d       ON d.id = c.document_id
LEFT JOIN products p   ON p.product_id = d.product_id
WHERE d.approval_status = 'approved'
  AND d.superseded_at IS NULL;

COMMENT ON VIEW production_chunk_embeddings IS
    'Approved, non-superseded chunks with their embeddings. The only source for production retrieval. '
    'product_id / product_name are only set for single-product documents (see documents.product_id).';

CREATE INDEX documents_product_id_idx       ON documents (product_id);
CREATE INDEX ingestion_runs_file_hash_idx   ON ingestion_runs (file_hash);
CREATE INDEX ingestion_runs_document_id_idx ON ingestion_runs (document_id);

COMMIT;
