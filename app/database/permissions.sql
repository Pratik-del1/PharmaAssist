-- Exact privileges for the PharmaAssist roles on the PharmaAssist tables.
--
-- Run as the table owner, after schema.sql.
--   psql pharmaassist -v ON_ERROR_STOP=1 -f app/database/permissions.sql
-- The database tests run this file in pharmaassist_test with
--   pharmaassist_app        -> pharmaassist_test_app
--   pharmaassist_reviewer   -> pharmaassist_test_reviewer
--   pharmaassist_view_owner -> pharmaassist_test_view_owner
--
-- ===========================================================================================
-- ONE-TIME ADMIN SETUP (run manually by nawinkmr, NOT part of this script)
-- ===========================================================================================
-- 1. In pharmaassist, BEFORE schema.sql: remove the schema-level default privileges that would
--    give the application role SELECT/INSERT on every new table and view.
--      ALTER DEFAULT PRIVILEGES FOR ROLE nawinkmr IN SCHEMA public
--          REVOKE SELECT, INSERT ON TABLES FROM pharmaassist_app;
--      ALTER DEFAULT PRIVILEGES FOR ROLE nawinkmr IN SCHEMA public
--          REVOKE USAGE ON SEQUENCES FROM pharmaassist_app;
--    Check (scoped to nawinkmr in schema public; expected 0 rows):
--      SELECT defaclobjtype, defaclacl FROM pg_default_acl
--      WHERE defaclrole = 'nawinkmr'::regrole AND defaclnamespace = 'public'::regnamespace;
--
-- 2. Reviewer login role (password stored only as a SCRAM hash in PostgreSQL and in .env):
--      CREATE ROLE pharmaassist_reviewer LOGIN PASSWORD '<scram-verifier>'
--          NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
--      GRANT CONNECT ON DATABASE pharmaassist TO pharmaassist_reviewer;
--      (in pharmaassist) GRANT USAGE ON SCHEMA public TO pharmaassist_reviewer;
--
-- 3. View owner: a non-login role that owns production_chunk_embeddings and can read only the
--    columns the view uses. Nobody can log in as it.
--      CREATE ROLE pharmaassist_view_owner NOLOGIN
--          NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
--
-- 4. Non-login stand-ins used only inside pharmaassist_test by the permission tests.
--    INHERIT FALSE: the test role never gets their privileges automatically, only via SET ROLE.
--      CREATE ROLE pharmaassist_test_app        NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
--      CREATE ROLE pharmaassist_test_reviewer   NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
--      CREATE ROLE pharmaassist_test_view_owner NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
--      GRANT pharmaassist_test_app        TO pharmaassist_test WITH INHERIT FALSE, SET TRUE;
--      GRANT pharmaassist_test_reviewer   TO pharmaassist_test WITH INHERIT FALSE, SET TRUE;
--      GRANT pharmaassist_test_view_owner TO pharmaassist_test WITH INHERIT FALSE, SET TRUE;
--      (in pharmaassist_test) GRANT CREATE ON SCHEMA public TO pharmaassist_test_view_owner;
--      -- PostgreSQL requires the new owner to have CREATE on the schema when a non-superuser
--      -- changes ownership. Only needed in the test database; nawinkmr is a superuser.
-- ===========================================================================================

BEGIN;

-- Start from nothing. Revoking a table privilege also removes the matching column privileges.
REVOKE ALL ON products, documents, document_chunks, chunk_embeddings, ingestion_runs
       FROM PUBLIC, pharmaassist_app, pharmaassist_reviewer, pharmaassist_view_owner;
REVOKE ALL ON production_chunk_embeddings
       FROM PUBLIC, pharmaassist_app, pharmaassist_reviewer;

-- ------------------------------------------------------------------ view owner
-- Exactly the columns production_chunk_embeddings reads (including the ones in its JOIN and
-- WHERE clauses). Nothing else: no storage_uri, file_hash, review names or ingestion log.
GRANT SELECT (chunk_id, embedding_model, embedding) ON chunk_embeddings TO pharmaassist_view_owner;
GRANT SELECT (id, document_id, chunk_index, content, page_number, paragraph_indices,
              table_indices, section_heading, heading_level)
      ON document_chunks TO pharmaassist_view_owner;
GRANT SELECT (id, file_name, document_type, source_type, source_url, retrieved_at, product_id,
              approval_status, superseded_at)
      ON documents TO pharmaassist_view_owner;
GRANT SELECT (product_id, product_name) ON products TO pharmaassist_view_owner;

-- ------------------------------------------------------------------ application role
-- Products: read only. Only nawinkmr adds products.
GRANT SELECT ON products TO pharmaassist_app;

-- Documents: insert new files as pending. The review columns and ingested_at are deliberately
-- missing, so the database always fills them with their defaults.
GRANT INSERT (file_name, file_hash, file_size_bytes, storage_uri, document_type, source_type,
              source_url, retrieved_at, product_id, extraction_metadata)
      ON documents TO pharmaassist_app;
-- Only enough to detect duplicates (also needed by ON CONFLICT (file_hash)) and read back the new id.
GRANT SELECT (id, file_hash) ON documents TO pharmaassist_app;

-- Chunks: insert, and read back only the new id.
GRANT INSERT (document_id, document_type, chunk_index, content, token_count, page_number,
              paragraph_indices, table_indices, section_heading, heading_level)
      ON document_chunks TO pharmaassist_app;
GRANT SELECT (id) ON document_chunks TO pharmaassist_app;

-- Embeddings: insert only (no ON CONFLICT; the primary key rejects duplicates).
GRANT INSERT (chunk_id, embedding_model, embedding) ON chunk_embeddings TO pharmaassist_app;

-- Ingestion log: insert only.
GRANT INSERT (file_name, file_hash, status, document_id, chunk_count, embedding_model,
              error_message, started_at)
      ON ingestion_runs TO pharmaassist_app;

-- Production retrieval: approved, non-superseded content only.
GRANT SELECT ON production_chunk_embeddings TO pharmaassist_app;

-- ------------------------------------------------------------------ reviewer role
-- Limitation: the database requires reviewed_by and reviewed_at for every review decision, but
-- it does not authenticate the human named in reviewed_by. Keep the reviewer credentials
-- (REVIEWER_DATABASE_URL) private to the people allowed to approve content.
-- Read everything needed to review pending content (but not the raw vectors).
GRANT SELECT ON products, documents, document_chunks, ingestion_runs, production_chunk_embeddings
      TO pharmaassist_reviewer;
-- Change ONLY the review columns. No INSERT, DELETE, or edits to content or source details.
GRANT UPDATE (approval_status, reviewed_by, reviewed_at, superseded_at)
      ON documents TO pharmaassist_reviewer;

-- ------------------------------------------------------------------ view ownership (last)
-- From now on the view reads the tables with pharmaassist_view_owner's limited privileges.
-- Existing grants on the view are kept; PostgreSQL records the new owner as their grantor.
ALTER VIEW production_chunk_embeddings OWNER TO pharmaassist_view_owner;

COMMIT;
