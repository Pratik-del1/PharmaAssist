"""Ingest one approved company PDF or DOCX file as a PENDING document.

Uses the application database role (DATABASE_URL), never the reviewer role.

Example:
    python -m scripts.ingest_documents --file data/raw/CZARLIV_SYRUP.pdf \
        --source-type product_leaflet --product-id CZARLIV
"""
import argparse
import sys

from app.database.connection import get_connection, require_current_user
from app.services.ingestion_service import ingest_file

APPLICATION_ROLE = "pharmaassist_app"
SOURCE_TYPES = ["product_leaflet", "product_information", "training_manual", "external_article"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest one PDF or DOCX file as a pending document.")
    parser.add_argument("--file", required=True, help="Path to the .pdf or .docx file")
    parser.add_argument("--source-type", required=True, choices=SOURCE_TYPES)
    parser.add_argument("--product-id", default=None,
                        help="Only if the WHOLE document is about this one product")
    args = parser.parse_args()

    connection = get_connection("DATABASE_URL")
    connection.autocommit = True  # each transaction in the ingestion code commits on its own
    try:
        require_current_user(connection, APPLICATION_ROLE)
        result = ingest_file(connection, args.file, args.source_type, product_id=args.product_id)
    except Exception as error:
        print(f"Ingestion failed (logged in ingestion_runs): {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    finally:
        connection.close()

    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
