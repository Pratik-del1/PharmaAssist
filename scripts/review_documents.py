"""Review ingested documents. Uses the reviewer database role (REVIEWER_DATABASE_URL).

Examples:
    python -m scripts.review_documents list
    python -m scripts.review_documents list --all
    python -m scripts.review_documents show 1
    python -m scripts.review_documents approve 1 --reviewer "Full Name"
    python -m scripts.review_documents reject 1 --reviewer "Full Name"
    python -m scripts.review_documents supersede 1
"""
import argparse
import sys

from app.database.connection import get_connection, require_current_user
from app.services.review_service import get_chunks, get_document, list_documents, set_review_status, supersede

REVIEWER_ROLE = "pharmaassist_reviewer"


def print_document(document: dict) -> None:
    for key, value in document.items():
        print(f"  {key}: {value}")


def print_chunks(chunks: list[dict]) -> None:
    for chunk in chunks:
        if chunk["page_number"] is not None:
            location = f"page {chunk['page_number']}"
        else:
            location = f"paragraphs {chunk['paragraph_indices']} tables {chunk['table_indices']}"
            if chunk["section_heading"]:
                location += f" heading '{chunk['section_heading']}'"
        print(f"\n--- chunk {chunk['chunk_index']} | {location} | {chunk['token_count']} tokens")
        print(chunk["content"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Review ingested documents.")
    commands = parser.add_subparsers(dest="command", required=True)

    list_command = commands.add_parser("list", help="List pending documents")
    list_command.add_argument("--all", action="store_true", help="List documents of every status")

    show_command = commands.add_parser("show", help="Show a document and all of its chunks")
    show_command.add_argument("document_id", type=int)

    for name in ("approve", "reject"):
        command = commands.add_parser(name, help=f"{name.capitalize()} a pending document")
        command.add_argument("document_id", type=int)
        command.add_argument("--reviewer", required=True, help="Name of the person making the decision")

    supersede_command = commands.add_parser("supersede", help="Retire an approved document")
    supersede_command.add_argument("document_id", type=int)

    args = parser.parse_args()

    connection = get_connection("REVIEWER_DATABASE_URL")
    connection.autocommit = True
    try:
        require_current_user(connection, REVIEWER_ROLE)

        if args.command == "list":
            documents = list_documents(connection, status=None if args.all else "pending")
            print(f"{len(documents)} document(s)")
            for document in documents:
                print(f"  #{document['id']}  {document['approval_status']:8}  {document['document_type']}  "
                      f"{document['source_type']:20}  product={document['product_id']}  {document['file_name']}")

        elif args.command == "show":
            document = get_document(connection, args.document_id)
            if document is None:
                print(f"No document with id {args.document_id}", file=sys.stderr)
                return 1
            print_document(document)
            print_chunks(get_chunks(connection, args.document_id))

        elif args.command in ("approve", "reject"):
            status = "approved" if args.command == "approve" else "rejected"
            if not set_review_status(connection, args.document_id, status, args.reviewer):
                print(f"Document {args.document_id} is not pending (or does not exist)", file=sys.stderr)
                return 1
            print(f"Document {args.document_id} {status} by {args.reviewer}")

        elif args.command == "supersede":
            if not supersede(connection, args.document_id):
                print(f"Document {args.document_id} is not an approved, current document", file=sys.stderr)
                return 1
            print(f"Document {args.document_id} superseded")
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
