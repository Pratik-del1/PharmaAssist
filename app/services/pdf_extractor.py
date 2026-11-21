from pathlib import Path
from typing import Union

import pymupdf


def extract_pdf_text(pdf_path: Union[str, Path]) -> dict:
    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    try:
        document = pymupdf.open(path)
    except pymupdf.FileDataError:
        raise ValueError(f"{path.name} is not a valid PDF")

    with document:
        # PyMuPDF can also open other formats (like .txt), so confirm it is a PDF.
        if not document.is_pdf:
            raise ValueError(f"{path.name} is not a PDF document")

        if document.needs_pass:
            raise ValueError(f"{path.name} is password-protected")

        pages = []
        for page_index, page in enumerate(document):
            pages.append({
                "page_number": page_index + 1,
                "text": page.get_text().strip(),
            })

    has_text = any(page["text"] for page in pages)
    if not has_text:
        raise ValueError(
            f"No extractable text found in {path.name}. "
            "It may be a scanned PDF; OCR is not supported yet."
        )

    return {
        "file_name": path.name,
        "page_count": len(pages),
        "pages": pages,
    }
