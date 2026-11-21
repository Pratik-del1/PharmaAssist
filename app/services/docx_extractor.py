from pathlib import Path
from typing import Optional, Union

from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from docx.table import Table
from docx.text.paragraph import Paragraph


def get_heading_level(paragraph: Paragraph) -> Optional[int]:
    # Word's built-in heading styles are "Title" and "Heading 1" to "Heading 9".
    style_name = paragraph.style.name if paragraph.style else ""

    if style_name == "Title":
        return 0

    if style_name.startswith("Heading "):
        level = style_name[len("Heading "):]
        if level.isdigit():
            return int(level)

    return None


def extract_docx_text(docx_path: Union[str, Path]) -> dict:
    path = Path(docx_path)

    if not path.is_file():
        raise FileNotFoundError(f"DOCX file not found: {path}")

    try:
        document = Document(str(path))
    except (PackageNotFoundError, KeyError, ValueError):
        raise ValueError(f"{path.name} is not a valid DOCX file")

    content = []
    paragraph_index = 0
    table_index = 0

    # iter_inner_content() returns paragraphs and tables in document order.
    for item in document.iter_inner_content():
        if isinstance(item, Paragraph):
            text = item.text.strip()
            if text:
                heading_level = get_heading_level(item)
                if heading_level is None:
                    content.append({
                        "type": "paragraph",
                        "text": text,
                        "paragraph_index": paragraph_index,
                    })
                else:
                    content.append({
                        "type": "heading",
                        "text": text,
                        "heading_level": heading_level,
                        "paragraph_index": paragraph_index,
                    })
            # Empty paragraphs still count, so the index matches the document position.
            paragraph_index += 1

        elif isinstance(item, Table):
            rows = []
            for row in item.rows:
                rows.append([cell.text.strip() for cell in row.cells])

            has_text = any(cell for row in rows for cell in row)
            if has_text:
                content.append({
                    "type": "table",
                    "table_index": table_index,
                    "rows": rows,
                })
            table_index += 1

    if not content:
        raise ValueError(f"No extractable text found in {path.name}")

    return {
        "file_name": path.name,
        "document_type": "docx",
        "content": content,
    }
