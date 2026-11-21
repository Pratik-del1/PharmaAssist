import re

# Invisible characters that PDFs sometimes leave in extracted text:
# zero-width space, byte order mark and soft hyphen.
INVISIBLE_CHARACTERS = ["​", "﻿", "­"]

# Space-like characters that should become a normal space:
# tab, non-breaking space and narrow non-breaking space.
SPACE_CHARACTERS = ["\t", " ", " "]


def clean_text(text: str) -> str:
    """Normalize whitespace and remove invisible characters.

    Letters, numbers, punctuation and symbols are never changed,
    so values like "2.5 mg", "±5%" and "37°C" stay exactly as they are.
    """
    # Use "\n" for every kind of line break (Windows, old Mac, form feed).
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")

    for character in SPACE_CHARACTERS:
        text = text.replace(character, " ")

    for character in INVISIBLE_CHARACTERS:
        text = text.replace(character, "")

    # Collapse repeated spaces into one.
    text = re.sub(r" {2,}", " ", text)

    # Remove spaces at the start and end of every line.
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Keep at most one blank line between paragraphs.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def split_into_sentences(text: str) -> list[str]:
    # Split after . ! or ? only when followed by whitespace,
    # so decimal values like "2.5 mg" are never split.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence for sentence in sentences if sentence]


def split_long_sentence(sentence: str, chunk_size: int, overlap: int) -> list[str]:
    # Fallback for a sentence longer than chunk_size: cut it by character count.
    pieces = []
    step = chunk_size - overlap  # always at least 1, checked in chunk_document()

    for start in range(0, len(sentence), step):
        pieces.append(sentence[start:start + chunk_size])
        if start + chunk_size >= len(sentence):
            break

    return pieces


def get_overlap_sentences(sentences: list[str], overlap: int) -> list[str]:
    # Take whole sentences from the end, as long as they fit within overlap characters.
    overlap_sentences = []

    for sentence in reversed(sentences):
        candidate = [sentence] + overlap_sentences
        if len(" ".join(candidate)) > overlap:
            break
        overlap_sentences = candidate

    return overlap_sentences


def chunk_page_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    current = []  # sentences in the chunk being built

    for sentence in split_into_sentences(text):
        if len(sentence) > chunk_size:
            if current:
                chunks.append(" ".join(current))
                current = []
            chunks.extend(split_long_sentence(sentence, chunk_size, overlap))
            continue

        if len(" ".join(current + [sentence])) <= chunk_size:
            current.append(sentence)
            continue

        # The chunk is full: save it and start a new one with some overlap.
        chunks.append(" ".join(current))
        current = get_overlap_sentences(current, overlap)

        # Drop the overlap if it would make the new chunk too long.
        if len(" ".join(current + [sentence])) > chunk_size:
            current = []

        current.append(sentence)

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_document(
    extracted: dict,
    product_id: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[dict]:
    """Split an extract_pdf_text() result into chunks with metadata.

    chunk_size and overlap are measured in characters, not tokens.
    Each page is chunked separately, so every chunk has one page_number.
    Empty pages produce no chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    # overlap must be smaller than chunk_size, otherwise chunking cannot move forward.
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []

    for page in extracted["pages"]:
        page_text = clean_text(page["text"])
        if not page_text:
            continue

        for text in chunk_page_text(page_text, chunk_size, overlap):
            chunks.append({
                "text": text,
                "page_number": page["page_number"],
                "chunk_index": len(chunks),
                "product_id": product_id,
                "file_name": extracted["file_name"],
            })

    return chunks


# ---------- DOCX chunking ----------

def table_to_lines(rows: list[list[str]]) -> list[str]:
    """Turn table rows into text lines, repeating the header labels on every row.

    [["Ingredient", "Strength"], ["Ingredient A", "100 mg"]]
    becomes ["Ingredient: Ingredient A | Strength: 100 mg"].
    The first row is treated as the header.
    """
    cleaned_rows = [[clean_text(cell) for cell in row] for row in rows]

    # A table with one row has no header, so just join its cells.
    if len(cleaned_rows) == 1:
        values = [cell for cell in cleaned_rows[0] if cell]
        return [" | ".join(values)] if values else []

    header = cleaned_rows[0]
    lines = []

    for row in cleaned_rows[1:]:
        parts = []
        for column, value in enumerate(row):
            if not value:
                continue
            label = header[column] if column < len(header) else ""
            if label:
                parts.append(f"{label}: {value}")
            else:
                parts.append(value)

        if parts:
            lines.append(" | ".join(parts))

    return lines


def make_docx_unit(section: dict, text: str, paragraph_index=None, table_index=None) -> dict:
    """Create one unit (a paragraph or a table row) and add it to the section."""
    unit = {
        "text": text,
        "paragraph_indices": [],
        "table_indices": [],
        "is_table_row": table_index is not None,
        "heading_text": None,
    }
    if paragraph_index is not None:
        unit["paragraph_indices"].append(paragraph_index)
    if table_index is not None:
        unit["table_indices"].append(table_index)

    # Attach the heading line to the first unit of its section,
    # so a heading never ends up alone in a chunk.
    if not section["units"] and section["heading"]:
        unit["text"] = section["heading"] + "\n" + unit["text"]
        unit["paragraph_indices"].insert(0, section["heading_paragraph_index"])
        unit["heading_text"] = section["heading"]

    section["units"].append(unit)
    return unit


def split_docx_into_sections(content: list[dict]) -> list[dict]:
    """Group content under the most recent explicit Word heading."""
    section = {"heading": None, "heading_level": None, "heading_paragraph_index": None, "units": []}
    sections = [section]

    for item in content:
        if item["type"] == "heading":
            section = {
                "heading": clean_text(item["text"]),
                "heading_level": item["heading_level"],
                "heading_paragraph_index": item["paragraph_index"],
                "units": [],
            }
            sections.append(section)

        elif item["type"] == "paragraph":
            text = clean_text(item["text"])
            if text:
                make_docx_unit(section, text, paragraph_index=item["paragraph_index"])

        elif item["type"] == "table":
            for line in table_to_lines(item["rows"]):
                make_docx_unit(section, line, table_index=item["table_index"])

    # A heading followed directly by another heading has no units and is skipped.
    return [section for section in sections if section["units"]]


def combine_units(units: list[dict]) -> dict:
    paragraph_indices = set()
    table_indices = set()
    for unit in units:
        paragraph_indices.update(unit["paragraph_indices"])
        table_indices.update(unit["table_indices"])

    return {
        "text": "\n".join(unit["text"] for unit in units),
        "paragraph_indices": sorted(paragraph_indices),
        "table_indices": sorted(table_indices),
    }


def get_overlap_units(units: list[dict], overlap: int) -> list[dict]:
    # get_overlap_sentences() only ever returns whole items from the end of the list,
    # so its length tells us how many whole units to carry over.
    texts = [unit["text"] for unit in units]
    count = len(get_overlap_sentences(texts, overlap))
    if count == 0:
        return []
    return units[-count:]


def split_long_paragraph_unit(unit: dict, chunk_size: int, overlap: int) -> list[dict]:
    pieces = []
    for text in chunk_page_text(unit["text"], chunk_size, overlap):
        paragraph_indices = list(unit["paragraph_indices"])
        # Only pieces that contain the heading line keep the heading's paragraph index.
        if unit["heading_text"] and not text.startswith(unit["heading_text"]):
            paragraph_indices = paragraph_indices[1:]
        pieces.append({
            "text": text,
            "paragraph_indices": paragraph_indices,
            "table_indices": list(unit["table_indices"]),
        })
    return pieces


def pack_docx_units(units: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    chunks = []
    current = []  # units in the chunk being built

    for unit in units:
        if len(unit["text"]) > chunk_size:
            if current:
                chunks.append(combine_units(current))
                current = []

            if unit["is_table_row"]:
                # Never split a table row: keep it whole, even if it is longer than chunk_size.
                chunks.append(combine_units([unit]))
            else:
                chunks.extend(split_long_paragraph_unit(unit, chunk_size, overlap))
            continue

        if len(combine_units(current + [unit])["text"]) <= chunk_size:
            current.append(unit)
            continue

        # The chunk is full: save it and start a new one with some overlap.
        chunks.append(combine_units(current))
        current = get_overlap_units(current, overlap)

        # Drop the overlap if it would make the new chunk too long.
        if len(combine_units(current + [unit])["text"]) > chunk_size:
            current = []

        current.append(unit)

    if current:
        chunks.append(combine_units(current))

    return chunks


def chunk_docx_document(
    extracted: dict,
    product_id: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[dict]:
    """Split an extract_docx_text() result into chunks with metadata.

    chunk_size and overlap are measured in characters, not tokens.
    Paragraphs and table rows are kept whole wherever possible.
    Only explicit Word heading styles start a new section.
    DOCX files have no reliable page numbers, so chunks have no page_number.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    # overlap must be smaller than chunk_size, otherwise chunking cannot move forward.
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []

    for section in split_docx_into_sections(extracted["content"]):
        for packed in pack_docx_units(section["units"], chunk_size, overlap):
            chunks.append({
                "text": packed["text"],
                "chunk_index": len(chunks),
                "product_id": product_id,
                "file_name": extracted["file_name"],
                "section_heading": section["heading"],
                "heading_level": section["heading_level"],
                "paragraph_indices": packed["paragraph_indices"],
                "table_indices": packed["table_indices"],
            })

    return chunks
