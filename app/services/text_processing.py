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
