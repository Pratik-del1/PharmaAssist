import re

import pytest
from docx import Document

from app.services.docx_extractor import extract_docx_text
from app.services.text_processing import (
    chunk_document,
    chunk_docx_document,
    clean_text,
    table_to_lines,
)


def make_document(page_texts):
    """Build a dictionary shaped like the output of extract_pdf_text()."""
    pages = []
    for index, text in enumerate(page_texts):
        pages.append({"page_number": index + 1, "text": text})

    return {
        "file_name": "demo_leaflet.pdf",
        "page_count": len(pages),
        "pages": pages,
    }


# ---------- clean_text() ----------

def test_clean_text_normalizes_whitespace():
    raw_text = "  DemoCompound   Alpha\t\tTablets  \r\n\n\n\n\nStorage conditions  "

    assert clean_text(raw_text) == "DemoCompound Alpha Tablets\n\nStorage conditions"


def test_clean_text_removes_invisible_characters():
    raw_text = "Demo​Compound﻿ Alpha"

    assert clean_text(raw_text) == "DemoCompound Alpha"


def test_clean_text_keeps_pharmaceutical_values():
    raw_text = "Amoxicillin  2.5 mg/5 ml, 0.5 ml, 0.25 mcg, ±5%, 37°C."

    cleaned = clean_text(raw_text)

    assert cleaned == "Amoxicillin 2.5 mg/5 ml, 0.5 ml, 0.25 mcg, ±5%, 37°C."
    for value in ["2.5 mg", "0.5 ml", "0.25 mcg", "±5%", "37°C"]:
        assert value in cleaned


def test_clean_text_keeps_section_headings_on_their_own_line():
    raw_text = "DOSAGE\nTake 2.5 mg daily.\n\nSTORAGE\nStore below 25°C."

    assert clean_text(raw_text) == raw_text


def test_clean_text_with_only_whitespace_returns_empty_string():
    assert clean_text("   \n\n\t   ") == ""


# ---------- chunk_document() ----------

def test_single_page_short_text_gives_one_chunk():
    document = make_document(["DemoCompound Alpha is a fictional tablet."])

    chunks = chunk_document(document, product_id="DEMO-001")

    assert len(chunks) == 1
    assert chunks[0]["text"] == "DemoCompound Alpha is a fictional tablet."
    assert chunks[0]["page_number"] == 1


def test_multi_page_chunks_keep_correct_page_numbers():
    document = make_document(["Text on page one.", "", "Text on page three."])

    chunks = chunk_document(document, product_id="DEMO-001")

    # The empty page 2 produces no chunk, but page 3 keeps its number.
    assert len(chunks) == 2
    assert chunks[0]["page_number"] == 1
    assert chunks[1]["page_number"] == 3
    assert chunks[1]["text"] == "Text on page three."


def test_chunks_overlap_with_previous_chunk():
    text = (
        "Store below 25°C. "
        "Keep away from light. "
        "Do not freeze. "
        "Shake well before use."
    )
    document = make_document([text])

    chunks = chunk_document(document, product_id="DEMO-001", chunk_size=50, overlap=25)

    assert [chunk["text"] for chunk in chunks] == [
        "Store below 25°C. Keep away from light.",
        "Keep away from light. Do not freeze.",
        "Do not freeze. Shake well before use.",
    ]


def test_chunks_are_not_longer_than_chunk_size():
    sentence = "DemoCompound Alpha contains 2.5 mg of fictional ingredient A. "
    document = make_document([sentence * 40])

    chunks = chunk_document(document, product_id="DEMO-001", chunk_size=200, overlap=50)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk["text"]) <= 200


def test_values_stay_intact_after_chunking():
    text = (
        "Amoxicillin oral suspension contains 2.5 mg/5 ml. "
        "The dose is 0.5 ml twice daily. "
        "Each drop contains 0.25 mcg. "
        "Tolerance is ±5%. "
        "Do not store above 37°C."
    )
    document = make_document([text])

    chunks = chunk_document(document, product_id="DEMO-001", chunk_size=60, overlap=20)
    all_chunk_text = " ".join(chunk["text"] for chunk in chunks)

    for value in ["2.5 mg/5 ml", "0.5 ml", "0.25 mcg", "±5%", "37°C"]:
        assert value in all_chunk_text


def test_long_sentence_is_split_with_fallback():
    long_sentence = "x" * 250  # one "sentence" with no full stops
    document = make_document([long_sentence])

    chunks = chunk_document(document, product_id="DEMO-001", chunk_size=100, overlap=20)

    assert len(chunks) == 3
    for chunk in chunks:
        assert len(chunk["text"]) <= 100
    # Each piece starts with the last 20 characters of the previous piece.
    assert chunks[1]["text"][:20] == chunks[0]["text"][-20:]


def test_chunks_keep_metadata():
    document = make_document(["First sentence here. Second sentence here.", "Page two text."])

    chunks = chunk_document(document, product_id="DEMO-002", chunk_size=25, overlap=5)

    assert [chunk["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert chunk["product_id"] == "DEMO-002"
        assert chunk["file_name"] == "demo_leaflet.pdf"
        assert set(chunk.keys()) == {
            "text", "page_number", "chunk_index", "product_id", "file_name"
        }


def test_document_with_only_empty_text_returns_no_chunks():
    document = make_document(["", "   \n\n  "])

    assert chunk_document(document, product_id="DEMO-001") == []


def test_overlap_must_be_smaller_than_chunk_size():
    document = make_document(["Some text."])

    with pytest.raises(ValueError, match="overlap must be smaller"):
        chunk_document(document, product_id="DEMO-001", chunk_size=100, overlap=100)


# ---------- chunk_docx_document() ----------

def make_docx_extracted(content):
    """Build a dictionary shaped like the output of extract_docx_text()."""
    return {
        "file_name": "demo_product.docx",
        "document_type": "docx",
        "content": content,
    }


def paragraph(text, index):
    return {"type": "paragraph", "text": text, "paragraph_index": index}


def heading(text, level, index):
    return {"type": "heading", "text": text, "heading_level": level, "paragraph_index": index}


def table(rows, index):
    return {"type": "table", "table_index": index, "rows": rows}


def test_table_to_lines_repeats_header_labels():
    rows = [
        ["Ingredient", "Strength"],
        ["Ingredient A", "100 mg"],
        ["Ingredient B", "50 mg"],
    ]

    assert table_to_lines(rows) == [
        "Ingredient: Ingredient A | Strength: 100 mg",
        "Ingredient: Ingredient B | Strength: 50 mg",
    ]


def test_table_to_lines_handles_empty_cells_and_single_row():
    rows = [
        ["", "CFU"],
        ["Strain A", "0.48 Billion"],
        ["", "2.02 Billion"],
        ["Strain B", ""],
        ["", ""],
    ]

    assert table_to_lines(rows) == [
        "Strain A | CFU: 0.48 Billion",
        "CFU: 2.02 Billion",
        "Strain B",
    ]
    assert table_to_lines([["Ingredient B", "0.25 mcg"]]) == ["Ingredient B | 0.25 mcg"]


def test_docx_without_headings():
    extracted = make_docx_extracted([
        paragraph("DemoCompound Alpha is a fictional tablet.", 0),
        paragraph("Store below 25°C.", 1),
    ])

    chunks = chunk_docx_document(extracted, product_id="DEMO-001")

    assert len(chunks) == 1
    assert chunks[0]["section_heading"] is None
    assert chunks[0]["heading_level"] is None


def test_docx_with_explicit_word_headings(tmp_path):
    docx_path = tmp_path / "product.docx"
    document = Document()
    document.add_heading("Composition", level=1)
    document.add_paragraph("Each tablet contains:")
    word_table = document.add_table(rows=2, cols=2)
    word_table.cell(0, 0).text = "Ingredient"
    word_table.cell(0, 1).text = "Strength"
    word_table.cell(1, 0).text = "Ingredient A"
    word_table.cell(1, 1).text = "100 mg"
    document.add_heading("Storage", level=2)
    document.add_paragraph("Store below 25°C.")
    document.save(docx_path)

    chunks = chunk_docx_document(extract_docx_text(docx_path), product_id="DEMO-001")

    assert len(chunks) == 2
    assert chunks[0]["text"] == (
        "Composition\nEach tablet contains:\nIngredient: Ingredient A | Strength: 100 mg"
    )
    assert chunks[0]["section_heading"] == "Composition"
    assert chunks[0]["heading_level"] == 1
    assert chunks[0]["paragraph_indices"] == [0, 1]
    assert chunks[0]["table_indices"] == [0]
    assert chunks[1]["text"] == "Storage\nStore below 25°C."
    assert chunks[1]["section_heading"] == "Storage"
    assert chunks[1]["heading_level"] == 2
    assert chunks[1]["paragraph_indices"] == [2, 3]


def test_multiple_paragraphs_are_packed_into_one_chunk():
    extracted = make_docx_extracted([
        paragraph("First paragraph.", 0),
        paragraph("Second paragraph.", 1),
        paragraph("Third paragraph.", 2),
    ])

    chunks = chunk_docx_document(extracted, product_id="DEMO-001")

    assert len(chunks) == 1
    assert chunks[0]["text"] == "First paragraph.\nSecond paragraph.\nThird paragraph."
    assert chunks[0]["paragraph_indices"] == [0, 1, 2]


def test_docx_order_is_preserved():
    extracted = make_docx_extracted([
        paragraph("Each tablet contains:", 0),
        table([["Ingredient", "Strength"], ["Ingredient A", "100 mg"]], 0),
        paragraph("Store below 25°C.", 2),
    ])

    chunks = chunk_docx_document(extracted, product_id="DEMO-001")

    assert chunks[0]["text"] == (
        "Each tablet contains:\n"
        "Ingredient: Ingredient A | Strength: 100 mg\n"
        "Store below 25°C."
    )
    assert chunks[0]["paragraph_indices"] == [0, 2]
    assert chunks[0]["table_indices"] == [0]


def test_document_with_only_tables():
    extracted = make_docx_extracted([
        table([["Ingredient", "Strength"], ["Ingredient A", "100 mg"], ["Ingredient B", "50 mg"]], 0),
    ])

    chunks = chunk_docx_document(extracted, product_id="DEMO-001")

    assert len(chunks) == 1
    assert chunks[0]["text"] == (
        "Ingredient: Ingredient A | Strength: 100 mg\n"
        "Ingredient: Ingredient B | Strength: 50 mg"
    )
    assert chunks[0]["section_heading"] is None
    assert chunks[0]["paragraph_indices"] == []
    assert chunks[0]["table_indices"] == [0]


def test_long_table_keeps_ingredient_with_strength():
    rows = [["Ingredient", "Strength"]]
    for number in range(1, 31):
        rows.append([f"Ingredient {number}", f"{number * 10} mg"])
    extracted = make_docx_extracted([table(rows, 0)])

    chunks = chunk_docx_document(extracted, product_id="DEMO-001", chunk_size=120, overlap=50)

    assert len(chunks) > 1
    row_pattern = re.compile(r"^Ingredient: Ingredient (\d+) \| Strength: (\d+) mg$")
    seen_rows = set()
    for chunk in chunks:
        assert chunk["table_indices"] == [0]
        for line in chunk["text"].split("\n"):
            match = row_pattern.match(line)
            assert match, f"broken table row: {line!r}"
            number, strength = int(match.group(1)), int(match.group(2))
            assert strength == number * 10  # each ingredient stays with its own strength
            seen_rows.add(number)
    assert seen_rows == set(range(1, 31))


def test_overlap_repeats_whole_paragraphs():
    extracted = make_docx_extracted([
        paragraph("First paragraph.", 0),
        paragraph("Second paragraph.", 1),
        paragraph("Third paragraph.", 2),
    ])

    chunks = chunk_docx_document(extracted, product_id="DEMO-001", chunk_size=40, overlap=20)

    assert [chunk["text"] for chunk in chunks] == [
        "First paragraph.\nSecond paragraph.",
        "Second paragraph.\nThird paragraph.",
    ]
    assert chunks[1]["paragraph_indices"] == [1, 2]


def test_overlap_never_splits_table_rows():
    rows = [["Ingredient", "Strength"]]
    for number in range(1, 7):
        rows.append([f"Ingredient {number}", f"{number} mg"])
    extracted = make_docx_extracted([table(rows, 0)])
    complete_rows = set(table_to_lines(rows))

    chunks = chunk_docx_document(extracted, product_id="DEMO-001", chunk_size=90, overlap=45)

    all_lines = []
    for chunk in chunks:
        for line in chunk["text"].split("\n"):
            assert line in complete_rows  # only whole rows, never partial ones
            all_lines.append(line)
    # Overlap happened: some rows appear in two chunks.
    assert len(all_lines) > len(complete_rows)


def test_docx_chunks_keep_metadata():
    extracted = make_docx_extracted([
        heading("Composition", 1, 0),
        paragraph("Each tablet contains 100 mg of Ingredient A.", 1),
        paragraph("Each tablet contains 50 mg of Ingredient B.", 2),
    ])

    chunks = chunk_docx_document(extracted, product_id="DEMO-002", chunk_size=60, overlap=10)

    assert [chunk["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert set(chunk.keys()) == {
            "text", "chunk_index", "product_id", "file_name",
            "section_heading", "heading_level", "paragraph_indices", "table_indices",
        }
        assert chunk["product_id"] == "DEMO-002"
        assert chunk["file_name"] == "demo_product.docx"
        assert "page_number" not in chunk


def test_section_metadata_is_kept_on_every_chunk():
    content = [heading("Storage", 1, 0)]
    for index in range(1, 6):
        content.append(paragraph(f"Storage instruction number {index}.", index))
    extracted = make_docx_extracted(content)

    chunks = chunk_docx_document(extracted, product_id="DEMO-001", chunk_size=80, overlap=0)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["section_heading"] == "Storage"
        assert chunk["heading_level"] == 1
    # The heading line itself only appears in the first chunk.
    assert chunks[0]["text"].startswith("Storage\n")
    assert chunks[0]["paragraph_indices"][0] == 0
    assert all(0 not in chunk["paragraph_indices"] for chunk in chunks[1:])


def test_empty_docx_content_returns_no_chunks():
    assert chunk_docx_document(make_docx_extracted([]), product_id="DEMO-001") == []


def test_heading_without_content_is_skipped():
    extracted = make_docx_extracted([
        heading("Product Information", 1, 0),
        heading("Storage", 2, 1),
        paragraph("Store below 25°C.", 2),
    ])

    chunks = chunk_docx_document(extracted, product_id="DEMO-001")

    assert len(chunks) == 1
    assert chunks[0]["section_heading"] == "Storage"
    assert chunks[0]["paragraph_indices"] == [1, 2]


def test_long_paragraph_is_split():
    long_text = "This is sentence number one about DemoCompound Alpha. " * 20
    extracted = make_docx_extracted([heading("Description", 1, 0), paragraph(long_text, 1)])

    chunks = chunk_docx_document(extracted, product_id="DEMO-001", chunk_size=200, overlap=50)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk["text"]) <= 200
        assert chunk["section_heading"] == "Description"
    assert chunks[0]["paragraph_indices"] == [0, 1]
    assert chunks[1]["paragraph_indices"] == [1]


def test_oversized_table_row_is_kept_whole():
    long_name = "Fictional Ingredient " + "with a very long chemical name " * 4
    rows = [["Ingredient", "Strength"], [long_name, "2.5 mg"]]
    extracted = make_docx_extracted([table(rows, 0)])

    chunks = chunk_docx_document(extracted, product_id="DEMO-001", chunk_size=60, overlap=10)

    expected_row = f"Ingredient: {clean_text(long_name)} | Strength: 2.5 mg"
    assert len(chunks) == 1
    assert chunks[0]["text"] == expected_row
    assert len(chunks[0]["text"]) > 60  # allowed: the row is never cut


def test_docx_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError, match="overlap must be smaller"):
        chunk_docx_document(make_docx_extracted([]), product_id="DEMO-001", chunk_size=100, overlap=100)


def test_pdf_chunking_is_unchanged():
    document = make_document(["Text on page one.", "Text on page two."])

    chunks = chunk_document(document, product_id="DEMO-001")

    assert chunks == [
        {"text": "Text on page one.", "page_number": 1, "chunk_index": 0,
         "product_id": "DEMO-001", "file_name": "demo_leaflet.pdf"},
        {"text": "Text on page two.", "page_number": 2, "chunk_index": 1,
         "product_id": "DEMO-001", "file_name": "demo_leaflet.pdf"},
    ]
