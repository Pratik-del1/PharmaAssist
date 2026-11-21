import pytest

from app.services.text_processing import chunk_document, clean_text


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
