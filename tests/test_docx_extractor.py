import zipfile

import pytest
from docx import Document

from app.services.docx_extractor import extract_docx_text


def test_extract_single_paragraph(tmp_path):
    docx_path = tmp_path / "leaflet.docx"
    document = Document()
    document.add_paragraph("DemoCompound Alpha is a fictional tablet.")
    document.save(docx_path)

    result = extract_docx_text(docx_path)

    assert result["file_name"] == "leaflet.docx"
    assert result["document_type"] == "docx"
    assert result["content"] == [
        {
            "type": "paragraph",
            "text": "DemoCompound Alpha is a fictional tablet.",
            "paragraph_index": 0,
        }
    ]


def test_extract_multiple_paragraphs(tmp_path):
    docx_path = tmp_path / "multi.docx"
    document = Document()
    document.add_paragraph("First paragraph.")
    document.add_paragraph("Second paragraph.")
    document.add_paragraph("Third paragraph.")
    document.save(docx_path)

    result = extract_docx_text(docx_path)

    texts = [item["text"] for item in result["content"]]
    indices = [item["paragraph_index"] for item in result["content"]]
    assert texts == ["First paragraph.", "Second paragraph.", "Third paragraph."]
    assert indices == [0, 1, 2]


def test_extract_headings_with_levels(tmp_path):
    docx_path = tmp_path / "headings.docx"
    document = Document()
    document.add_heading("DemoCompound Alpha", level=0)
    document.add_heading("Product Composition", level=1)
    document.add_heading("Active Ingredients", level=2)
    document.save(docx_path)

    result = extract_docx_text(docx_path)

    headings = [(item["text"], item["heading_level"]) for item in result["content"]]
    assert headings == [
        ("DemoCompound Alpha", 0),
        ("Product Composition", 1),
        ("Active Ingredients", 2),
    ]
    assert all(item["type"] == "heading" for item in result["content"])


def test_extract_table_rows_and_cells(tmp_path):
    docx_path = tmp_path / "table.docx"
    document = Document()
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Ingredient"
    table.cell(0, 1).text = "Strength"
    table.cell(1, 0).text = "Ingredient A"
    table.cell(1, 1).text = "100 mg"
    document.save(docx_path)

    result = extract_docx_text(docx_path)

    assert result["content"] == [
        {
            "type": "table",
            "table_index": 0,
            "rows": [["Ingredient", "Strength"], ["Ingredient A", "100 mg"]],
        }
    ]


def test_paragraph_and_table_order_is_preserved(tmp_path):
    docx_path = tmp_path / "order.docx"
    document = Document()
    document.add_heading("Composition", level=1)
    document.add_paragraph("Each tablet contains:")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Ingredient A"
    table.cell(0, 1).text = "100 mg"
    document.add_paragraph("")  # empty paragraph: skipped, but still counted
    document.add_paragraph("Store below 25°C.")
    document.save(docx_path)

    result = extract_docx_text(docx_path)

    assert [item["type"] for item in result["content"]] == [
        "heading", "paragraph", "table", "paragraph"
    ]
    assert result["content"][0]["paragraph_index"] == 0
    assert result["content"][1]["paragraph_index"] == 1
    assert result["content"][3]["paragraph_index"] == 3
    assert result["content"][3]["text"] == "Store below 25°C."


def test_empty_document_raises_error(tmp_path):
    docx_path = tmp_path / "empty.docx"
    Document().save(docx_path)

    with pytest.raises(ValueError, match="No extractable text"):
        extract_docx_text(docx_path)


def test_missing_file_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_docx_text(tmp_path / "does_not_exist.docx")


def test_invalid_docx_raises_error(tmp_path):
    docx_path = tmp_path / "fake.docx"
    docx_path.write_text("This is plain text, not a Word document.")

    with pytest.raises(ValueError, match="not a valid DOCX"):
        extract_docx_text(docx_path)


def test_zip_file_that_is_not_word_raises_error(tmp_path):
    # A .docx is a zip file, but not every zip file is a Word document.
    docx_path = tmp_path / "archive.docx"
    with zipfile.ZipFile(docx_path, "w") as archive:
        archive.writestr("notes.txt", "Not a Word document")

    with pytest.raises(ValueError, match="not a valid DOCX"):
        extract_docx_text(docx_path)


def test_pharmaceutical_values_are_preserved(tmp_path):
    docx_path = tmp_path / "values.docx"
    document = Document()
    document.add_paragraph("Each 5 ml contains 2.5 mg. Dose: 0.5 ml. Store at 37°C.")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Ingredient B"
    table.cell(0, 1).text = "0.25 mcg"
    document.save(docx_path)

    result = extract_docx_text(docx_path)

    paragraph_text = result["content"][0]["text"]
    assert paragraph_text == "Each 5 ml contains 2.5 mg. Dose: 0.5 ml. Store at 37°C."
    assert result["content"][1]["rows"] == [["Ingredient B", "0.25 mcg"]]
