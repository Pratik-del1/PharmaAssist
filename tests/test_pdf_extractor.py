import pymupdf
import pytest

from app.services.pdf_extractor import extract_pdf_text


def create_pdf(path, page_texts):
    """Create a small PDF with one page per item in page_texts.

    An empty string creates a blank page.
    """
    document = pymupdf.open()
    for text in page_texts:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_extract_single_page_pdf(tmp_path):
    pdf_path = tmp_path / "leaflet.pdf"
    create_pdf(pdf_path, ["DemoCompound Alpha leaflet"])

    result = extract_pdf_text(pdf_path)

    assert result["file_name"] == "leaflet.pdf"
    assert result["page_count"] == 1
    assert result["pages"][0]["page_number"] == 1
    assert result["pages"][0]["text"] == "DemoCompound Alpha leaflet"


def test_extract_multiple_page_pdf(tmp_path):
    pdf_path = tmp_path / "multi.pdf"
    create_pdf(pdf_path, ["Page one text", "Page two text", "Page three text"])

    result = extract_pdf_text(pdf_path)

    assert result["page_count"] == 3
    assert [page["page_number"] for page in result["pages"]] == [1, 2, 3]
    assert result["pages"][1]["text"] == "Page two text"


def test_empty_page_is_kept_when_other_pages_have_text(tmp_path):
    pdf_path = tmp_path / "with_blank.pdf"
    create_pdf(pdf_path, ["First page", "", "Third page"])

    result = extract_pdf_text(pdf_path)

    assert result["page_count"] == 3
    assert result["pages"][1] == {"page_number": 2, "text": ""}
    assert result["pages"][2]["text"] == "Third page"


def test_missing_file_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_pdf_text(tmp_path / "does_not_exist.pdf")


def test_invalid_pdf_raises_error(tmp_path):
    pdf_path = tmp_path / "fake.pdf"
    pdf_path.write_text("This is plain text, not a PDF.")

    with pytest.raises(ValueError, match="not a valid PDF"):
        extract_pdf_text(pdf_path)


def test_non_pdf_document_raises_error(tmp_path):
    # PyMuPDF opens .txt files without an error, so the is_pdf check must catch it.
    text_path = tmp_path / "notes.txt"
    text_path.write_text("Some notes")

    with pytest.raises(ValueError, match="not a PDF document"):
        extract_pdf_text(text_path)


def test_pdf_without_text_raises_error(tmp_path):
    pdf_path = tmp_path / "blank.pdf"
    create_pdf(pdf_path, ["", ""])

    with pytest.raises(ValueError, match="No extractable text"):
        extract_pdf_text(pdf_path)
