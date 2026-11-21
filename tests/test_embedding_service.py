import copy

import numpy as np
import pytest

from app.services.embedding_service import (
    EMBEDDING_DIMENSION,
    MAX_TOKENS,
    count_tokens,
    embed_chunks,
    embed_query,
    find_chunks_over_token_limit,
    get_model,
)

# Note: the first test run downloads the model (about 135 MB) and needs internet.


def make_pdf_chunk(text, chunk_index=0):
    return {
        "text": text,
        "page_number": 1,
        "chunk_index": chunk_index,
        "product_id": "DEMO-001",
        "file_name": "demo_leaflet.pdf",
    }


def make_docx_chunk(text, chunk_index=0):
    return {
        "text": text,
        "chunk_index": chunk_index,
        "product_id": "DEMO-002",
        "file_name": "demo_product.docx",
        "section_heading": "Composition",
        "heading_level": 1,
        "paragraph_indices": [0, 1],
        "table_indices": [0],
    }


def test_model_is_loaded_only_once():
    first = get_model()
    second = get_model()

    assert first is second
    assert first.get_sentence_embedding_dimension() == EMBEDDING_DIMENSION
    assert first.max_seq_length == MAX_TOKENS


def test_embed_single_chunk():
    chunks = [make_pdf_chunk("DemoCompound Alpha contains 100 mg of Ingredient A.")]

    result = embed_chunks(chunks)

    assert len(result) == 1
    assert len(result[0]["embedding"]) == EMBEDDING_DIMENSION


def test_embed_multiple_chunks_in_batches():
    texts = [
        "DemoCompound Alpha contains 100 mg of Ingredient A.",
        "Store below 25°C in a dry place.",
        "Take 10 ml twice a day or as directed.",
    ]
    chunks = [make_pdf_chunk(text, index) for index, text in enumerate(texts)]

    batch_result = embed_chunks(chunks, batch_size=2)

    assert [chunk["text"] for chunk in batch_result] == texts
    # A chunk embedded inside a batch matches the same chunk embedded alone.
    # Small float differences are possible because batches are padded.
    for chunk, embedded in zip(chunks, batch_result):
        single = embed_chunks([chunk])[0]["embedding"]
        assert np.allclose(embedded["embedding"], single, atol=1e-5)


def test_embeddings_have_384_dimensions_and_unit_length():
    chunks = [
        make_pdf_chunk("Each 5 ml contains 2.5 mg."),
        make_docx_chunk("Ingredient: Ingredient A | Strength: 100 mg"),
    ]

    vectors = [chunk["embedding"] for chunk in embed_chunks(chunks)] + [embed_query("dosage")]

    for vector in vectors:
        assert len(vector) == EMBEDDING_DIMENSION
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)


def test_metadata_is_preserved_and_input_is_not_changed():
    chunks = [
        make_pdf_chunk("Store below 25°C."),
        make_docx_chunk("Ingredient: Ingredient A | Strength: 100 mg", chunk_index=1),
    ]
    original = copy.deepcopy(chunks)

    result = embed_chunks(chunks)

    assert chunks == original  # the input was not modified
    for before, after in zip(chunks, result):
        assert set(after.keys()) == set(before.keys()) | {"embedding"}
        for key, value in before.items():
            assert after[key] == value
    # Changing the result does not change the input.
    result[1]["paragraph_indices"].append(99)
    assert chunks[1]["paragraph_indices"] == [0, 1]


def test_empty_input():
    assert embed_chunks([]) == []

    with pytest.raises(ValueError, match="query cannot be empty"):
        embed_query("   ")


def test_too_long_chunk_raises_error_before_embedding():
    long_text = "DemoCompound Alpha contains fictional ingredient A. " * 100
    chunks = [
        make_pdf_chunk("A short chunk.", chunk_index=0),
        make_pdf_chunk(long_text, chunk_index=1),
    ]

    assert count_tokens(long_text) > MAX_TOKENS
    too_long = find_chunks_over_token_limit(chunks)
    assert too_long == [{
        "file_name": "demo_leaflet.pdf",
        "chunk_index": 1,
        "token_count": count_tokens(long_text),
    }]

    with pytest.raises(ValueError, match=r"demo_leaflet\.pdf chunk 1: \d+ tokens"):
        embed_chunks(chunks)


def test_count_tokens_includes_special_tokens():
    # [CLS] and [SEP] are always added, so an empty string is 2 tokens.
    assert count_tokens("") == 2


def test_query_is_closer_to_relevant_chunk():
    # A basic smoke test only. It does not prove retrieval quality.
    chunks = embed_chunks([
        make_pdf_chunk("Store below 25°C in a dry place, protected from light."),
        make_pdf_chunk("Each tablet contains 100 mg of Ingredient A.", chunk_index=1),
    ])
    query = embed_query("How should the syrup be stored?")

    storage_score = np.dot(query, chunks[0]["embedding"])
    composition_score = np.dot(query, chunks[1]["embedding"])
    assert storage_score > composition_score
