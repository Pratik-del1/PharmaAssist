import copy

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384
MAX_TOKENS = 512  # includes the special [CLS] and [SEP] tokens

# Recommended by the BGE model card for short search queries.
# It is added to queries only, never to document chunks.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# The loaded model is kept here so it is only loaded once.
model = None


def get_model() -> SentenceTransformer:
    global model
    if model is None:
        model = SentenceTransformer(MODEL_NAME, device="cpu")
    return model


def count_tokens(text: str) -> int:
    """Count tokens the way the model sees them, including [CLS] and [SEP]."""
    tokenizer = get_model().tokenizer
    token_ids = tokenizer(text, truncation=False, verbose=False)["input_ids"]
    return len(token_ids)


def find_chunks_over_token_limit(chunks: list[dict]) -> list[dict]:
    too_long = []
    for chunk in chunks:
        token_count = count_tokens(chunk["text"])
        if token_count > MAX_TOKENS:
            too_long.append({
                "file_name": chunk.get("file_name"),
                "chunk_index": chunk.get("chunk_index"),
                "token_count": token_count,
            })
    return too_long


def embed_chunks(chunks: list[dict], batch_size: int = 32) -> list[dict]:
    """Return a copy of each chunk with an "embedding" added.

    Raises ValueError before embedding anything if a chunk is longer than
    MAX_TOKENS, because the model would silently cut off the extra text.
    """
    if not chunks:
        return []

    too_long = find_chunks_over_token_limit(chunks)
    if too_long:
        details = "; ".join(
            f"{chunk['file_name']} chunk {chunk['chunk_index']}: {chunk['token_count']} tokens"
            for chunk in too_long
        )
        raise ValueError(
            f"{len(too_long)} chunk(s) exceed the {MAX_TOKENS}-token limit "
            f"and would be truncated: {details}"
        )

    texts = [chunk["text"] for chunk in chunks]
    vectors = get_model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embedded_chunks = []
    for chunk, vector in zip(chunks, vectors):
        # deepcopy so the caller's chunk (including its lists) is never changed.
        new_chunk = copy.deepcopy(chunk)
        new_chunk["embedding"] = vector.tolist()
        embedded_chunks.append(new_chunk)

    return embedded_chunks


def embed_query(query: str) -> list[float]:
    query = query.strip()
    if not query:
        raise ValueError("query cannot be empty")

    text = QUERY_INSTRUCTION + query

    token_count = count_tokens(text)
    if token_count > MAX_TOKENS:
        raise ValueError(
            f"query is {token_count} tokens, which exceeds the {MAX_TOKENS}-token limit"
        )

    vector = get_model().encode(text, normalize_embeddings=True, show_progress_bar=False)
    return vector.tolist()
