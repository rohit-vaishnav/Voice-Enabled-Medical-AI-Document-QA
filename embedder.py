"""
embedder.py
Generate sentence embeddings using sentence-transformers.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# Load once at module level (cached for performance)
_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def generate_embeddings(chunks: list[str]) -> np.ndarray:
    """
    Encode a list of text chunks into embedding vectors.

    Args:
        chunks: List of text strings.

    Returns:
        2D numpy array of shape (n_chunks, embedding_dim).
    """
    model = _get_model()
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.astype("float32")


def embed_query(query: str) -> np.ndarray:
    """
    Encode a single query string into an embedding vector.

    Args:
        query: Query text.

    Returns:
        1D numpy array of shape (embedding_dim,).
    """
    model = _get_model()
    vec = model.encode([query], convert_to_numpy=True)
    return vec.astype("float32")
