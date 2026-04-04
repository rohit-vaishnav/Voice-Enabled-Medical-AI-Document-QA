"""
vector_store.py
Build and query a FAISS index for semantic search over chunk embeddings.
"""

import numpy as np
import faiss


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a flat L2 FAISS index from embeddings.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def search_faiss(
    index: faiss.Index,
    chunks: list,
    query_embedding: np.ndarray,
    top_k: int = 3,
    query_text: str = ""
) -> list:
    """
    Retrieve top-k chunks using semantic search + keyword fallback.

    The keyword fallback ensures that even if embedding similarity misses
    a chunk (e.g. "WBC Count 10570"), direct keyword matching catches it.
    """
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # Step 1: Semantic search — fetch more candidates than needed
    fetch_k = min(top_k * 4, len(chunks))
    distances, indices = index.search(query_embedding, fetch_k)

    semantic_results = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            semantic_results.append(chunks[idx])

    # Step 2: Keyword fallback — find chunks containing query keywords
    keyword_results = []
    if query_text:
        # Use words longer than 2 chars as keywords
        keywords = [w.lower().strip("?.,!") for w in query_text.split() if len(w) > 2]
        for chunk in chunks:
            chunk_lower = chunk.lower()
            match_count = sum(1 for kw in keywords if kw in chunk_lower)
            if match_count >= 1 and chunk not in semantic_results:
                keyword_results.append((match_count, chunk))
        # Sort by number of keyword matches (best first)
        keyword_results.sort(key=lambda x: x[0], reverse=True)
        keyword_results = [c for _, c in keyword_results]

    # Step 3: Merge deduplicated results (semantic first, then keyword)
    combined = []
    seen = set()
    for chunk in semantic_results + keyword_results:
        if chunk not in seen:
            seen.add(chunk)
            combined.append(chunk)
        if len(combined) >= top_k:
            break

    return combined[:top_k]


def save_index(index: faiss.Index, path: str) -> None:
    faiss.write_index(index, path)


def load_index(path: str) -> faiss.Index:
    return faiss.read_index(path)
