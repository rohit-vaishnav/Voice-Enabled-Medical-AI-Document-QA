"""
chunker.py
Split cleaned text into overlapping chunks — pure Python, no LangChain needed.
"""


def chunk_text(text: str, chunk_size: int = 400, chunk_overlap: int = 50) -> list:
    """
    Split text into overlapping chunks without any external dependencies.

    Args:
        text: Cleaned document text.
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap characters between consecutive chunks.

    Returns:
        List of text chunks.
    """
    chunks = []
    start  = 0
    length = len(text)

    while start < length:
        end = start + chunk_size

        # Try to break at a natural boundary (sentence or word)
        if end < length:
            # Prefer breaking at ". " or "\n"
            for sep in ['. ', '\n', ' ']:
                pos = text.rfind(sep, start, end)
                if pos != -1 and pos > start:
                    end = pos + len(sep)
                    break

        chunk = text[start:end].strip()
        if len(chunk) > 30:
            chunks.append(chunk)

        # Move forward with overlap
        start = end - chunk_overlap
        if start <= 0 or start >= length:
            break

    return chunks
