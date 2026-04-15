"""
chunker.py
Memory-efficient text chunker — pure Python, no LangChain needed.
"""


def chunk_text(text: str, chunk_size: int = 600, chunk_overlap: int = 60) -> list:
    """
    Split text into overlapping chunks efficiently.
    Uses larger chunk_size (600) to reduce total chunk count and memory usage.
    Skips duplicate or near-empty chunks.

    Args:
        text: Cleaned document text.
        chunk_size: Max characters per chunk (larger = fewer chunks = less RAM).
        chunk_overlap: Overlap characters between chunks.

    Returns:
        List of text chunks (max 300 chunks to prevent MemoryError).
    """
    # Hard limit on input text length to prevent runaway processing
    MAX_TEXT_LEN = 150_000  # ~150KB of text is more than enough for any lab report
    if len(text) > MAX_TEXT_LEN:
        text = text[:MAX_TEXT_LEN]

    chunks = []
    start  = 0
    length = len(text)
    seen   = set()  # deduplicate identical chunks

    while start < length:
        end = start + chunk_size

        # Try to break at a natural boundary
        if end < length:
            for sep in ['. ', '\n', ' ']:
                pos = text.rfind(sep, start, end)
                if pos != -1 and pos > start + 50:
                    end = pos + len(sep)
                    break

        chunk = text[start:end].strip()

        # Only keep meaningful, unique chunks
        if len(chunk) > 30:
            key = chunk[:80]  # use first 80 chars as dedup key
            if key not in seen:
                seen.add(key)
                chunks.append(chunk)

        # Safety: stop if too many chunks (prevents MemoryError)
        if len(chunks) >= 300:
            break

        # Move forward with overlap
        next_start = end - chunk_overlap
        if next_start <= start:  # prevent infinite loop
            next_start = start + chunk_size
        start = next_start

        if start >= length:
            break

    # Free the seen set from memory
    del seen

    return chunks
