"""
chunker.py
Split cleaned text into overlapping chunks for embedding.
"""

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int = 250, chunk_overlap: int = 100) -> list[str]:
    """
    Split text into chunks with overlap to preserve context across boundaries.

    Args:
        text: Cleaned document text.
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap characters between consecutive chunks.

    Returns:
        List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    # Remove empty/tiny chunks
    chunks = [c.strip() for c in chunks if len(c.strip()) > 30]
    return chunks
