"""
chunker.py — Recursive Character Text Splitting.

This is the industry-standard chunking strategy used in production
RAG systems. It tries to split on the coarsest separator first
(paragraph boundaries), falling back to finer separators only when
a chunk is still too large.

This preserves semantic coherence: paragraphs stay intact when
possible, sentences aren't broken mid-word.
"""

from dataclasses import dataclass, field

from config import CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS
from document_loader import Document


@dataclass
class Chunk:
    """A single chunk of text with metadata tracing it back to its source."""
    text: str
    metadata: dict = field(default_factory=dict)


def _split_text(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """
    Recursively split *text* using the hierarchy of *separators*.

    Algorithm:
    1. Pick the first separator that actually appears in the text.
    2. Split the text on that separator.
    3. Merge consecutive splits into chunks that fit within chunk_size.
    4. If any merged piece is STILL too large, recurse with the next
       (finer) separator.
    """
    # Base case: text already fits
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Find the best (coarsest) separator that exists in the text
    separator = ""
    remaining_separators = []
    for i, sep in enumerate(separators):
        if sep == "":
            separator = sep
            remaining_separators = []
            break
        if sep in text:
            separator = sep
            remaining_separators = separators[i + 1:]
            break

    # Split on the chosen separator
    if separator:
        pieces = text.split(separator)
    else:
        # Character-level split (last resort)
        pieces = list(text)

    # Merge small pieces back together until they fill a chunk
    chunks: list[str] = []
    current = ""

    for piece in pieces:
        # What would the merged text look like?
        candidate = current + separator + piece if current else piece

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            # Flush the current chunk
            if current.strip():
                chunks.append(current)
            # If this single piece is too big, recurse with finer separators
            if len(piece) > chunk_size and remaining_separators:
                sub_chunks = _split_text(piece, remaining_separators, chunk_size)
                chunks.extend(sub_chunks)
                current = ""
            else:
                current = piece

    # Don't forget the last accumulated chunk
    if current.strip():
        chunks.append(current)

    return chunks


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """
    Apply overlap by prepending the tail of the previous chunk
    to the beginning of each subsequent chunk.

    This ensures that information at chunk boundaries is not lost
    during retrieval.
    """
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped: list[str] = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:]
        overlapped.append(prev_tail + chunks[i])

    return overlapped


def chunk_document(document: Document) -> list[Chunk]:
    """
    Split a single Document into a list of Chunks.

    Each Chunk inherits the parent document's metadata plus
    a chunk_index for traceability.
    """
    raw_chunks = _split_text(document.text, SEPARATORS, CHUNK_SIZE)
    raw_chunks = _apply_overlap(raw_chunks, CHUNK_OVERLAP)

    chunks: list[Chunk] = []
    for i, text in enumerate(raw_chunks):
        chunk_metadata = {
            **document.metadata,
            "chunk_index": i,
            "chunk_total": len(raw_chunks),
        }
        chunks.append(Chunk(text=text, metadata=chunk_metadata))

    return chunks


def chunk_documents(documents: list[Document]) -> list[Chunk]:
    """Chunk a list of Documents and return a flat list of Chunks."""
    all_chunks: list[Chunk] = []
    for doc in documents:
        doc_chunks = chunk_document(doc)
        all_chunks.extend(doc_chunks)
        print(f"  [CHUNK] {doc.metadata.get('source', '?')}: {len(doc_chunks)} chunks")
    return all_chunks
