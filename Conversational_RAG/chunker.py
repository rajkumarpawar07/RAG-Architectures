from dataclasses import dataclass, field
from config import CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS
from document_loader import Document

@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)

def _split_text(text: str, separators: list[str], chunk_size: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

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

    pieces = text.split(separator) if separator else list(text)
    chunks = []
    current = ""

    for piece in pieces:
        candidate = current + separator + piece if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current.strip():
                chunks.append(current)
            if len(piece) > chunk_size and remaining_separators:
                chunks.extend(_split_text(piece, remaining_separators, chunk_size))
                current = ""
            else:
                current = piece

    if current.strip():
        chunks.append(current)

    return chunks

def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        overlapped.append(chunks[i - 1][-overlap:] + chunks[i])
    return overlapped

def chunk_document(document: Document) -> list[Chunk]:
    raw_chunks = _split_text(document.text, SEPARATORS, CHUNK_SIZE)
    raw_chunks = _apply_overlap(raw_chunks, CHUNK_OVERLAP)
    
    chunks = []
    for i, text in enumerate(raw_chunks):
        meta = {**document.metadata, "chunk_index": i, "chunk_total": len(raw_chunks)}
        chunks.append(Chunk(text=text, metadata=meta))
    return chunks

def chunk_documents(documents: list[Document]) -> list[Chunk]:
    all_chunks = []
    for doc in documents:
        doc_chunks = chunk_document(doc)
        all_chunks.extend(doc_chunks)
        print(f"  [CHUNK] {doc.metadata.get('source', '?')}: {len(doc_chunks)} chunks")
    return all_chunks
