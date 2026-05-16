import hashlib
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict


def _get_chunk_id(source: str, index: int, text: str) -> str:
    raw = f"{source}_{index}_{text[:64]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Recursive character splitter with overlap."""
    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks: list[str] = []

    def _split(txt: str, seps: list[str]):
        sep = seps[0]
        parts = txt.split(sep) if sep else list(txt)
        current, current_len = [], 0

        for part in parts:
            part_len = len(part) + len(sep)
            if current_len + part_len > chunk_size and current:
                chunks.append(sep.join(current))
                # retain overlap
                while current and current_len > chunk_overlap:
                    current_len -= len(current[0]) + len(sep)
                    current.pop(0)
            if len(part) > chunk_size and len(seps) > 1:
                _split(part, seps[1:])
            else:
                current.append(part)
                current_len += part_len

        if current:
            chunks.append(sep.join(current))

    _split(text, separators)
    return [c.strip() for c in chunks if c.strip()]


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    all_chunks: list[Chunk] = []

    for doc in documents:
        source = doc["metadata"].get("source", "unknown")
        splits = _split_text(doc["text"], chunk_size, chunk_overlap)

        for i, text in enumerate(splits):
            all_chunks.append(
                Chunk(
                    id=_get_chunk_id(source, i, text),
                    text=text,
                    metadata={**doc["metadata"], "chunk_index": i},
                )
            )

    logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
    return all_chunks
