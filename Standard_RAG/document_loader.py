"""
document_loader.py — Load and extract text from PDFs and text files.

Uses Docling for PDF parsing (high-quality structured extraction)
and plain file I/O for .txt files. Each loaded file is represented
as a Document dataclass carrying its text and metadata.
"""

from dataclasses import dataclass, field
from pathlib import Path

from docling.document_converter import DocumentConverter

from config import DATA_DIR


@dataclass
class Document:
    """A single loaded document with its text content and metadata."""
    text: str
    metadata: dict = field(default_factory=dict)


# ──────────────────────────────────────────────
# Individual loaders
# ──────────────────────────────────────────────

def load_pdf(file_path: Path) -> Document:
    """
    Extract text from a PDF using Docling's DocumentConverter.

    Docling parses the PDF into a structured document and exports
    it as Markdown, which preserves headings, tables, and lists
    far better than naive text extraction.
    """
    converter = DocumentConverter()
    result = converter.convert(str(file_path))
    text = result.document.export_to_markdown()

    return Document(
        text=text,
        metadata={
            "source": file_path.name,
            "source_path": str(file_path),
            "file_type": "pdf",
        },
    )


def load_text(file_path: Path) -> Document:
    """
    Load a plain text file, trying UTF-8 first then falling back
    to latin-1 (which never raises a decode error).
    """
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = file_path.read_text(encoding="latin-1")

    return Document(
        text=text,
        metadata={
            "source": file_path.name,
            "source_path": str(file_path),
            "file_type": "txt",
        },
    )


# ──────────────────────────────────────────────
# Directory scanner
# ──────────────────────────────────────────────

# Map of supported extensions -> loader functions
_LOADERS = {
    ".pdf": load_pdf,
    ".txt": load_text,
}


def load_directory(dir_path: Path | None = None) -> list[Document]:
    """
    Recursively scan *dir_path* for supported files and load them.

    Returns a list of Document objects. Unsupported file types are
    silently skipped.
    """
    dir_path = dir_path or DATA_DIR

    if not dir_path.exists():
        raise FileNotFoundError(f"Data directory not found: {dir_path}")

    documents: list[Document] = []
    supported = set(_LOADERS.keys())

    for file_path in sorted(dir_path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in supported:
            loader = _LOADERS[file_path.suffix.lower()]
            print(f"  [LOAD] Loading: {file_path.name}")
            try:
                doc = loader(file_path)
                if doc.text.strip():
                    documents.append(doc)
                    print(f"         OK ({len(doc.text):,} chars)")
                else:
                    print(f"         SKIP (empty after extraction)")
            except Exception as e:
                print(f"         ERROR: {e}")

    return documents
