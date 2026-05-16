from pathlib import Path
from docling.document_converter import DocumentConverter
import logging

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".html", ".md", ".txt", ".pptx"}


def load_document(file_path: Path) -> str:
    """Parse a single document into markdown using Docling."""
    converter = DocumentConverter()
    result = converter.convert(str(file_path))
    return result.document.export_to_markdown()


def load_directory(directory_path: Path) -> list[dict]:
    """
    Load all supported documents from a directory.
    Returns a list of {text, metadata} dicts.
    """
    if not directory_path.exists():
        logger.warning(f"Data directory not found: {directory_path}")
        return []

    documents = []
    for file_path in directory_path.rglob("*"):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            logger.info(f"Parsing: {file_path.name}")
            text = load_document(file_path)
            documents.append({
                "text": text,
                "metadata": {"source": file_path.name, "path": str(file_path)},
            })
        except Exception as e:
            logger.error(f"Failed to parse {file_path.name}: {e}")

    logger.info(f"Loaded {len(documents)} documents from {directory_path}")
    return documents
