from pathlib import Path
from docling.document_converter import DocumentConverter
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

def load_directory(data_dir: Path) -> list[Document]:
    """Parse all supported documents in the directory using Docling."""
    converter = DocumentConverter()
    docs = []

    if not data_dir.exists() or not data_dir.is_dir():
        logger.warning(f"Data directory {data_dir} does not exist.")
        return docs

    for file_path in data_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in [".pdf", ".docx", ".txt", ".md"]:
            logger.info(f"Parsing {file_path.name} with Docling...")
            result = converter.convert(str(file_path))
            markdown_content = result.document.export_to_markdown()
            
            doc = Document(
                page_content=markdown_content,
                metadata={"source": file_path.name}
            )
            docs.append(doc)
            
    return docs
