from pathlib import Path
from docling.document_converter import DocumentConverter

def load_document(file_path: Path) -> str:
    """Parses a document into markdown using Docling."""
    converter = DocumentConverter()
    result = converter.convert(str(file_path))
    return result.document.export_to_markdown()

def load_directory(directory_path: Path) -> list[dict]:
    """Loads all supported documents in a directory and returns a list of dictionaries with text and metadata."""
    if not directory_path.exists():
        return []

    documents = []
    # Docling supports many formats natively
    supported_extensions = {'.pdf', '.docx', '.html', '.md', '.txt', '.pptx'}
    
    for file_path in directory_path.rglob('*'):
        if file_path.suffix.lower() in supported_extensions:
            try:
                text = load_document(file_path)
                documents.append({
                    "text": text,
                    "metadata": {"source": file_path.name}
                })
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")
                
    return documents
