from dataclasses import dataclass, field
from pathlib import Path
import fitz  # PyMuPDF

from config import DATA_DIR

@dataclass
class Document:
    text: str
    metadata: dict = field(default_factory=dict)

def load_pdf(file_path: Path) -> Document:
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    
    return Document(
        text=text,
        metadata={
            "source": file_path.name,
            "source_path": str(file_path),
            "file_type": "pdf",
        },
    )

def load_text(file_path: Path) -> Document:
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = file_path.read_text(encoding="latin-1")
    return Document(
        text=text,
        metadata={"source": file_path.name, "source_path": str(file_path), "file_type": "txt"},
    )

_LOADERS = {".pdf": load_pdf, ".txt": load_text}

def load_directory(dir_path: Path | None = None) -> list[Document]:
    dir_path = dir_path or DATA_DIR
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
        return []

    documents = []
    for file_path in sorted(dir_path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in _LOADERS:
            loader = _LOADERS[file_path.suffix.lower()]
            print(f"  [LOAD] {file_path.name}")
            try:
                doc = loader(file_path)
                if doc.text.strip():
                    documents.append(doc)
            except Exception as e:
                print(f"         ERROR: {e}")
    return documents
