import hashlib
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split documents into overlapping chunks with deterministic IDs."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    
    # Add deterministic IDs to prevent duplicate inserts later
    for i, chunk in enumerate(chunks):
        content_hash = hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest()
        chunk.metadata["chunk_id"] = f"{chunk.metadata.get('source', 'unknown')}_{i}_{content_hash[:8]}"
        
    return chunks
