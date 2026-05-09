import nltk
from dataclasses import dataclass, field
from config import CHUNK_SIZE, CHUNK_OVERLAP
from document_loader import Document

# Download necessary NLTK data for sentence tokenization
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass

@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)

def _split_into_sentences(text: str) -> list[str]:
    try:
        return nltk.sent_tokenize(text)
    except Exception:
        # Fallback to naive splitting
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

def chunk_document(document: Document) -> list[Chunk]:
    sentences = _split_into_sentences(document.text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= CHUNK_SIZE:
            current_chunk += sentence + " "
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            # Handle sentences longer than CHUNK_SIZE
            if len(sentence) > CHUNK_SIZE:
                # Basic split if a single sentence is huge
                words = sentence.split()
                temp = ""
                for word in words:
                    if len(temp) + len(word) <= CHUNK_SIZE:
                        temp += word + " "
                    else:
                        chunks.append(temp.strip())
                        temp = word + " "
                current_chunk = temp
            else:
                current_chunk = sentence + " "
                
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    final_chunks = []
    for i, text in enumerate(chunks):
        meta = {**document.metadata, "chunk_index": i}
        final_chunks.append(Chunk(text=text, metadata=meta))
    return final_chunks

def chunk_documents(documents: list[Document]) -> list[Chunk]:
    all_chunks = []
    for doc in documents:
        doc_chunks = chunk_document(doc)
        all_chunks.extend(doc_chunks)
    return all_chunks
