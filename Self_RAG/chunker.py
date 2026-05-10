import hashlib
from pydantic import BaseModel

class Chunk(BaseModel):
    id: str
    text: str
    metadata: dict

def get_chunk_id(source: str, chunk_index: int, text: str) -> str:
    """Create a unique deterministic hash for a chunk based on its content and source."""
    content_to_hash = f"{source}_{chunk_index}_{text}"
    return hashlib.md5(content_to_hash.encode("utf-8")).hexdigest()

def recursive_character_split(text: str, chunk_size: int, chunk_overlap: int, separators: list[str]) -> list[str]:
    """
    Splits text recursively using a list of separators.
    """
    final_chunks = []
    
    # Try to find a separator that splits the text into chunks smaller than chunk_size
    separator = separators[-1] # Fallback to empty string (character split)
    for s in separators:
        if s == "":
            break
        if s in text:
            separator = s
            break
            
    if separator:
        splits = text.split(separator)
    else:
        splits = list(text) # Character level
        
    good_splits = []
    _good_splits = []
    for s in splits:
        if len(s) < chunk_size:
            _good_splits.append(s)
        else:
            if _good_splits:
                good_splits.append(separator.join(_good_splits))
                _good_splits = []
            if not s: continue
            # If a single split is larger than chunk_size, we need to recurse
            next_separators = separators[separators.index(separator)+1:] if separator in separators else separators
            good_splits.extend(recursive_character_split(s, chunk_size, chunk_overlap, next_separators))
    
    if _good_splits:
        good_splits.append(separator.join(_good_splits))
        
    # Merge splits into chunks of size <= chunk_size, with overlap
    current_chunk = []
    current_length = 0
    
    for split in good_splits:
        split_len = len(split) + (len(separator) if current_chunk else 0)
        
        if current_length + split_len > chunk_size and current_chunk:
            final_chunks.append(separator.join(current_chunk))
            
            # Start new chunk with overlap
            # Simple strategy: keep removing from the beginning of current_chunk until it fits with new split
            # and is under overlap size limit
            while current_length > chunk_overlap or (current_length + split_len > chunk_size and current_chunk):
                current_length -= len(current_chunk[0]) + len(separator)
                current_chunk.pop(0)
                
            current_chunk.append(split)
            current_length += split_len
        else:
            current_chunk.append(split)
            current_length += split_len
            
    if current_chunk:
        final_chunks.append(separator.join(current_chunk))
        
    return final_chunks

def chunk_documents(documents: list[dict], chunk_size: int = 800, chunk_overlap: int = 150) -> list[Chunk]:
    """
    Splits documents into smaller chunks using recursive character splitting.
    """
    chunks = []
    separators = ["\n\n", "\n", ". ", " ", ""]
    
    for doc in documents:
        text = doc["text"]
        source = doc["metadata"].get("source", "unknown")
        
        text_chunks = recursive_character_split(text, chunk_size, chunk_overlap, separators)
        
        for i, chunk_text in enumerate(text_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
                
            chunk_id = get_chunk_id(source, i, chunk_text)
            chunks.append(Chunk(
                id=chunk_id,
                text=chunk_text,
                metadata={
                    "source": source,
                    "chunk_index": i
                }
            ))
            
    return chunks
