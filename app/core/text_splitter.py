"""
Text splitter for document chunking.
"""
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings


class TextSplitter:
    """Split documents into chunks with overlap."""
    
    def __init__(self):
        settings = get_settings()
        self.chunk_size = settings.chunking.chunk_size
        self.chunk_overlap = settings.chunking.chunk_overlap
        
        # Separators prioritized for Chinese + Markdown content
        self.separators = [
            "\n## ",      # Markdown H2
            "\n### ",     # Markdown H3
            "\n#### ",    # Markdown H4
            "\n\n",       # Paragraph break
            "\n",         # Line break
            "。",         # Chinese period
            "；",         # Chinese semicolon
            ".",          # English period
            " ",          # Space
        ]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects with preserved metadata
        """
        chunks = []
        
        for doc in documents:
            split_docs = self.splitter.split_documents([doc])
            
            # Add chunk index to metadata
            for i, chunk in enumerate(split_docs):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["total_chunks"] = len(split_docs)
            
            chunks.extend(split_docs)
        
        return chunks
    
    def split_text(self, text: str, metadata: dict = None) -> List[Document]:
        """
        Split a text string into chunks.
        
        Args:
            text: Text to split
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of Document chunks
        """
        texts = self.splitter.split_text(text)
        
        documents = []
        for i, text_chunk in enumerate(texts):
            doc = Document(
                page_content=text_chunk,
                metadata={
                    **(metadata or {}),
                    "chunk_index": i,
                    "total_chunks": len(texts),
                }
            )
            documents.append(doc)
        
        return documents
