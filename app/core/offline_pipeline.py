"""
Offline data processing pipeline.
Orchestrates document parsing, chunking, embedding, and vector store ingestion.
"""
import json
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from app.config import get_settings
from app.core.document_parser import DocumentParser
from app.core.text_splitter import TextSplitter
from app.core.vector_store import get_vectorstore


class OfflinePipeline:
    """Offline data processing pipeline for document ingestion."""
    
    def __init__(self):
        self.settings = get_settings()
        self.parser = DocumentParser()
        self.splitter = TextSplitter()
        self.vectorstore = get_vectorstore("documents")
    
    def ingest_file(self, file_path: str) -> dict:
        """
        Ingest a single file into the vector store.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with ingestion results
        """
        # Parse document
        documents = self.parser.parse_file(file_path)
        
        # Split into chunks
        chunks = self.splitter.split_documents(documents)
        
        # Add to vector store
        ids = self.vectorstore.upsert_documents(chunks)
        
        # Update BM25 index
        self._update_bm25_index(chunks)
        
        return {
            "file": file_path,
            "chunks_created": len(chunks),
            "ids": ids,
        }
    
    def ingest_directory(self, dir_path: str) -> dict:
        """
        Ingest all supported files from a directory.
        
        Args:
            dir_path: Path to the directory
            
        Returns:
            Dictionary with ingestion results
        """
        # Parse all documents
        documents = self.parser.parse_directory(dir_path)
        
        if not documents:
            return {"files_processed": 0, "chunks_created": 0}
        
        # Split into chunks
        chunks = self.splitter.split_documents(documents)
        
        # Add to vector store
        ids = self.vectorstore.upsert_documents(chunks)
        
        # Update BM25 index
        self._update_bm25_index(chunks)
        
        return {
            "files_processed": len(documents),
            "chunks_created": len(chunks),
            "ids": ids,
        }
    
    def ingest_knowledge_base(self) -> dict:
        """
        Ingest the pre-built knowledge base.
        
        Returns:
            Dictionary with ingestion results
        """
        kb_dir = self.settings.knowledge_base_dir
        
        if not kb_dir.exists():
            return {"status": "knowledge_base_dir_not_found"}
        
        return self.ingest_directory(str(kb_dir))
    
    def _update_bm25_index(self, chunks: List[Document]) -> None:
        """
        Update BM25 index with new chunks.
        
        Args:
            chunks: List of document chunks to add
        """
        bm25_index_path = self.settings.bm25_index_path
        
        # Load existing index
        existing_index = []
        if bm25_index_path.exists():
            with open(bm25_index_path, "r", encoding="utf-8") as f:
                existing_index = json.load(f)
        
        # Add new chunks
        for chunk in chunks:
            entry = {
                "chunk_id": chunk.metadata.get("chunk_id", ""),
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", ""),
                "filename": chunk.metadata.get("filename", ""),
            }
            existing_index.append(entry)
        
        # Save updated index
        with open(bm25_index_path, "w", encoding="utf-8") as f:
            json.dump(existing_index, f, ensure_ascii=False, indent=2)


# Singleton instance
_pipeline: OfflinePipeline = None


def get_offline_pipeline() -> OfflinePipeline:
    """Get or create offline pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = OfflinePipeline()
    return _pipeline
