"""
BM25 keyword-based retriever.
"""
import json
from typing import List, Tuple

import jieba
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from app.config import get_settings


class BM25Retriever:
    """BM25 keyword-based document retriever with Chinese tokenization."""
    
    def __init__(self, index_path: str):
        """
        Initialize BM25 retriever.
        
        Args:
            index_path: Path to the BM25 index JSON file
        """
        self.index_path = index_path
        self.settings = get_settings()
        
        # Load index data
        self.documents_data = []  # List of {chunk_id, text, source, filename}
        self.corpus = []  # Tokenized corpus for BM25
        self.bm25 = None
        
        self._load_index()
    
    def _load_index(self) -> None:
        """Load BM25 index from file."""
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                self.documents_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.documents_data = []
        
        # Build tokenized corpus
        self.corpus = [self._tokenize(doc["text"]) for doc in self.documents_data]
        
        # Build BM25 model if corpus is not empty
        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)
        else:
            self.bm25 = None
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using jieba for Chinese + simple splitting for English.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Use jieba for Chinese text
        tokens = list(jieba.cut(text))
        # Filter out whitespace and punctuation
        tokens = [t.strip() for t in tokens if t.strip()]
        return tokens
    
    def retrieve(self, query: str, top_k: int = None) -> List[Document]:
        """
        Retrieve documents using BM25 scoring.
        
        Args:
            query: Query text
            top_k: Number of documents to return
            
        Returns:
            List of Document objects with scores
        """
        if top_k is None:
            top_k = self.settings.retrieval.bm25_top_k
        
        if not self.bm25 or not self.documents_data:
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Calculate BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        # Build result documents
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include documents with positive scores
                doc_data = self.documents_data[idx]
                doc = Document(
                    page_content=doc_data["text"],
                    metadata={
                        "chunk_id": doc_data["chunk_id"],
                        "source": doc_data["source"],
                        "filename": doc_data["filename"],
                        "score": float(scores[idx]),
                        "retrieval_method": "bm25",
                    }
                )
                results.append(doc)
        
        return results
    
    def reload_index(self) -> None:
        """Reload the BM25 index from file."""
        self._load_index()
    
    def add_documents(self, documents: List[dict]) -> None:
        """
        Add documents to the BM25 index.
        
        Args:
            documents: List of document dicts with chunk_id, text, source, filename
        """
        # Add to documents_data
        self.documents_data.extend(documents)
        
        # Update tokenized corpus
        for doc in documents:
            self.corpus.append(self._tokenize(doc["text"]))
        
        # Rebuild BM25 model
        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)
        
        # Save to file
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.documents_data, f, ensure_ascii=False, indent=2)
    
    def remove_document(self, chunk_id: str) -> bool:
        """
        Remove a document from the BM25 index.
        
        Args:
            chunk_id: ID of the document to remove
            
        Returns:
            True if removed, False if not found
        """
        # Find index
        idx = None
        for i, doc in enumerate(self.documents_data):
            if doc["chunk_id"] == chunk_id:
                idx = i
                break
        
        if idx is None:
            return False
        
        # Remove from data structures
        self.documents_data.pop(idx)
        self.corpus.pop(idx)
        
        # Rebuild BM25 model
        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)
        else:
            self.bm25 = None
        
        # Save to file
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.documents_data, f, ensure_ascii=False, indent=2)
        
        return True
