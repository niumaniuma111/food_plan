"""
Chroma vector store wrapper.
"""
from typing import List, Optional, Tuple
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma

from app.config import get_settings
from app.core.embedding import get_embedding_service


class VectorStoreService:
    """Chroma vector store service."""
    
    def __init__(self, collection_name: str = "documents"):
        """
        Initialize vector store.
        
        Args:
            collection_name: Name of the Chroma collection
        """
        settings = get_settings()
        self.collection_name = collection_name
        self.persist_directory = settings.chroma.persist_dir
        
        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize Chroma with LangChain
        embedding_service = get_embedding_service()
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_service.embeddings,
            persist_directory=self.persist_directory,
        )
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of document IDs
        """
        return self.vectorstore.add_documents(documents)
    
    def upsert_documents(self, documents: List[Document]) -> List[str]:
        """
        Upsert documents (insert or update).
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of document IDs
        """
        # Generate IDs for documents
        import hashlib
        
        ids = []
        for doc in documents:
            # Create unique ID from content hash
            content_hash = hashlib.md5(
                f"{doc.metadata.get('source', '')}:{doc.page_content[:100]}".encode()
            ).hexdigest()
            ids.append(content_hash)
            doc.metadata["chunk_id"] = content_hash
        
        # Use add_documents with IDs (Chroma will upsert)
        return self.vectorstore.add_documents(documents, ids=ids)
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        filter: Optional[dict] = None
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of similar Document objects
        """
        return self.vectorstore.similarity_search(query, k=k, filter=filter)
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 5,
        filter: Optional[dict] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents with scores.
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of (Document, score) tuples
        """
        return self.vectorstore.similarity_search_with_score(query, k=k, filter=filter)
    
    def delete_documents(self, ids: List[str]) -> bool:
        """
        Delete documents by IDs.
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            True if successful
        """
        self.vectorstore.delete(ids=ids)
        return True
    
    def get_collection_stats(self) -> dict:
        """
        Get collection statistics.
        
        Returns:
            Dictionary with collection stats
        """
        collection = self.vectorstore._collection
        count = collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
        }


# Singleton instances for different collections
_vectorstore_instances = {}


def get_vectorstore(collection_name: str = "documents") -> VectorStoreService:
    """Get or create vector store instance for a collection."""
    if collection_name not in _vectorstore_instances:
        _vectorstore_instances[collection_name] = VectorStoreService(collection_name)
    return _vectorstore_instances[collection_name]
