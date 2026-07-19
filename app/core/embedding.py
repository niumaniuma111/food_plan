"""
DashScope Embedding wrapper for text vectorization.
"""
from typing import List

from langchain_community.embeddings import DashScopeEmbeddings

from app.config import get_settings


class EmbeddingService:
    """DashScope Embedding service for text vectorization."""
    
    def __init__(self):
        settings = get_settings()
        self.model = settings.embedding.model
        
        self.embeddings = DashScopeEmbeddings(
            model=self.model,
            dashscope_api_key=settings.dashscope_api_key,
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        return self.embeddings.embed_query(text)
    
    def embed_documents_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Embed documents in batches to avoid API rate limits.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts per batch
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings


# Singleton instance
_embedding_service: EmbeddingService = None


def get_embedding_service() -> EmbeddingService:
    """Get or create embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
