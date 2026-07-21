"""
Cross-encoder reranker for document relevance scoring.
"""
from typing import List

from langchain_core.documents import Document
from langsmith import traceable
from sentence_transformers import CrossEncoder

from app.config import get_settings


class Reranker:
    """Cross-encoder based document reranker."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize reranker.
        
        Args:
            model_name: Name of the cross-encoder model
        """
        self.settings = get_settings()
        self.model_name = model_name
        
        # Load cross-encoder model (small ~80MB, CPU compatible)
        self.model = CrossEncoder(model_name)
    
    @traceable(name="Reranker", run_type="retriever")
    def rerank(self, query: str, documents: List[Document], top_k: int = None) -> List[Document]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: Query text
            documents: List of Document objects to rerank
            top_k: Number of top documents to return
            
        Returns:
            List of reranked Document objects
        """
        if top_k is None:
            top_k = self.settings.retrieval.rerank_top_k
        
        if not documents:
            return []
        
        # Create (query, document) pairs for scoring
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Predict relevance scores
        scores = self.model.predict(pairs)
        
        # Sort documents by score
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k documents with updated metadata
        results = []
        for doc, score in scored_docs[:top_k]:
            # Update metadata with rerank score
            doc.metadata["rerank_score"] = float(score)
            results.append(doc)
        
        return results


# Singleton instance
_reranker: Reranker = None


def get_reranker() -> Reranker:
    """Get or create reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
