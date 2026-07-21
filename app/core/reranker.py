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
        
        # Expand query with intent keywords to help cross-encoder understand user intent
        # e.g. "番茄炒蛋怎么做" -> "番茄炒蛋 做法 食谱 怎么做"
        expanded_query = self._expand_query(query)
        
        # Create (query, document) pairs for scoring
        pairs = [(expanded_query, doc.page_content) for doc in documents]
        
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

    def _expand_query(self, query: str) -> str:
        """
        Expand query with intent keywords to improve reranking accuracy.
        
        Adds domain-specific keywords based on common question patterns
        so the cross-encoder better understands user intent.
        """
        intent_keywords = {
            # 做法类问题
            "怎么做": "做法 食谱 烹饪方法",
            "怎样做": "做法 食谱 烹饪方法",
            "怎么炒": "做法 食谱 烹饪方法",
            "怎么煮": "做法 食谱 烹饪方法",
            "怎么蒸": "做法 食谱 烹饪方法",
            "怎么炖": "做法 食谱 烹饪方法",
            "怎么烤": "做法 食谱 烹饪方法",
            "怎么煎": "做法 食谱 烹饪方法",
            "如何制作": "做法 食谱 烹饪方法",
            "做法": "食谱 烹饪方法 步骤",
            # 营养/功效类
            "营养": "营养价值 功效 健康",
            "功效": "营养价值 功效 作用",
            "好处": "营养价值 功效 作用",
            # 食材类
            "能不能吃": "适合 禁忌 注意事项",
            "可以吃吗": "适合 禁忌 注意事项",
        }
        
        expansion = query
        for pattern, keywords in intent_keywords.items():
            if pattern in query:
                expansion = f"{query} {keywords}"
                break
        
        return expansion


# Singleton instance
_reranker: Reranker = None


def get_reranker() -> Reranker:
    """Get or create reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
