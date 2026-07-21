"""
Hybrid retriever combining vector search and BM25 with RRF fusion.
"""
from typing import List, Dict
from collections import defaultdict

from langchain_core.documents import Document
from langsmith import traceable

from app.config import get_settings
from app.core.vector_store import get_vectorstore
from app.core.bm25_retriever import BM25Retriever


class HybridRetriever:
    """
    Hybrid retriever combining vector search (Chroma) and keyword search (BM25)
    using Reciprocal Rank Fusion (RRF).
    """
    
    def __init__(self, bm25_retriever: BM25Retriever):
        """
        Initialize hybrid retriever.
        
        Args:
            bm25_retriever: BM25 retriever instance
        """
        self.settings = get_settings()
        self.bm25_retriever = bm25_retriever
        self.vectorstore = get_vectorstore("documents")
        self.feedback_vectorstore = get_vectorstore("feedback_qa")
        
        # RRF constant
        self.rrf_k = 60
        
        # Feedback QA weight multiplier
        self.feedback_weight = 1.5
    
    @traceable(name="Hybrid Retrieval", run_type="retriever")
    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve documents using hybrid search with RRF fusion.
        
        Args:
            query: Query text
            
        Returns:
            List of Document objects sorted by RRF score
        """
        vector_top_k = self.settings.retrieval.vector_top_k
        bm25_top_k = self.settings.retrieval.bm25_top_k
        
        # 1. Vector search from documents collection
        vector_docs = self.vectorstore.similarity_search_with_score(query, k=vector_top_k)
        
        # 2. Vector search from feedback_qa collection (approved feedback)
        feedback_docs = []
        try:
            feedback_docs = self.feedback_vectorstore.similarity_search_with_score(
                query, k=vector_top_k, filter={"status": "approved"}
            )
        except Exception:
            # feedback_qa collection might not exist yet
            pass
        
        # 3. BM25 keyword search
        bm25_docs = self.bm25_retriever.retrieve(query, top_k=bm25_top_k)
        
        # 4. RRF fusion
        rrf_scores = self._rrf_fusion(vector_docs, bm25_docs, feedback_docs)
        
        # 5. Sort by RRF score and return top results
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        
        results = []
        for chunk_id, data in sorted_items[:self.settings.retrieval.rerank_top_k * 3]:
            doc = Document(
                page_content=data["content"],
                metadata={
                    "chunk_id": chunk_id,
                    "source": data["source"],
                    "filename": data.get("filename", ""),
                    "rrf_score": data["score"],
                    "retrieval_methods": data["methods"],
                }
            )
            results.append(doc)
        
        return results
    
    def _rrf_fusion(
        self,
        vector_docs: List[tuple],
        bm25_docs: List[Document],
        feedback_docs: List[tuple] = None
    ) -> Dict[str, dict]:
        """
        Perform Reciprocal Rank Fusion (RRF) on multiple retrieval results.
        
        RRF formula: score(d) = Σ 1/(k + rank_i(d))
        
        Args:
            vector_docs: Results from vector search (Document, score) tuples
            bm25_docs: Results from BM25 search
            feedback_docs: Results from feedback_qa collection
            
        Returns:
            Dictionary mapping chunk_id to {content, source, score, methods}
        """
        rrf_scores: Dict[str, dict] = defaultdict(lambda: {
            "content": "",
            "source": "",
            "filename": "",
            "score": 0.0,
            "methods": [],
        })
        
        # Process vector search results
        for rank, (doc, score) in enumerate(vector_docs):
            chunk_id = doc.metadata.get("chunk_id", "")
            if chunk_id:
                rrf_scores[chunk_id]["content"] = doc.page_content
                rrf_scores[chunk_id]["source"] = doc.metadata.get("source", "")
                rrf_scores[chunk_id]["filename"] = doc.metadata.get("filename", "")
                rrf_scores[chunk_id]["score"] += 1.0 / (self.rrf_k + rank + 1)
                if "vector" not in rrf_scores[chunk_id]["methods"]:
                    rrf_scores[chunk_id]["methods"].append("vector")
        
        # Process BM25 results
        for rank, doc in enumerate(bm25_docs):
            chunk_id = doc.metadata.get("chunk_id", "")
            if chunk_id:
                rrf_scores[chunk_id]["content"] = doc.page_content
                rrf_scores[chunk_id]["source"] = doc.metadata.get("source", "")
                rrf_scores[chunk_id]["filename"] = doc.metadata.get("filename", "")
                rrf_scores[chunk_id]["score"] += 1.0 / (self.rrf_k + rank + 1)
                if "bm25" not in rrf_scores[chunk_id]["methods"]:
                    rrf_scores[chunk_id]["methods"].append("bm25")
        
        # Process feedback_qa results (with weight boost)
        if feedback_docs:
            for rank, (doc, score) in enumerate(feedback_docs):
                chunk_id = doc.metadata.get("chunk_id", "")
                if chunk_id:
                    rrf_scores[chunk_id]["content"] = doc.page_content
                    rrf_scores[chunk_id]["source"] = doc.metadata.get("source", "")
                    rrf_scores[chunk_id]["filename"] = doc.metadata.get("filename", "")
                    # Apply feedback weight boost
                    rrf_scores[chunk_id]["score"] += self.feedback_weight / (self.rrf_k + rank + 1)
                    if "feedback" not in rrf_scores[chunk_id]["methods"]:
                        rrf_scores[chunk_id]["methods"].append("feedback")
        
        return rrf_scores
