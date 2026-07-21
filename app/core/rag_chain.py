"""
RAG chain orchestration.
Combines memory retrieval, hybrid search, reranking, and LLM generation.
"""
import asyncio
from typing import AsyncGenerator, List, Tuple

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable, trace

from app.config import get_settings
from app.core.memory import get_memory
from app.core.hybrid_retriever import HybridRetriever
from app.core.reranker import get_reranker
from app.core.prompt_builder import PromptBuilder
from app.core.bm25_retriever import BM25Retriever


class RAGChain:
    """
    RAG chain orchestrator.
    Handles the full pipeline: memory → retrieval → reranking → generation.
    """
    
    def __init__(self, bm25_retriever: BM25Retriever):
        """
        Initialize RAG chain.
        
        Args:
            bm25_retriever: BM25 retriever instance
        """
        self.settings = get_settings()
        self.memory = get_memory()
        self.hybrid_retriever = HybridRetriever(bm25_retriever)
        self.reranker = get_reranker()
        self.prompt_builder = PromptBuilder()
        
        # Initialize LLM with streaming support
        self.llm = ChatOpenAI(
            model=self.settings.llm.model,
            temperature=self.settings.llm.temperature,
            api_key=self.settings.dashscope_api_key,
            base_url=self.settings.llm.api_base,
            streaming=True,
        )
    
    @traceable(name="LLM Generation", run_type="llm")
    async def _call_llm(self, messages):
        """Call LLM with tracing."""
        return self.llm.astream(messages)
    
    async def astream(
        self,
        query: str,
        session_id: str
    ) -> AsyncGenerator[Tuple[str, List[dict]], None]:
        """
        Stream RAG response with sources.
        
        Args:
            query: User's question
            session_id: Session identifier
            
        Yields:
            Tuple of (token, sources) where sources is only included in first yield
        """
        with trace(name="RAG Pipeline", run_type="chain", inputs={"query": query, "session_id": session_id}) as parent_run:
            # 1. Retrieve documents (synchronous, direct call to preserve trace context)
            memory_docs = self.memory.retrieve(session_id, query, 3)
            
            retrieved_docs = self.hybrid_retriever.retrieve(query)
            
            ranked_docs = self.reranker.rerank(query, retrieved_docs)
            
            # 2. Build prompt
            prompt = self.prompt_builder.build(query, ranked_docs, memory_docs)
            
            # 3. Format sources for frontend
            sources = self.prompt_builder.format_sources(ranked_docs)
            
            # 4. Stream LLM response
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=query),
            ]
            
            full_response = ""
            first_chunk = True
            
            async for chunk in await self._call_llm(messages):
                token = chunk.content
                if not token:
                    continue
                full_response += token
                
                if first_chunk:
                    yield token, sources
                    first_chunk = False
                else:
                    yield token, []
            
            # Record final output on parent trace
            parent_run.outputs = {"response": full_response, "sources": sources}
        
        # 5. Store conversation turn in memory (background)
        try:
            turn_number = self._get_next_turn_number(session_id)
            await asyncio.to_thread(
                self.memory.store_turn, session_id, query, full_response, turn_number
            )
        except Exception:
            pass  # Don't fail if memory storage fails
    
    def _get_next_turn_number(self, session_id: str) -> int:
        """Get the next turn number for a session."""
        recent_turns = self.memory.get_recent_turns(session_id, n_turns=100)
        if not recent_turns:
            return 1
        return max(t["turn_number"] for t in recent_turns) + 1
    
    def memory_retrieve(self, session_id: str, query: str, top_k: int = 3) -> List[Document]:
        """Retrieve relevant conversation history."""
        return self.memory.retrieve(session_id, query, top_k)
    
    def memory_store_turn(self, session_id: str, user_message: str, ai_message: str) -> None:
        """Store a conversation turn in memory."""
        turn_number = self._get_next_turn_number(session_id)
        self.memory.store_turn(session_id, user_message, ai_message, turn_number)


# Singleton instance
_rag_chain: RAGChain = None


def get_rag_chain(bm25_retriever: BM25Retriever = None) -> RAGChain:
    """Get or create RAG chain instance."""
    global _rag_chain
    if _rag_chain is None and bm25_retriever is not None:
        _rag_chain = RAGChain(bm25_retriever)
    return _rag_chain
