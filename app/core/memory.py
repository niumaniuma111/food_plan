"""
Short-term memory module using embedding-based storage and retrieval.
Stores conversation history in Chroma for semantic retrieval.
"""
import uuid
from datetime import datetime
from typing import List, Optional

from langchain_core.documents import Document

from app.config import get_settings
from app.core.vector_store import get_vectorstore


class ShortTermMemory:
    """
    Short-term memory using embedding-based storage.
    Stores conversation turns and retrieves semantically relevant history.
    """
    
    COLLECTION_NAME = "conversation_memory"
    
    def __init__(self):
        """Initialize short-term memory."""
        self.settings = get_settings()
        self.turn_limit = self.settings.memory.turn_limit
        self.vectorstore = get_vectorstore(self.COLLECTION_NAME)
    
    def store_turn(
        self,
        session_id: str,
        user_message: str,
        ai_message: str,
        turn_number: int
    ) -> dict:
        """
        Store a conversation turn (user + AI messages).
        
        Args:
            session_id: Session identifier
            user_message: User's message
            ai_message: AI's response
            turn_number: Turn number in the conversation
            
        Returns:
            Dictionary with stored message IDs
        """
        timestamp = datetime.now().isoformat()
        
        # Store user message
        user_doc = Document(
            page_content=user_message,
            metadata={
                "session_id": session_id,
                "turn_number": turn_number,
                "role": "user",
                "timestamp": timestamp,
            }
        )
        
        # Store AI message
        ai_doc = Document(
            page_content=ai_message,
            metadata={
                "session_id": session_id,
                "turn_number": turn_number,
                "role": "assistant",
                "timestamp": timestamp,
            }
        )
        
        # Add to vector store
        ids = self.vectorstore.add_documents([user_doc, ai_doc])
        
        # Enforce turn limit
        self._enforce_turn_limit(session_id)
        
        return {
            "user_id": ids[0] if len(ids) > 0 else None,
            "ai_id": ids[1] if len(ids) > 1 else None,
        }
    
    def retrieve(
        self,
        session_id: str,
        query: str,
        top_k: int = 3
    ) -> List[Document]:
        """
        Retrieve semantically relevant conversation history.
        
        Args:
            session_id: Session identifier
            query: Current query to match against history
            top_k: Number of relevant turns to retrieve
            
        Returns:
            List of Document objects with conversation history
        """
        # Search for relevant messages
        results = self.vectorstore.similarity_search(
            query,
            k=top_k * 2,  # Get more to filter by session
            filter={"session_id": session_id}
        )
        
        # Sort by turn number to maintain conversation order
        results.sort(key=lambda x: x.metadata.get("turn_number", 0))
        
        return results[:top_k]
    
    def get_recent_turns(
        self,
        session_id: str,
        n_turns: int = 3
    ) -> List[dict]:
        """
        Get the most recent conversation turns.
        
        Args:
            session_id: Session identifier
            n_turns: Number of recent turns to retrieve
            
        Returns:
            List of turn dictionaries with user/ai messages
        """
        # Get all messages for this session
        collection = self.vectorstore.vectorstore._collection
        results = collection.get(
            where={"session_id": session_id},
            include=["documents", "metadatas"]
        )
        
        if not results["ids"]:
            return []
        
        # Group by turn number
        turns = {}
        for doc, meta in zip(results["documents"], results["metadatas"]):
            turn_num = meta.get("turn_number", 0)
            if turn_num not in turns:
                turns[turn_num] = {}
            
            role = meta.get("role", "unknown")
            turns[turn_num][role] = doc
        
        # Sort by turn number and get recent
        sorted_turns = sorted(turns.items(), key=lambda x: x[0], reverse=True)
        recent = sorted_turns[:n_turns]
        recent.sort(key=lambda x: x[0])  # Re-sort chronologically
        
        return [{"turn_number": t[0], **t[1]} for t in recent]
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear all memory for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful
        """
        collection = self.vectorstore.vectorstore._collection
        
        # Get all IDs for this session
        results = collection.get(where={"session_id": session_id})
        
        if results["ids"]:
            collection.delete(ids=results["ids"])
        
        return True
    
    def _enforce_turn_limit(self, session_id: str) -> None:
        """
        Remove oldest turns if exceeding turn limit.
        
        Args:
            session_id: Session identifier
        """
        collection = self.vectorstore.vectorstore._collection
        
        # Get all messages for this session
        results = collection.get(
            where={"session_id": session_id},
            include=["metadatas"]
        )
        
        if not results["ids"]:
            return
        
        # Find unique turn numbers
        turn_numbers = set()
        for meta in results["metadatas"]:
            turn_numbers.add(meta.get("turn_number", 0))
        
        # If within limit, nothing to do
        if len(turn_numbers) <= self.turn_limit:
            return
        
        # Find turns to remove (oldest)
        sorted_turns = sorted(turn_numbers)
        turns_to_remove = set(sorted_turns[:len(sorted_turns) - self.turn_limit])
        
        # Find IDs to delete
        ids_to_delete = []
        for doc_id, meta in zip(results["ids"], results["metadatas"]):
            if meta.get("turn_number", 0) in turns_to_remove:
                ids_to_delete.append(doc_id)
        
        # Delete old turns
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)


# Singleton instance
_memory: ShortTermMemory = None


def get_memory() -> ShortTermMemory:
    """Get or create memory instance."""
    global _memory
    if _memory is None:
        _memory = ShortTermMemory()
    return _memory
