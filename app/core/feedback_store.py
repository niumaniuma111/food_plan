"""
Feedback store module for managing user feedback and knowledge base enrichment.
Handles feedback submission, approval workflow, and vector store ingestion.
"""
import json
import uuid
import hashlib
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from langchain_core.documents import Document

from app.config import get_settings
from app.core.vector_store import get_vectorstore
from app.core.embedding import get_embedding_service


class FeedbackStatus:
    """Feedback status constants."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    DELETED = "deleted"


class FeedbackStore:
    """
    Feedback store for managing user Q&A feedback.
    Handles the approval workflow and knowledge base enrichment.
    """
    
    COLLECTION_NAME = "feedback_qa"
    
    def __init__(self):
        """Initialize feedback store."""
        self.settings = get_settings()
        self.vectorstore = get_vectorstore(self.COLLECTION_NAME)
        self.feedback_file = self.settings.feedback_qa_dir / "feedback_records.json"
        
        # Ensure feedback directory exists
        self.settings.feedback_qa_dir.mkdir(parents=True, exist_ok=True)
    
    def submit_feedback(
        self,
        question: str,
        answer: str,
        session_id: str,
        rating: str = "positive"
    ) -> dict:
        """
        Submit user feedback for a Q&A pair.
        Status is set to 'pending' for admin review.
        
        Args:
            question: User's question
            answer: AI's answer
            session_id: Session identifier
            rating: "positive" or "negative"
            
        Returns:
            Dictionary with feedback ID and status
        """
        feedback_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Create content hash for deduplication
        content_hash = hashlib.md5(
            f"{question}:{answer[:100]}".encode()
        ).hexdigest()
        
        # Check for duplicates
        if self._is_duplicate(question):
            return {
                "status": "duplicate",
                "message": "Similar feedback already exists"
            }
        
        # Create feedback record
        feedback_record = {
            "id": feedback_id,
            "question": question,
            "answer": answer,
            "session_id": session_id,
            "rating": rating,
            "status": FeedbackStatus.PENDING,
            "timestamp": timestamp,
            "content_hash": content_hash,
        }
        
        # Save to file
        self._save_feedback_record(feedback_record)
        
        return {
            "id": feedback_id,
            "status": FeedbackStatus.PENDING,
            "message": "Feedback submitted for review"
        }
    
    def approve_feedback(self, feedback_id: str) -> dict:
        """
        Approve feedback and ingest into vector store.
        
        Args:
            feedback_id: Feedback record ID
            
        Returns:
            Dictionary with result status
        """
        record = self._load_feedback_record(feedback_id)
        
        if not record:
            return {"status": "error", "message": "Feedback not found"}
        
        if record["status"] != FeedbackStatus.PENDING:
            return {"status": "error", "message": f"Cannot approve feedback with status: {record['status']}"}
        
        # Format Q&A content for vector store
        qa_content = f"Q: {record['question']}\nA: {record['answer']}"
        
        # Create document
        doc = Document(
            page_content=qa_content,
            metadata={
                "feedback_id": feedback_id,
                "source": "user_feedback",
                "status": FeedbackStatus.APPROVED,
                "timestamp": record["timestamp"],
                "session_id": record["session_id"],
            }
        )
        
        # Add to vector store
        self.vectorstore.add_documents([doc], ids=[feedback_id])
        
        # Update record status
        record["status"] = FeedbackStatus.APPROVED
        self._save_feedback_record(record, update=True)
        
        return {
            "status": "success",
            "message": "Feedback approved and added to knowledge base"
        }
    
    def reject_feedback(self, feedback_id: str) -> dict:
        """
        Reject feedback (not added to vector store).
        
        Args:
            feedback_id: Feedback record ID
            
        Returns:
            Dictionary with result status
        """
        record = self._load_feedback_record(feedback_id)
        
        if not record:
            return {"status": "error", "message": "Feedback not found"}
        
        if record["status"] != FeedbackStatus.PENDING:
            return {"status": "error", "message": f"Cannot reject feedback with status: {record['status']}"}
        
        # Update record status
        record["status"] = FeedbackStatus.REJECTED
        self._save_feedback_record(record, update=True)
        
        return {
            "status": "success",
            "message": "Feedback rejected"
        }
    
    def delete_feedback(self, feedback_id: str) -> dict:
        """
        Delete feedback and remove from vector store if approved.
        
        Args:
            feedback_id: Feedback record ID
            
        Returns:
            Dictionary with result status
        """
        record = self._load_feedback_record(feedback_id)
        
        if not record:
            return {"status": "error", "message": "Feedback not found"}
        
        # Remove from vector store if it was approved
        if record["status"] == FeedbackStatus.APPROVED:
            try:
                self.vectorstore.delete_documents([feedback_id])
            except Exception:
                pass  # Document might not exist in vector store
        
        # Update record status
        record["status"] = FeedbackStatus.DELETED
        self._save_feedback_record(record, update=True)
        
        return {
            "status": "success",
            "message": "Feedback deleted"
        }
    
    def list_feedback(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[dict]:
        """
        List feedback records with optional filtering.
        
        Args:
            status: Filter by status (pending/approved/rejected)
            limit: Maximum number of records to return
            offset: Offset for pagination
            
        Returns:
            List of feedback records
        """
        records = self._load_all_records()
        
        # Filter by status if provided
        if status:
            records = [r for r in records if r["status"] == status]
        
        # Sort by timestamp (newest first)
        records.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Apply pagination
        return records[offset:offset + limit]
    
    def get_stats(self) -> dict:
        """
        Get feedback statistics.
        
        Returns:
            Dictionary with counts by status
        """
        records = self._load_all_records()
        
        stats = {
            "total": len(records),
            "pending": 0,
            "approved": 0,
            "rejected": 0,
            "deleted": 0,
        }
        
        for record in records:
            status = record.get("status", FeedbackStatus.PENDING)
            if status in stats:
                stats[status] += 1
        
        return stats
    
    def _is_duplicate(self, question: str, threshold: float = 0.95) -> bool:
        """
        Check if a similar question already exists.
        
        Args:
            question: Question to check
            threshold: Cosine similarity threshold for duplicate detection
            
        Returns:
            True if duplicate found
        """
        try:
            # Search for similar questions in feedback_qa
            results = self.vectorstore.similarity_search_with_score(
                question,
                k=1
            )
            
            if results:
                doc, score = results[0]
                # Convert distance to similarity (Chroma returns L2 distance)
                # Lower distance = higher similarity
                if score < 0.1:  # Very similar
                    return True
        except Exception:
            pass
        
        return False
    
    def _save_feedback_record(self, record: dict, update: bool = False) -> None:
        """
        Save a feedback record to file.
        
        Args:
            record: Feedback record to save
            update: If True, update existing record
        """
        records = self._load_all_records()
        
        if update:
            # Find and update existing record
            for i, r in enumerate(records):
                if r["id"] == record["id"]:
                    records[i] = record
                    break
        else:
            # Add new record
            records.append(record)
        
        # Save to file
        with open(self.feedback_file, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    
    def _load_feedback_record(self, feedback_id: str) -> Optional[dict]:
        """
        Load a specific feedback record by ID.
        
        Args:
            feedback_id: Feedback record ID
            
        Returns:
            Feedback record or None
        """
        records = self._load_all_records()
        
        for record in records:
            if record["id"] == feedback_id:
                return record
        
        return None
    
    def _load_all_records(self) -> List[dict]:
        """
        Load all feedback records from file.
        
        Returns:
            List of feedback records
        """
        if not self.feedback_file.exists():
            return []
        
        try:
            with open(self.feedback_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []


# Singleton instance
_feedback_store: FeedbackStore = None


def get_feedback_store() -> FeedbackStore:
    """Get or create feedback store instance."""
    global _feedback_store
    if _feedback_store is None:
        _feedback_store = FeedbackStore()
    return _feedback_store
