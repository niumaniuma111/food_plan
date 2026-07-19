"""
Feedback API router for managing user feedback.
"""
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.feedback_store import get_feedback_store


router = APIRouter(tags=["feedback"])


class FeedbackSubmitRequest(BaseModel):
    """Request model for submitting feedback."""
    question: str
    answer: str
    session_id: str
    rating: str = "positive"  # "positive" or "negative"


class FeedbackListParams(BaseModel):
    """Parameters for listing feedback."""
    status: Optional[str] = None
    limit: int = 50
    offset: int = 0


@router.post("/feedback")
async def submit_feedback(request: FeedbackSubmitRequest):
    """
    Submit user feedback for a Q&A pair.
    Feedback is stored as 'pending' for admin review.
    """
    feedback_store = get_feedback_store()
    
    result = feedback_store.submit_feedback(
        question=request.question,
        answer=request.answer,
        session_id=request.session_id,
        rating=request.rating,
    )
    
    return result


@router.get("/feedback/list")
async def list_feedback(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """
    List feedback records with optional filtering.
    
    Query parameters:
    - status: Filter by status (pending/approved/rejected/deleted)
    - limit: Maximum number of records (default: 50)
    - offset: Offset for pagination (default: 0)
    """
    feedback_store = get_feedback_store()
    
    records = feedback_store.list_feedback(
        status=status,
        limit=limit,
        offset=offset,
    )
    
    return {
        "records": records,
        "count": len(records),
    }


@router.put("/feedback/{feedback_id}/approve")
async def approve_feedback(feedback_id: str):
    """
    Approve feedback and add to knowledge base.
    The Q&A pair will be embedded and added to the vector store.
    """
    feedback_store = get_feedback_store()
    
    result = feedback_store.approve_feedback(feedback_id)
    
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result


@router.put("/feedback/{feedback_id}/reject")
async def reject_feedback(feedback_id: str):
    """
    Reject feedback. The Q&A pair will not be added to the knowledge base.
    """
    feedback_store = get_feedback_store()
    
    result = feedback_store.reject_feedback(feedback_id)
    
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result


@router.delete("/feedback/{feedback_id}")
async def delete_feedback(feedback_id: str):
    """
    Delete feedback record.
    If the feedback was approved, it will also be removed from the vector store.
    """
    feedback_store = get_feedback_store()
    
    result = feedback_store.delete_feedback(feedback_id)
    
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result


@router.get("/feedback/stats")
async def get_feedback_stats():
    """
    Get feedback statistics.
    Returns counts by status (pending/approved/rejected/deleted).
    """
    feedback_store = get_feedback_store()
    
    return feedback_store.get_stats()
