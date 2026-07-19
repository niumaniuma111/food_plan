"""
Pydantic schemas for API request/response models.
"""
from typing import Optional, List
from pydantic import BaseModel


# Chat schemas
class ChatRequest(BaseModel):
    query: str
    session_id: str


class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[dict]] = None


# Document schemas
class DocumentInfo(BaseModel):
    filename: str
    size: int
    created_at: float
    type: str


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    vector_store: dict


# Feedback schemas
class FeedbackSubmitRequest(BaseModel):
    question: str
    answer: str
    session_id: str
    rating: str = "positive"


class FeedbackRecord(BaseModel):
    id: str
    question: str
    answer: str
    session_id: str
    rating: str
    status: str
    timestamp: str


class FeedbackListResponse(BaseModel):
    records: List[FeedbackRecord]
    count: int


class FeedbackStatsResponse(BaseModel):
    total: int
    pending: int
    approved: int
    rejected: int
    deleted: int


# Memory schemas
class ClearMemoryRequest(BaseModel):
    session_id: str
