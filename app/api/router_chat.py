"""
Chat API router with SSE streaming support.
"""
import json
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.rag_chain import get_rag_chain


router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    query: str
    session_id: str


@router.post("/chat")
async def chat(request: ChatRequest, req: Request):
    """
    Chat endpoint with SSE streaming.
    
    Returns a stream of tokens with sources included in the first chunk.
    """
    # Get BM25 retriever from app state
    bm25_retriever = req.app.state.bm25_retriever
    
    # Get RAG chain
    rag_chain = get_rag_chain(bm25_retriever)
    
    async def event_generator():
        """Generate SSE events for streaming response."""
        try:
            sources_sent = False
            
            async for token, sources in rag_chain.astream(request.query, request.session_id):
                # Build event data
                event_data = {
                    "type": "token",
                    "content": token,
                }
                
                # Include sources in first event
                if not sources_sent and sources:
                    event_data["sources"] = sources
                    sources_sent = True
                
                # Yield SSE formatted event
                yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
            
            # Send done event
            done_event = {"type": "done"}
            yield f"data: {json.dumps(done_event)}\n\n"
            
        except Exception as e:
            # Send error event
            error_event = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@router.post("/memory/clear")
async def clear_memory(session_id: str):
    """
    Clear conversation memory for a session.
    """
    from app.core.memory import get_memory
    
    memory = get_memory()
    result = memory.clear_session(session_id)
    
    return {
        "status": "success" if result else "error",
        "message": "Memory cleared" if result else "Failed to clear memory"
    }
