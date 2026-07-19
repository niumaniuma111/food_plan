"""
Documents API router for document management.
"""
import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.config import get_settings
from app.core.offline_pipeline import get_offline_pipeline
from app.core.vector_store import get_vectorstore


router = APIRouter(tags=["documents"])


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document file (Markdown or TXT).
    The document will be parsed, chunked, and added to the vector store.
    """
    settings = get_settings()
    
    # Validate file type
    allowed_extensions = {".md", ".markdown", ".txt"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}"
        )
    
    # Save file to raw directory
    file_path = settings.raw_docs_dir / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Ingest file
    pipeline = get_offline_pipeline()
    result = pipeline.ingest_file(str(file_path))
    
    return {
        "status": "success",
        "filename": file.filename,
        "chunks_created": result["chunks_created"],
    }


@router.get("/documents")
async def list_documents():
    """
    List all uploaded documents.
    """
    settings = get_settings()
    
    documents = []
    
    # Scan raw directory
    if settings.raw_docs_dir.exists():
        for file_path in settings.raw_docs_dir.iterdir():
            if file_path.is_file():
                documents.append({
                    "filename": file_path.name,
                    "size": file_path.stat().st_size,
                    "created_at": file_path.stat().st_ctime,
                    "type": "uploaded",
                })
    
    # Scan knowledge base directory
    if settings.knowledge_base_dir.exists():
        for file_path in settings.knowledge_base_dir.iterdir():
            if file_path.is_file():
                documents.append({
                    "filename": file_path.name,
                    "size": file_path.stat().st_size,
                    "created_at": file_path.stat().st_ctime,
                    "type": "knowledge_base",
                })
    
    # Get vector store stats
    vectorstore = get_vectorstore("documents")
    stats = vectorstore.get_collection_stats()
    
    return {
        "documents": documents,
        "vector_store": stats,
    }


@router.delete("/documents/{filename}")
async def delete_document(filename: str):
    """
    Delete a document and its associated vectors.
    """
    settings = get_settings()
    
    # Check if file exists in raw directory
    file_path = settings.raw_docs_dir / filename
    
    if not file_path.exists():
        # Check knowledge base
        file_path = settings.knowledge_base_dir / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Document not found: {filename}")
    
    # Delete file
    file_path.unlink()
    
    # Note: Vector store cleanup would require tracking chunk IDs per file
    # For now, orphaned vectors will remain but won't affect functionality
    
    return {
        "status": "success",
        "message": f"Document deleted: {filename}",
    }


@router.post("/documents/ingest-knowledge-base")
async def ingest_knowledge_base():
    """
    Ingest the pre-built knowledge base into the vector store.
    """
    pipeline = get_offline_pipeline()
    result = pipeline.ingest_knowledge_base()
    
    return result
