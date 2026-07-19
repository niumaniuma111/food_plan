"""
Document parser for Markdown and TXT files.
"""
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader


class DocumentParser:
    """Parse Markdown and TXT files into Document objects."""
    
    SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt"}
    
    @staticmethod
    def parse_file(file_path: str) -> List[Document]:
        """
        Parse a single file into Document objects.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of Document objects
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = path.suffix.lower()
        
        if ext not in DocumentParser.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}")
        
        # Load document based on file type
        if ext in {".md", ".markdown"}:
            loader = UnstructuredMarkdownLoader(str(file_path))
        else:  # .txt
            loader = TextLoader(str(file_path), encoding="utf-8")
        
        documents = loader.load()
        
        # Add metadata
        for doc in documents:
            doc.metadata["source"] = str(file_path)
            doc.metadata["filename"] = path.name
            doc.metadata["file_type"] = ext[1:]  # Remove the dot
        
        return documents
    
    @staticmethod
    def parse_directory(dir_path: str) -> List[Document]:
        """
        Parse all supported files in a directory.
        
        Args:
            dir_path: Path to the directory
            
        Returns:
            List of Document objects
        """
        path = Path(dir_path)
        
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Directory not found: {dir_path}")
        
        all_documents = []
        
        for ext in DocumentParser.SUPPORTED_EXTENSIONS:
            for file_path in path.glob(f"*{ext}"):
                try:
                    docs = DocumentParser.parse_file(str(file_path))
                    all_documents.extend(docs)
                except Exception as e:
                    print(f"Warning: Failed to parse {file_path}: {e}")
        
        return all_documents
