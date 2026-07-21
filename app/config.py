"""
Application configuration management.
Loads settings from .env and config.yaml.
"""
import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class LLMConfig(BaseModel):
    model: str = "qwen-plus"
    temperature: float = 0.7
    api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class EmbeddingConfig(BaseModel):
    model: str = "text-embedding-v3"
    api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class ChunkingConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 64


class RetrievalConfig(BaseModel):
    vector_top_k: int = 20
    bm25_top_k: int = 20
    rerank_top_k: int = 5


class MemoryConfig(BaseModel):
    turn_limit: int = 10


class ChromaConfig(BaseModel):
    persist_dir: str = "./data/chroma_db"


class Settings(BaseSettings):
    # API Keys
    dashscope_api_key: str = ""
    
    # Sub-configs
    llm: LLMConfig = LLMConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    memory: MemoryConfig = MemoryConfig()
    chroma: ChromaConfig = ChromaConfig()
    
    # Paths
    project_root: Path = PROJECT_ROOT
    knowledge_base_dir: Path = PROJECT_ROOT / "data" / "knowledge_base"
    raw_docs_dir: Path = PROJECT_ROOT / "data" / "raw"
    feedback_qa_dir: Path = PROJECT_ROOT / "data" / "feedback_qa"
    bm25_index_path: Path = PROJECT_ROOT / "data" / "bm25_index.json"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


def load_config() -> Settings:
    """Load configuration from config.yaml and .env"""
    config_path = PROJECT_ROOT / "config.yaml"
    
    yaml_config = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f) or {}
    
    # Get API key from environment
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    
    settings = Settings(
        dashscope_api_key=api_key,
        llm=LLMConfig(**yaml_config.get("llm", {})),
        embedding=EmbeddingConfig(**yaml_config.get("embedding", {})),
        chunking=ChunkingConfig(**yaml_config.get("chunking", {})),
        retrieval=RetrievalConfig(**yaml_config.get("retrieval", {})),
        memory=MemoryConfig(**yaml_config.get("memory", {})),
        chroma=ChromaConfig(**yaml_config.get("chroma", {})),
    )
    
    return settings


# Global settings singleton
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get global settings instance"""
    global _settings
    if _settings is None:
        _settings = load_config()
    return _settings
