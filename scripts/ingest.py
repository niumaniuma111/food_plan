"""
Script to ingest documents into the vector store.
Usage: python -m scripts.ingest [--knowledge-base | --file FILE | --dir DIR]
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.config import get_settings
from app.core.offline_pipeline import get_offline_pipeline


def ingest_knowledge_base():
    """Ingest the pre-built knowledge base."""
    print("🔄 Ingesting knowledge base...")
    pipeline = get_offline_pipeline()
    result = pipeline.ingest_knowledge_base()
    print(f"✅ Knowledge base ingested: {result}")
    return result


def ingest_file(file_path: str):
    """Ingest a single file."""
    print(f"🔄 Ingesting file: {file_path}")
    pipeline = get_offline_pipeline()
    result = pipeline.ingest_file(file_path)
    print(f"✅ File ingested: {result}")
    return result


def ingest_directory(dir_path: str):
    """Ingest all files in a directory."""
    print(f"🔄 Ingesting directory: {dir_path}")
    pipeline = get_offline_pipeline()
    result = pipeline.ingest_directory(dir_path)
    print(f"✅ Directory ingested: {result}")
    return result


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest documents into vector store")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--knowledge-base", action="store_true", help="Ingest knowledge base")
    group.add_argument("--file", type=str, help="Ingest a single file")
    group.add_argument("--dir", type=str, help="Ingest all files in a directory")
    
    args = parser.parse_args()
    
    # Ensure API key is set
    settings = get_settings()
    if not settings.dashscope_api_key:
        print("❌ Error: DASHSCOPE_API_KEY not set in .env file")
        sys.exit(1)
    
    if args.knowledge_base:
        ingest_knowledge_base()
    elif args.file:
        ingest_file(args.file)
    elif args.dir:
        ingest_directory(args.dir)


if __name__ == "__main__":
    main()
