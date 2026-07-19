"""
Script to clear the vector database.
Usage: python -m scripts.clear_db [--all | --collection NAME]
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import chromadb
from app.config import get_settings


def clear_all():
    """Clear all collections."""
    settings = get_settings()
    
    print("🔄 Clearing all collections...")
    
    client = chromadb.PersistentClient(path=settings.chroma.persist_dir)
    
    # List all collections
    collections = client.list_collections()
    
    for collection in collections:
        print(f"  - Deleting collection: {collection.name}")
        client.delete_collection(collection.name)
    
    print("✅ All collections cleared")


def clear_collection(collection_name: str):
    """Clear a specific collection."""
    settings = get_settings()
    
    print(f"🔄 Clearing collection: {collection_name}")
    
    client = chromadb.PersistentClient(path=settings.chroma.persist_dir)
    
    try:
        client.delete_collection(collection_name)
        print(f"✅ Collection '{collection_name}' cleared")
    except ValueError:
        print(f"⚠️ Collection '{collection_name}' not found")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clear vector database")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Clear all collections")
    group.add_argument("--collection", type=str, help="Clear a specific collection")
    
    args = parser.parse_args()
    
    if args.all:
        clear_all()
    elif args.collection:
        clear_collection(args.collection)


if __name__ == "__main__":
    main()
