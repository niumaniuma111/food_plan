"""
Utility helper functions.
"""
import hashlib
import uuid
from datetime import datetime


def generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())


def content_hash(text: str) -> str:
    """Generate MD5 hash for content deduplication."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def timestamp_now() -> str:
    """Get current ISO format timestamp."""
    return datetime.now().isoformat()


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to max_length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."
