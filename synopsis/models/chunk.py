"""Content chunk related data models."""

from typing import Dict, Any, Optional
from dataclasses import dataclass

from .toc import TOCEntry


@dataclass
class ContentChunk:
    """콘텐츠 청크를 나타내는 데이터 클래스"""

    content: str
    toc_entry: TOCEntry
    chunk_index: int
    total_chunks: int
    section_info: Optional[Dict[str, Any]] = None
