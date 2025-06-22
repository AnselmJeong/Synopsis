"""TOC (Table of Contents) related data models."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class TOCEntry:
    """TOC 항목을 나타내는 데이터 클래스"""

    title: str
    level: int
    file_path: str
    hierarchy: List[Dict[str, Any]]
    anchor: Optional[str] = None
