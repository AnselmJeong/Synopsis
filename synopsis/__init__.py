"""
Synopsis - EPUB 분석 및 검색 시스템

계층적 TOC 구조를 추출하고 지능적 청킹으로 데이터베이스에 저장하여
의미적 검색과 키워드 검색을 지원하는 패키지입니다.
"""

__version__ = "0.1.0"
__author__ = "Synopsis Team"
__email__ = "synopsis@example.com"

# Core classes and functions
from .core.processor import EPUBProcessor
from .core.search import SearchSystem, SearchType, SearchResult
from .core.database import setup_database, check_database_status
from .core.config import Config, validate_config

# Data models
from .models.toc import TOCEntry
from .models.chunk import ContentChunk

# Utilities
from .utils.format import (
    format_search_results,
    format_hierarchy_path,
    format_detailed_hierarchy,
    interactive_result_viewer,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    # Core classes
    "EPUBProcessor",
    "SearchSystem",
    "SearchType",
    "SearchResult",
    "Config",
    # Data models
    "TOCEntry",
    "ContentChunk",
    # Database functions
    "setup_database",
    "check_database_status",
    "validate_config",
    # Utilities
    "format_search_results",
    "format_hierarchy_path",
    "format_detailed_hierarchy",
    "interactive_result_viewer",
]
