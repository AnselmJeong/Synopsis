"""Core functionality for Synopsis package."""

from .processor import EPUBProcessor
from .search import SearchSystem, SearchType, SearchResult
from .database import setup_database, check_database_status
from .config import Config, validate_config

__all__ = [
    "EPUBProcessor",
    "SearchSystem",
    "SearchType",
    "SearchResult",
    "Config",
    "setup_database",
    "check_database_status",
    "validate_config",
]
