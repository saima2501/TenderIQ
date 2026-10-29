"""
backend.database package
Part of Phase 4: Baseline Demo Pipeline
"""

from .db_manager import (
    DatabaseManager,
    DB_FILE_PATH
)

__all__ = [
    "DatabaseManager",
    "DB_FILE_PATH"
]
