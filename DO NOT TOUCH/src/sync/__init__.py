# __init__.py
"""Synchronization module for TikTok Wizard (v4.1).

Exports Holy-Grail UTF-8 lyric ingestion, monotonic ASR-to-lyric line window
synchronization, and recursive Surround-and-Fill discrepancy resolution.
"""

from __future__ import annotations

from .discrepancy_resolver import DiscrepancyResolver
from .ingestor import UserLyricWord, parse_user_lyric, romanize_word
from .line_synchronizer import (
    MonotonicAlignmentMap,
    align_words_monotonic,
    build_line_windows,
    normalize_token,
    transcribe_vocal,
)

__all__ = [
    "DiscrepancyResolver",
    "MonotonicAlignmentMap",
    "UserLyricWord",
    "align_words_monotonic",
    "build_line_windows",
    "normalize_token",
    "parse_user_lyric",
    "romanize_word",
    "transcribe_vocal",
]
