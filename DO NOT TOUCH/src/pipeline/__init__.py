"""Pipeline package for TikTok Wizard (v4.1)."""

from __future__ import annotations

import os

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from ..align.integrity_verifier import IntegrityVerifier
from ..align.pipeline import UniversalEngine
from .ass_writer import ass_timestamp, write_ass_karaoke
from .lyrics_editor import (
    export_easy_lyrics_file,
    import_easy_lyrics_to_entries,
    parse_ass_to_easy_lyrics,
)
from .render import build_ffmpeg_command, find_default_background
from .schema import KaraokeEntry, KaraokeMeta, KaraokeWord, LyricLine, LyricWord

__all__ = [
    "IntegrityVerifier",
    "KaraokeEntry",
    "KaraokeMeta",
    "KaraokeWord",
    "LyricLine",
    "LyricWord",
    "UniversalEngine",
    "ass_timestamp",
    "build_ffmpeg_command",
    "export_easy_lyrics_file",
    "find_default_background",
    "import_easy_lyrics_to_entries",
    "parse_ass_to_easy_lyrics",
    "write_ass_karaoke",
]
