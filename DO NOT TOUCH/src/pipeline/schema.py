#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShotCut TikTok Automator — Pipeline Schema

Defines the data model for lyric-first forced alignment karaoke pipeline.
Lyrics are the ground truth; audio is aligned to the text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LyricWord:
    """A single word from the lyrics, with aligned start/end timestamps."""
    text: str                      # The word as written in lyrics.txt
    normalized: str                # Lowercased, punctuation-stripped for matching
    start: Optional[float] = None  # Aligned start seconds from forced alignment
    end: Optional[float] = None    # Aligned end seconds
    confidence: float = 0.0        # Alignment confidence (0.0–1.0)
    matched: bool = False          # Whether directly matched to audio
    matched_to: Optional[str] = None  # Whisper word text that was matched


@dataclass
class LyricLine:
    """A line of lyrics, containing words after alignment."""
    index: int                     # 0-based line number
    text: str                      # Raw line text
    words: List[LyricWord] = field(default_factory=list)  # Words in this line
    start: Optional[float] = None  # Aligned start seconds
    end: Optional[float] = None    # Aligned end seconds


@dataclass
class KaraokeWord:
    """A word within a KaraokeEntry, with a \\k duration tag."""
    text: str
    duration_cs: int               # Duration in centiseconds for \\k tag
    start: float = 0.0             # Absolute start time (seconds)
    end: float = 0.0               # Absolute end time (seconds)


@dataclass
class KaraokeEntry:
    """One ASS Dialogue event row with per-word \\k durations."""
    line_index: int
    words: List[KaraokeWord]       # Words with \\k durations
    start: float                   # Dialogue start (seconds)
    end: float                     # Dialogue end (seconds)
    raw_text: str = ""             # Original line text


@dataclass
class KaraokeMeta:
    """Machine-readable sidecar JSON for each karaoke render."""
    schema_version: str = "shotcut_tiktok.karaoke_meta.v2"
    lyrics_checksum: str = ""
    generated_at: str = ""
    timing_method: str = "forced_alignment_dp"
    word_count: int = 0
    matched_words: int = 0
    alignment_confidence_mean: float = 0.0
    vocal_regions: List[List[float]] = field(default_factory=list)
    total_vocal_seconds: float = 0.0
