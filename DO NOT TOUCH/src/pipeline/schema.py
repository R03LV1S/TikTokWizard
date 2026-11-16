#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShotCut TikTok Automator — Pipeline Schema (v4.1)

Defines the immutable data contracts for the Holy-Grail lyric-first forced
alignment karaoke pipeline. User lyrics are the ground truth; audio is aligned
to the text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LyricWord:
    """A single word from the user's lyrics, with aligned start/end timestamps."""

    text: str                           # Immutable surface word as written in <lyrics>.txt
    normalized: str                     # Lowercased, accent-folded key for matching
    start: Optional[float] = None       # Aligned start seconds
    end: Optional[float] = None         # Aligned end seconds
    confidence: float = 0.0             # Calibrated acoustic confidence (0.0-1.0)
    matched: bool = False               # Whether directly matched/cemented
    matched_to: Optional[str] = None    # Diagnostic transcript token if applicable
    is_parenthetical: bool = False      # True if inside (...) background/echo span
    is_cemented: bool = False           # True once locked at >= word_target (0.95)
    duration_cs: int = 0                # Duration in centiseconds
    vocal_effect: str = "NONE"          # NONE, STUTTER, ECHO_THROW, HELD_NOTE, REVERB_TAIL


@dataclass
class LyricLine:
    """A line of lyrics, containing words in strict 1:1 line order."""

    index: int                          # 0-based line number
    text: str                           # Immutable raw line text
    words: List[LyricWord] = field(default_factory=list)
    start: Optional[float] = None       # Aligned start seconds
    end: Optional[float] = None         # Aligned end seconds
    is_parenthetical: bool = False


@dataclass
class KaraokeWord:
    """A word within a KaraokeEntry, with a \\k duration tag in centiseconds."""

    text: str
    duration_cs: int                    # Duration in centiseconds for \k tag
    start: float = 0.0                  # Absolute start time (seconds)
    end: float = 0.0                    # Absolute end time (seconds)
    confidence: float = 1.0
    is_parenthetical: bool = False


@dataclass
class KaraokeEntry:
    """One ASS Dialogue event row (strictly 1 per input lyric line)."""

    line_index: int
    words: List[KaraokeWord]            # Words with \k durations
    start: float                        # Dialogue start (seconds)
    end: float                          # Dialogue end (seconds)
    raw_text: str = ""                  # Original line text


@dataclass
class KaraokeMeta:
    """Machine-readable sidecar JSON for each karaoke alignment/render."""

    schema_version: str = "shotcut_tiktok.karaoke_meta.v4_1"
    lyrics_checksum: str = ""
    generated_at: str = ""
    timing_method: str = "greenfield_v4_1_real_mms_fa"
    word_count: int = 0
    matched_words: int = 0
    cemented_words_ge_95_count: int = 0
    uncemented_words_count: int = 0
    alignment_confidence_mean: float = 0.0
    tau_song: float = 0.23
    vocal_regions: List[List[float]] = field(default_factory=list)
    total_vocal_seconds: float = 0.0
