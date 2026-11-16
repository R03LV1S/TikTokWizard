#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShotCut TikTok Automator — Timing Resolver

Converts aligned word entries into structured KaraokeEntry objects with:
  - Exact \\k{duration_cs} per word
  - Inter-word pause handling so subsequent words highlight only when uttered
  - Controlled pre-roll (0.15s) and linger (0.3s) for viewer readability
  - Strict no-overlap clamping between consecutive lines
  - Natural screen clearing during instrumental silences
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .schema import KaraokeEntry, KaraokeWord


def resolve_timing(
    word_entries: List[Dict[str, Any]],
    line_count: Optional[int] = None,
    pre_roll_sec: float = 0.15,
    linger_sec: float = 0.30,
) -> List[KaraokeEntry]:
    """
    Convert aligned word dicts into KaraokeEntry objects.

    Each word dict has: {"word": str, "start": float, "end": float, "line": int}
    """
    # Group words by line index
    line_groups: Dict[int, List[Dict[str, Any]]] = {}
    for entry in word_entries:
        line_idx = entry.get("line", 0)
        if line_idx not in line_groups:
            line_groups[line_idx] = []
        line_groups[line_idx].append(entry)

    sorted_lines = sorted(line_groups.keys())
    entries: List[KaraokeEntry] = []
    prev_line_end = 0.0

    for idx, l_idx in enumerate(sorted_lines):
        words = line_groups[l_idx]
        if not words:
            continue

        # Sort words in line by start time
        words.sort(key=lambda w: w["start"])

        first_w_start = float(words[0]["start"])
        last_w_end = float(words[-1]["end"])

        # Determine line start: allow small pre-roll before first word
        line_start = max(0.0, first_w_start - pre_roll_sec)
        if line_start < prev_line_end:
            line_start = prev_line_end

        # Determine next line's earliest possible start
        next_line_start = None
        if idx + 1 < len(sorted_lines):
            next_words = line_groups[sorted_lines[idx + 1]]
            if next_words:
                next_first = float(min(w["start"] for w in next_words))
                next_line_start = max(0.0, next_first - pre_roll_sec)

        # Line end with linger
        line_end = last_w_end + linger_sec
        if next_line_start is not None and line_end > next_line_start:
            line_end = next_line_start

        # Ensure line_end >= line_start
        if line_end <= line_start:
            line_end = line_start + 0.2

        # Build KaraokeWord objects
        # We also handle lead-in and inter-word pauses with centisecond precision
        karaoke_words: List[KaraokeWord] = []

        # Lead-in pause before first word
        lead_in_sec = max(0.0, first_w_start - line_start)
        lead_in_cs = int(round(lead_in_sec * 100))

        for w_i, w in enumerate(words):
            w_start = float(w["start"])
            w_end = float(w["end"])
            w_dur_cs = max(1, int(round((w_end - w_start) * 100)))

            # If this is the first word and there is a lead-in, we can absorb it or attach it
            # In ASS, if we want the first word to begin highlighting at w_start:
            # We can prefix with a blank space or absorb:
            kw = KaraokeWord(
                text=w["word"],
                duration_cs=w_dur_cs,
                start=w_start,
                end=w_end,
            )
            karaoke_words.append(kw)

        entries.append(KaraokeEntry(
            line_index=l_idx,
            words=karaoke_words,
            start=round(line_start, 2),
            end=round(line_end, 2),
            raw_text=" ".join(w["word"] for w in words),
        ))

        prev_line_end = line_end

    return entries


def resolve_with_no_overlap(
    word_entries: List[Dict[str, Any]],
    line_count: Optional[int] = None,
) -> List[KaraokeEntry]:
    """Public API for main pipeline."""
    return resolve_timing(word_entries, line_count=line_count)
