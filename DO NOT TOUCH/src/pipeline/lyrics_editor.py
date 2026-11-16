"""Bidirectional human-readable sync sheet and ASS karaoke parser (v4.1).

Preserves 100% of user surface characters (including parentheses `( ... )` and
UTF-8 characters like `'enseñarme'`), supports `[MM:SS.cc]` start and end line
tags, and round-trips `(XXcs)` per-word durations without stripping punctuation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .schema import KaraokeEntry, KaraokeWord


def _fmt_mmss_cc(seconds: float) -> str:
    s = max(0.0, float(seconds))
    total_cs = int(round(s * 100.0))
    m = total_cs // 6000
    rem = total_cs % 6000
    sec = rem // 100
    cs = rem % 100
    return f"[{m:02d}:{sec:02d}.{cs:02d}]"


def _parse_ts_to_sec(ts_str: str) -> float:
    parts = ts_str.strip().split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600.0 + int(parts[1]) * 60.0 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60.0 + float(parts[1])
        return float(parts[0])
    except ValueError:
        return 0.0


def parse_ass_to_easy_lyrics(ass_path: Path) -> List[Dict[str, Any]]:
    """Parse an ASS karaoke file into structured lines with words, pauses, and timestamps."""
    ass_path = Path(ass_path)
    if not ass_path.exists():
        return []

    content = ass_path.read_text(encoding="utf-8")
    dialogues: List[Dict[str, Any]] = []

    for line in content.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",", 9)
        if len(parts) < 10:
            continue
        start_sec = _parse_ts_to_sec(parts[1])
        end_sec = _parse_ts_to_sec(parts[2])
        text_raw = parts[9]

        # Strip override positioning/alignment tags like {\an5\pos(X,Y)}
        cleaned = re.sub(r"\{\\an\d+[^}]*\}", "", text_raw)

        # Extract {\k<duration>}Word tokens (including empty pause {\k<gap>} tags)
        tokens = re.findall(r"\{\\k(\d+)\}([^{]*)", cleaned)
        words: List[Dict[str, Any]] = []
        cursor = start_sec

        if tokens:
            for cs_str, word_txt in tokens:
                dur_cs = max(1, int(cs_str))
                dur_sec = float(dur_cs) / 100.0
                w = word_txt.strip()
                if w:
                    words.append(
                        {
                            "word": w,
                            "duration_cs": dur_cs,
                            "start": round(cursor, 4),
                            "end": round(cursor + dur_sec, 4),
                        }
                    )
                cursor += dur_sec
        else:
            # Fallback if static ASS line without {\k} tags
            plain_words = re.sub(r"\{[^}]*\}", "", cleaned).split()
            if plain_words:
                span = max(0.10 * len(plain_words), end_sec - start_sec)
                per_w = span / float(len(plain_words))
                for w in plain_words:
                    dur_cs = max(3, int(round(per_w * 100.0)))
                    words.append(
                        {
                            "word": w,
                            "duration_cs": dur_cs,
                            "start": round(cursor, 4),
                            "end": round(cursor + per_w, 4),
                        }
                    )
                    cursor += per_w

        if words:
            dialogues.append(
                {
                    "start": start_sec,
                    "end": max(end_sec, cursor),
                    "words": words,
                }
            )
    return dialogues


def export_easy_lyrics_file(ass_path: Path, output_file: Path) -> bool:
    """Export an intuitive, human-editable sync file with start/end timestamps and (XXcs) word durations."""
    dialogues = parse_ass_to_easy_lyrics(Path(ass_path))
    if not dialogues:
        return False

    out = [
        "# ==========================================================================",
        "# TIKTOK WIZARD - EASY-EDIT LYRICS & SYNC EDITOR",
        "# ==========================================================================",
        "# HOW TO EDIT THIS FILE:",
        "# 1. You can freely edit words, fix typos, or change punctuation.",
        "# 2. [MM:SS.cc] at the beginning of each line is the start timestamp.",
        "# 3. Optional: (XXcs) after any word sets its sung highlight duration.",
        "#    (100cs = 1.0 second. If you remove (XXcs) or add new words, TikTok Wizard",
        "#     automatically balances durations smoothly across the line!)",
        "# 4. Save this file! TikTok Wizard automatically detects your edits and updates",
        "#    the render without needing to touch complex .ass or .json files.",
        "# ==========================================================================\n",
    ]

    for d in dialogues:
        ts_start = _fmt_mmss_cc(float(d["start"]))
        ts_end = _fmt_mmss_cc(float(d["end"]))
        word_parts = [f"{w['word']} ({int(w['duration_cs']):2d}cs)" for w in d["words"]]
        out.append(f"{ts_start} {' '.join(word_parts)} {ts_end}")

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")
    return True


def import_easy_lyrics_to_entries(
    easy_lyrics_path: Path, default_line_dur: float = 2.5
) -> List[KaraokeEntry]:
    """Parse a synced lyrics file into KaraokeEntry objects while preserving parentheses and UTF-8."""
    easy_lyrics_path = Path(easy_lyrics_path)
    if not easy_lyrics_path.exists():
        return []

    raw_lines = easy_lyrics_path.read_text(encoding="utf-8-sig").splitlines()
    entries: List[KaraokeEntry] = []

    line_counter = 0
    for line in raw_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Extract leading [MM:SS.cc] timestamp if present
        m_start = re.match(r"^\[([0-9:.]+)\]\s*(.*)$", line)
        start_sec = 0.0
        rest = line
        if m_start:
            start_sec = _parse_ts_to_sec(m_start.group(1))
            rest = m_start.group(2).strip()

        # Extract optional trailing [MM:SS.cc] timestamp if present
        explicit_end_sec: Optional[float] = None
        m_end = re.search(r"\s*\[([0-9:.]+)\]\s*$", rest)
        if m_end:
            explicit_end_sec = _parse_ts_to_sec(m_end.group(1))
            rest = rest[: m_end.start()].strip()

        # Match `word (XXcs)` or standalone `word` WITHOUT stripping parentheses from `(Gracias,` or `gracias)`
        pattern = re.compile(r"(\S+?)\s*\(\s*(\d+)cs\)|(\S+)")
        raw_words: List[Tuple[str, Optional[int]]] = []
        for match in pattern.finditer(rest):
            if match.group(1) is not None:
                w_text = match.group(1)
                cs_val: Optional[int] = int(match.group(2))
            else:
                w_text = match.group(3)
                cs_val = None
            if w_text:
                raw_words.append((w_text, cs_val))

        if not raw_words:
            continue

        assigned_cs = [c for _, c in raw_words if c is not None]
        total_assigned_cs = sum(assigned_cs)
        unassigned_count = sum(1 for _, c in raw_words if c is None)

        target_line_dur = (
            max(0.20, explicit_end_sec - start_sec)
            if explicit_end_sec is not None and explicit_end_sec > start_sec
            else default_line_dur
        )
        default_dur_cs = int(round(target_line_dur * 100.0))

        if unassigned_count > 0:
            remaining_cs = max(unassigned_count * 12, default_dur_cs - total_assigned_cs)
            per_w_cs = max(10, remaining_cs // unassigned_count)
        else:
            per_w_cs = 25

        k_words: List[KaraokeWord] = []
        curr_time = start_sec
        for w_text, cs_val in raw_words:
            dur_cs = max(3, cs_val if cs_val is not None else per_w_cs)
            dur_sec = float(dur_cs) / 100.0
            k_words.append(
                KaraokeWord(
                    text=w_text,
                    duration_cs=dur_cs,
                    start=round(curr_time, 4),
                    end=round(curr_time + dur_sec, 4),
                )
            )
            curr_time += dur_sec

        line_end = (
            max(curr_time, explicit_end_sec)
            if explicit_end_sec is not None
            else curr_time
        )
        entries.append(
            KaraokeEntry(
                line_index=line_counter,
                words=k_words,
                start=round(start_sec, 4),
                end=round(line_end, 4),
                raw_text=" ".join(w.text for w in k_words),
            )
        )
        line_counter += 1

    return entries
