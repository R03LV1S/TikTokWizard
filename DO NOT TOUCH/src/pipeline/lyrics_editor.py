#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TikTok Wizard — Dual Machine & Human-Readable Sync & Highlight Timings Editor (v5.7)

Features:
1. Section 1 (Machine-Readable): Full precision JSON containing line onsets/offsets,
   word timestamps, durations, posteriors, and model confidence scores.
2. Section 2 (Human-Readable): Intuitive, easy-to-fix plain text format:
     Line 1: [00:14.79 -> 00:17.35] If (21cs) you (33cs) want (26cs) love (96cs)
   Allows creators to edit words, typos, line timestamps, or highlight durations directly
   in any text editor (Notepad, VS Code, etc.).
3. Adaptive Duration Balancing: If durations are omitted or words added, timings
   are automatically and smoothly balanced across the line interval.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .schema import KaraokeEntry, KaraokeWord


def to_seconds(ts_str: str) -> float:
    """Convert timestamp string (HH:MM:SS.cc or MM:SS.cc or SS.cc) to float seconds."""
    parts = ts_str.strip().split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600.0 + int(parts[1]) * 60.0 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60.0 + float(parts[1])
        return float(parts[0])
    except (ValueError, IndexError):
        return 0.0


def format_timestamp_raw(seconds: float) -> str:
    """Format seconds into MM:SS.cc without outer brackets."""
    s = max(0.0, float(seconds))
    total_cs = int(round(s * 100.0))
    m = total_cs // 6000
    rem = total_cs % 6000
    sec = rem // 100
    cs = rem % 100
    return f"{m:02d}:{sec:02d}.{cs:02d}"


def format_timestamp(seconds: float) -> str:
    """Format seconds into [MM:SS.cc] string."""
    return f"[{format_timestamp_raw(seconds)}]"


def parse_ass_to_easy_lyrics(ass_path: Path) -> List[Dict[str, Any]]:
    """
    Parse an ASS karaoke file into structured lines with words and timestamps.
    """
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
        start_sec = to_seconds(parts[1].strip())
        end_sec = to_seconds(parts[2].strip())
        text_raw = parts[9]

        # Strip standard styling tags {\an5\pos(X,Y)}
        cleaned = re.sub(r"^{\\an\d+\\pos\([^)]+\)}", "", text_raw)

        # Match {\k<duration>}Word tokens
        tokens = re.findall(r"\{\\k(\d+)\}([^{]*)", cleaned)
        words: List[Dict[str, Any]] = []
        curr_time = start_sec
        for cs_str, word_txt in tokens:
            dur_cs = int(cs_str)
            w = word_txt.strip()
            if w:
                dur_sec = dur_cs / 100.0
                words.append({
                    "word": w,
                    "duration_cs": dur_cs,
                    "start_sec": round(curr_time, 3),
                    "end_sec": round(curr_time + dur_sec, 3),
                })
                curr_time += dur_sec

        if words:
            dialogues.append({
                "start": start_sec,
                "end": max(end_sec, curr_time),
                "words": words,
            })
    return dialogues


def export_sync_editor_file(
    ass_path: Path,
    output_file: Path,
    manifest: Optional[Dict[str, Any]] = None,
    project_name: str = "",
) -> bool:
    """
    Export a dual machine-readable and human-readable companion editor file.
    Section 1 contains full JSON metadata with timestamps and confidences.
    Section 2 contains human-readable easy-to-fix lines for manual corrections.
    """
    lines_data: List[Dict[str, Any]] = []

    # Priority 1: Use manifest data if available
    if manifest and "lines" in manifest and "words" in manifest:
        raw_words = manifest.get("words", [])
        for li in manifest.get("lines", []):
            l_idx = int(li.get("index", 1)) - 1
            l_start = float(li.get("start_sec", 0.0))
            l_end = float(li.get("end_sec", l_start + 2.0))
            l_text = str(li.get("text", ""))

            l_words = [w for w in raw_words if int(w.get("line_index", -1)) == l_idx]
            word_entries: List[Dict[str, Any]] = []
            for w in l_words:
                w_st = float(w.get("start_sec", l_start))
                w_en = float(w.get("end_sec", w_st + 0.20))
                dur_cs = int(round((w_en - w_st) * 100.0))
                word_entries.append({
                    "word": str(w.get("word", "")),
                    "start_sec": round(w_st, 3),
                    "end_sec": round(w_en, 3),
                    "duration_cs": max(1, dur_cs),
                    "posterior": round(float(w.get("posterior", 0.9)), 3),
                    "confidence": round(float(w.get("confidence", 0.9)), 3),
                })
            lines_data.append({
                "line_index": l_idx,
                "start_sec": round(l_start, 3),
                "end_sec": round(l_end, 3),
                "text": l_text,
                "words": word_entries,
            })

    # Priority 2: Reconstruct from ASS file if manifest not provided or empty
    if not lines_data and ass_path.exists():
        parsed_dialogues = parse_ass_to_easy_lyrics(ass_path)
        for idx, d in enumerate(parsed_dialogues):
            lines_data.append({
                "line_index": idx,
                "start_sec": round(float(d["start"]), 3),
                "end_sec": round(float(d["end"]), 3),
                "text": " ".join(w["word"] for w in d["words"]),
                "words": d["words"],
            })

    if not lines_data:
        return False

    proj_name = project_name or output_file.parent.name or "TikTok Wizard Project"
    total_words = sum(len(l["words"]) for l in lines_data)

    # Prepare Section 1: Machine-Readable JSON
    machine_payload = {
        "project": proj_name,
        "engine_version": "5.7",
        "schema_version": "tiktok_wizard.sync_editor.v5.7",
        "total_lines": len(lines_data),
        "total_words": total_words,
        "lines": lines_data,
    }
    machine_json_str = json.dumps(machine_payload, indent=2, ensure_ascii=False)

    # Prepare Section 2: Human-Readable Easy-to-Fix Lines
    human_lines: List[str] = []
    for idx, ld in enumerate(lines_data, 1):
        st_str = format_timestamp_raw(ld["start_sec"])
        en_str = format_timestamp_raw(ld["end_sec"])
        word_tokens: List[str] = []
        for w in ld["words"]:
            dur_cs = int(w.get("duration_cs", 30))
            word_tokens.append(f"{w['word']} ({dur_cs}cs)")
        human_lines.append(f"Line {idx:2d}: [{st_str} -> {en_str}]  {' '.join(word_tokens)}")

    file_content = [
        "# ==============================================================================",
        f"# 🪄 TIKTOK WIZARD — SYNC & HIGHLIGHT TIMINGS EDITOR (v5.7)",
        f"# Project: {proj_name}",
        "# ==============================================================================",
        "# INSTRUCTIONS FOR CREATORS & AUTOMATION:",
        "#",
        "# • SECTION 1 below contains MACHINE-READABLE data (exact JSON with timestamps,",
        "#   durations, confidence scores, and line boundaries).",
        "#",
        "# • SECTION 2 below contains HUMAN-READABLE easy-to-fix lyrics where you can",
        "#   freely make manual corrections in Notepad or any text editor:",
        "#     - Edit line start/end timings: [00:14.79 -> 00:17.35]",
        "#     - Edit word highlight durations: (XXcs) = centiseconds (100cs = 1.0 second).",
        "#     - Edit words, fix typos, add punctuation, or add missing lyrics.",
        "#     - If you omit (XXcs) or add new words, TikTok Wizard automatically and",
        "#       smoothly balances durations across the line interval!",
        "#",
        "# • SAVE THIS FILE: TikTok Wizard automatically detects edits on next render,",
        "#   or select 'Reload Manual Corrections' from the project menu (Option 9).",
        "# ==============================================================================\n",
        "# ==============================================================================",
        "# SECTION 1: MACHINE-READABLE TIMESTAMPS & HIGHLIGHT DATA (JSON)",
        "# ==============================================================================",
        "# [MACHINE_DATA_START]",
        machine_json_str,
        "# [MACHINE_DATA_END]\n",
        "# ==============================================================================",
        "# SECTION 2: HUMAN-READABLE EASY-TO-FIX EDITOR (EDIT TIMESTAMPS & HIGHLIGHTS HERE)",
        "# ==============================================================================",
        "# Format: Line <N>: [START -> END]  Word1 (XXcs) Word2 (XXcs) ...",
        "# ==============================================================================\n",
    ]
    file_content.extend(human_lines)
    file_content.append("")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(file_content), encoding="utf-8")
    return True


def export_easy_lyrics_file(
    ass_path: Path,
    output_file: Path,
    manifest: Optional[Dict[str, Any]] = None,
    project_name: str = "",
) -> bool:
    """Backward-compatible alias for export_sync_editor_file."""
    return export_sync_editor_file(
        ass_path=ass_path,
        output_file=output_file,
        manifest=manifest,
        project_name=project_name,
    )


def import_sync_editor_to_entries(
    editor_path: Path,
    default_line_dur: float = 2.5,
) -> List[KaraokeEntry]:
    """
    Parse a dual sync editor file into KaraokeEntry objects for write_ass_karaoke.
    Prioritizes Section 2 (human manual edits). If Section 2 has no lines, falls
    back to Section 1 (machine-readable JSON).
    """
    if not editor_path.exists():
        return []

    content = editor_path.read_text(encoding="utf-8-sig")
    lines = content.splitlines()

    entries: List[KaraokeEntry] = []
    line_pattern = re.compile(
        r"^(?:Line\s*\d+\s*:\s*)?(?:\[([0-9:.]+)\s*->\s*([0-9:.]+)\]|\[([0-9:.]+)\](?:\s*->\s*\[([0-9:.]+)\])?)\s*(.*)$"
    )

    # 1. Parse Section 2 (Human-Readable lines)
    human_lines: List[str] = []
    in_section_2 = False

    for raw in lines:
        s = raw.strip()
        if "SECTION 2:" in s:
            in_section_2 = True
            continue
        if in_section_2 and s and not s.startswith("#"):
            human_lines.append(s)
        elif not in_section_2 and s and not s.startswith("#") and not s.startswith("{") and not s.startswith('"'):
            # Also catch standalone human-readable lines outside explicit section header
            if line_pattern.match(s):
                human_lines.append(s)

    if human_lines:
        for l_idx, line in enumerate(human_lines):
            m = line_pattern.match(line)
            if not m:
                continue

            ts_start_str = m.group(1) or m.group(3)
            ts_end_str = m.group(2) or m.group(4)
            rest = m.group(5).strip()

            start_sec = to_seconds(ts_start_str)
            end_sec = to_seconds(ts_end_str) if ts_end_str else 0.0

            # Match Word tokens with optional (XXcs) duration
            tokens = re.findall(r"([^\s()]+)(?:\s*\(([0-9]+)cs\))?", rest)
            if not tokens:
                continue

            raw_words: List[Tuple[str, Optional[int]]] = []
            for w_text, cs_str in tokens:
                dur_cs = int(cs_str) if cs_str else None
                raw_words.append((w_text, dur_cs))

            # Duration allocation & auto-balancing
            assigned_cs = [c for _, c in raw_words if c is not None]
            total_assigned_cs = sum(assigned_cs)
            unassigned_count = len([1 for _, c in raw_words if c is None])

            # Determine available duration for unassigned words
            if end_sec > start_sec:
                target_dur_cs = int(round((end_sec - start_sec) * 100.0))
            else:
                target_dur_cs = int(default_line_dur * 100.0)

            if unassigned_count > 0:
                remaining_cs = max(unassigned_count * 15, target_dur_cs - total_assigned_cs)
                per_w_cs = max(15, remaining_cs // unassigned_count)
            else:
                per_w_cs = 30

            k_words: List[KaraokeWord] = []
            curr_time = start_sec
            for w_text, cs_val in raw_words:
                dur_cs = cs_val if cs_val is not None else per_w_cs
                dur_sec = dur_cs / 100.0
                k_words.append(KaraokeWord(
                    text=w_text,
                    duration_cs=dur_cs,
                    start=round(curr_time, 3),
                    end=round(curr_time + dur_sec, 3),
                ))
                curr_time += dur_sec

            line_end = max(end_sec, curr_time + 0.25) if end_sec > start_sec else curr_time + 0.35
            entries.append(KaraokeEntry(
                line_index=l_idx,
                words=k_words,
                start=round(start_sec, 3),
                end=round(line_end, 3),
                raw_text=" ".join(w.text for w in k_words),
            ))

        if entries:
            return entries

    # 2. Fallback to Section 1 (Machine-Readable JSON)
    if "# [MACHINE_DATA_START]" in content and "# [MACHINE_DATA_END]" in content:
        try:
            json_blob = content.split("# [MACHINE_DATA_START]")[1].split("# [MACHINE_DATA_END]")[0].strip()
            data = json.loads(json_blob)
            for li in data.get("lines", []):
                l_idx = int(li.get("line_index", 0))
                l_start = float(li.get("start_sec", 0.0))
                l_end = float(li.get("end_sec", l_start + 2.0))
                k_words: List[KaraokeWord] = []
                for w in li.get("words", []):
                    w_st = float(w.get("start_sec", l_start))
                    w_en = float(w.get("end_sec", w_st + 0.20))
                    dur_cs = int(w.get("duration_cs") or max(1, int(round((w_en - w_st) * 100.0))))
                    k_words.append(KaraokeWord(
                        text=str(w.get("word", "")),
                        duration_cs=dur_cs,
                        start=round(w_st, 3),
                        end=round(w_en, 3),
                    ))
                entries.append(KaraokeEntry(
                    line_index=l_idx,
                    words=k_words,
                    start=round(l_start, 3),
                    end=round(l_end, 3),
                    raw_text=str(li.get("text", " ".join(w.text for w in k_words))),
                ))
            return entries
        except Exception:
            pass

    return entries


def import_easy_lyrics_to_entries(
    easy_lyrics_path: Path,
    default_line_dur: float = 2.5,
) -> List[KaraokeEntry]:
    """Backward-compatible alias for import_sync_editor_to_entries."""
    return import_sync_editor_to_entries(
        editor_path=easy_lyrics_path,
        default_line_dur=default_line_dur,
    )
