"""1:1 Synced Lyric Exporter and Parser (v4.1).

Exports cemented words into human-editable, 1:1 line-preserved synced lyric files
(`lyrics_synced.txt`) formatted as:
    [MM:SS.cc] Word1 (XXcs) Word2 (XXcs) ... [MM:SS.cc]

Invariants:
  1. Strict 1:1 line count: N lyric lines -> N non-empty lines in `lyrics_synced.txt`
     (never collapses all words onto a single line).
  2. Exact UTF-8 surface text preservation (`'enseñarme'`, parentheses `'(Gracias,'`).
  3. Exact centisecond rounding (`int(round(seconds * 100.0))`) with zero truncation drift.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def format_lrc_tag(seconds: float) -> str:
    """Format seconds as `[MM:SS.cc]` with exact centisecond rounding."""
    s = max(0.0, float(seconds))
    total_cs = int(round(s * 100.0))
    minutes = total_cs // 6000
    rem = total_cs % 6000
    secs = rem // 100
    cs = rem % 100
    return f"[{minutes:02d}:{secs:02d}.{cs:02d}]"


def _parse_lrc_tag(tag_str: str) -> float:
    parts = tag_str.strip().strip("[]").split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600.0 + int(parts[1]) * 60.0 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60.0 + float(parts[1])
        return float(parts[0])
    except ValueError:
        return 0.0


def export_lrc(
    cemented_words: Sequence[Dict[str, Any]],
    output_path: str | Path,
    lines: Optional[Sequence[str]] = None,
) -> None:
    """Export 1:1 synced lyric lines to `output_path` in UTF-8.

    Supports both:
      - Word-level dicts carrying `line_index` (groups words by `line_index` so
        425 words across 95 lines emit 95 synced lines).
      - Line-level dicts (or word dicts without `line_index`), emitting 1 line
        per entry.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not cemented_words and not lines:
        out_path.write_text("", encoding="utf-8")
        return

    has_line_index = any("line_index" in w for w in cemented_words)
    output_lines: List[str] = []

    if has_line_index:
        grouped: Dict[int, List[Dict[str, Any]]] = {}
        for w in cemented_words:
            li = int(w.get("line_index", 0))
            grouped.setdefault(li, []).append(w)

        total_lines = len(lines) if lines is not None else (max(grouped.keys()) + 1 if grouped else 0)
        prev_end = 0.0

        for li in range(total_lines):
            lwords = grouped.get(li, [])
            if lwords:
                line_start = float(lwords[0].get("start_sec", prev_end))
                line_end = max(
                    line_start + 0.03,
                    float(lwords[-1].get("end_sec", line_start + 0.20)),
                )
                word_tokens: List[str] = []
                for w in lwords:
                    w_txt = str(w.get("word", w.get("text", "")))
                    w_st = float(w.get("start_sec", line_start))
                    w_en = max(w_st + 0.03, float(w.get("end_sec", w_st + 0.10)))
                    dur_cs = int(w.get("duration_cs") or max(3, int(round((w_en - w_st) * 100.0))))
                    word_tokens.append(f"{w_txt} ({dur_cs:2d}cs)")
                output_lines.append(
                    f"{format_lrc_tag(line_start)} {' '.join(word_tokens)} {format_lrc_tag(line_end)}"
                )
                prev_end = line_end
            elif lines is not None and li < len(lines):
                # Fallback if an input line had zero aligned words
                output_lines.append(
                    f"{format_lrc_tag(prev_end)} {lines[li]} {format_lrc_tag(prev_end + 0.20)}"
                )
    else:
        for item in cemented_words:
            if "words" in item and isinstance(item["words"], list) and item["words"]:
                lwords = item["words"]
                line_start = float(item.get("start_sec", lwords[0].get("start_sec", 0.0)))
                line_end = float(item.get("end_sec", lwords[-1].get("end_sec", line_start + 0.20)))
                word_tokens = []
                for w in lwords:
                    w_txt = str(w.get("word", w.get("text", "")))
                    w_st = float(w.get("start_sec", line_start))
                    w_en = max(w_st + 0.03, float(w.get("end_sec", w_st + 0.10)))
                    dur_cs = int(w.get("duration_cs") or max(3, int(round((w_en - w_st) * 100.0))))
                    word_tokens.append(f"{w_txt} ({dur_cs:2d}cs)")
                output_lines.append(
                    f"{format_lrc_tag(line_start)} {' '.join(word_tokens)} {format_lrc_tag(line_end)}"
                )
            else:
                start_sec = float(
                    item.get("start_sec", float(item.get("start_ms", 0.0)) / 1000.0)
                )
                end_sec = float(
                    item.get(
                        "end_sec",
                        float(item.get("end_ms", start_sec * 1000.0 + 200.0)) / 1000.0,
                    )
                )
                end_sec = max(start_sec + 0.03, end_sec)
                word_txt = str(item.get("word", item.get("text", "")))
                dur_cs = int(
                    item.get("duration_cs")
                    or max(3, int(round((end_sec - start_sec) * 100.0)))
                )
                output_lines.append(
                    f"{format_lrc_tag(start_sec)} {word_txt} ({dur_cs:2d}cs) {format_lrc_tag(end_sec)}"
                )

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for ln in output_lines:
            f.write(ln + "\n")


def parse_lrc_synced(input_path: str | Path) -> List[Dict[str, Any]]:
    """Parse a `lyrics_synced.txt` file back into line and word timing dicts."""
    p = Path(input_path)
    if not p.is_file():
        return []

    raw_lines = p.read_text(encoding="utf-8-sig").splitlines()
    parsed_lines: List[Dict[str, Any]] = []
    token_pattern = re.compile(r"(\S+?)\s*\(\s*(\d+)cs\)|(\S+)")

    for line_idx, raw in enumerate(raw_lines):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue

        start_sec = 0.0
        m_lead = re.match(r"^\[([0-9:.]+)\]\s*(.*)$", s)
        body = s
        if m_lead:
            start_sec = _parse_lrc_tag(m_lead.group(1))
            body = m_lead.group(2).strip()

        end_sec: Optional[float] = None
        m_tail = re.search(r"\s*\[([0-9:.]+)\]\s*$", body)
        if m_tail:
            end_sec = _parse_lrc_tag(m_tail.group(1))
            body = body[: m_tail.start()].strip()

        words: List[Dict[str, Any]] = []
        cursor = start_sec
        for m in token_pattern.finditer(body):
            if m.group(1) is not None:
                w_txt = m.group(1)
                dur_cs = max(3, int(m.group(2)))
            else:
                w_txt = m.group(3)
                dur_cs = 20
            dur_sec = float(dur_cs) / 100.0
            words.append(
                {
                    "word": w_txt,
                    "line_index": line_idx,
                    "start_sec": round(cursor, 4),
                    "end_sec": round(cursor + dur_sec, 4),
                    "duration_cs": dur_cs,
                }
            )
            cursor += dur_sec

        parsed_lines.append(
            {
                "index": line_idx + 1,
                "start_sec": round(start_sec, 4),
                "end_sec": round(end_sec if end_sec is not None else cursor, 4),
                "words": words,
                "text": " ".join(w["word"] for w in words),
            }
        )

    return parsed_lines
