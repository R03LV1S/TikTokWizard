"""ASS karaoke subtitle writer for pipeline output (v4.1).

Generates valid SubStation Alpha v4.00+ (`karaoke.ass`) files with:
  - Strict 1:1 line count (N lyric lines -> N `Dialogue:` events, zero merging).
  - Exact centisecond timestamp conversion (`H:MM:SS.cc`) from either seconds
    (`start_sec`/`end_sec`) or milliseconds (`start_ms`/`end_ms`).
  - Per-word `{\\k<XXcs>}` karaoke tags and explicit `{\\k<gap_cs>}` inter-word
    breath pauses.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def format_ass_time(seconds: float) -> str:
    """Format seconds into ASS timestamp `H:MM:SS.cc` with exact rounding."""
    s = max(0.0, float(seconds))
    total_cs = int(round(s * 100.0))
    h = total_cs // 360000
    rem = total_cs % 360000
    m = rem // 6000
    rem %= 6000
    sec = rem // 100
    cs = rem % 100
    return f"{h}:{m:02d}:{sec:02d}.{cs:02d}"


def _extract_start_end_sec(item: Dict[str, Any]) -> tuple[float, float]:
    """Extract (start_sec, end_sec) from a dict supporting either *_sec or *_ms keys."""
    if "start_sec" in item or "end_sec" in item:
        st = float(item.get("start_sec", 0.0))
        en = float(item.get("end_sec", st + 0.20))
    else:
        st = float(item.get("start_ms", 0.0)) / 1000.0
        en = float(item.get("end_ms", st * 1000.0 + 200.0)) / 1000.0
    return max(0.0, st), max(st + 0.03, en)


def _build_karaoke_text_from_words(
    lwords: Sequence[Dict[str, Any]], line_start: float
) -> str:
    parts: List[str] = []
    prev_end = line_start
    for w in lwords:
        w_st, w_en = _extract_start_end_sec(w)
        gap_cs = int(round((w_st - prev_end) * 100.0))
        if gap_cs >= 3:
            parts.append(f"{{\\k{gap_cs}}}")
        dur_cs = int(
            w.get("duration_cs") or max(3, int(round((w_en - w_st) * 100.0)))
        )
        w_txt = str(w.get("word", w.get("text", "")))
        parts.append(f"{{\\k{max(3, dur_cs)}}}{w_txt}")
        prev_end = w_en
    return " ".join(parts)


def write_ass(
    dialogues: Sequence[Dict[str, Any]],
    output_path: str | Path,
    title: str = "TikTok Wizard Karaoke",
    video_width: int = 1080,
    video_height: int = 1920,
    style_name: str = "Default",
) -> None:
    """Write a 1:1 unmerged ASS subtitle file with karaoke events.

    Supports:
      1. Line-level dialogue dicts (`{"start_ms": ..., "end_ms": ..., "text": ...}`
         or `{"start_sec": ..., "end_sec": ..., "words": [...]}`).
      2. Flat word-level dicts carrying `line_index` (automatically grouped by
         `line_index` into 1:1 `Dialogue:` lines).
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if caller passed flat word dicts with `line_index` and no pre-rendered `text`
    is_flat_word_list = (
        bool(dialogues)
        and any("line_index" in d for d in dialogues)
        and all("text" not in d and "words" not in d for d in dialogues)
    )

    dialogue_rows: List[str] = []

    if is_flat_word_list:
        grouped: Dict[int, List[Dict[str, Any]]] = {}
        for w in dialogues:
            li = int(w.get("line_index", 0))
            grouped.setdefault(li, []).append(w)

        for li in sorted(grouped.keys()):
            lwords = grouped[li]
            if not lwords:
                continue
            st_sec, _ = _extract_start_end_sec(lwords[0])
            _, en_sec = _extract_start_end_sec(lwords[-1])
            en_sec = max(st_sec + 0.05, en_sec)
            k_text = _build_karaoke_text_from_words(lwords, st_sec)
            dialogue_rows.append(
                f"Dialogue: 0,{format_ass_time(st_sec)},{format_ass_time(en_sec)},"
                f"{style_name},,0,0,0,,{k_text}"
            )
    else:
        for dialog in dialogues:
            st_sec, en_sec = _extract_start_end_sec(dialog)
            if "words" in dialog and isinstance(dialog["words"], list) and dialog["words"]:
                lwords = dialog["words"]
                w0_st, _ = _extract_start_end_sec(lwords[0])
                _, wn_en = _extract_start_end_sec(lwords[-1])
                st_sec = min(st_sec, w0_st) if st_sec > 0 else w0_st
                en_sec = max(en_sec, wn_en, st_sec + 0.05)
                text_payload = _build_karaoke_text_from_words(lwords, st_sec)
            else:
                raw_text = str(dialog.get("text", dialog.get("word", "")))
                if "{\\k" in raw_text:
                    text_payload = raw_text
                else:
                    dur_cs = max(3, int(round((en_sec - st_sec) * 100.0)))
                    text_payload = f"{{\\k{dur_cs}}}{raw_text}" if raw_text else ""

            dialogue_rows.append(
                f"Dialogue: 0,{format_ass_time(st_sec)},{format_ass_time(en_sec)},"
                f"{style_name},,0,0,0,,{text_payload}"
            )

    header = [
        "[Script Info]",
        f"Title: {title}",
        "ScriptType: v4.00+",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        f"PlayResX: {int(video_width)}",
        f"PlayResY: {int(video_height)}",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Default,Arial,68,&H0000FFFF,&H00FFFFFF,&H00000000,&H64000000,"
        "-1,0,0,0,100,100,0,0,1,3,0,2,40,40,180,1",
        "Style: Karaoke,Arial,68,&H0000FFFF,&H00FFFFFF,&H00000000,&H64000000,"
        "-1,0,0,0,100,100,0,0,1,3,0,2,40,40,180,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for h_line in header:
            f.write(h_line + "\n")
        for row in dialogue_rows:
            f.write(row + "\n")
