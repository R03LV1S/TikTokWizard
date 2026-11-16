#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShotCut TikTok Automator — ASS Karaoke Writer (v4.2)

Supports multiple dynamic karaoke display styles:
1. "word_highlight" (Default): Single line with real-time per-word wipe.
2. "word_pop": Active highlighted word dynamically scales up (+30% size pop effect).
3. "underline": Active highlighted word receives distinct dynamic underline accent.
4. "clean_static": Clean readable subtitle line without active karaoke wipe.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from .schema import KaraokeEntry, KaraokeWord


def ass_timestamp(seconds: float) -> str:
    """Format seconds into ASS timestamp: H:MM:SS.cc (centiseconds)."""
    total_cs = max(0, int(round(float(seconds) * 100.0)))
    h = total_cs // 360000
    total_cs %= 360000
    m = total_cs // 6000
    total_cs %= 6000
    s = total_cs // 100
    cs = total_cs % 100
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _resolve_font(lyrics_cfg: dict, config_karaoke: dict) -> str:
    """Return the font family name for use in the ASS style."""
    if lyrics_cfg and "font" in lyrics_cfg:
        f = str(lyrics_cfg["font"])
        f_clean = f.lower().replace("-", " ").replace("_", " ").strip()
        if "arial narrow" in f_clean:
            return "Arial Narrow"
        if "arial" in f_clean and "narrow" not in f_clean:
            return "Arial"
        if "impact" in f_clean:
            return "Impact"
        if "segoe" in f_clean:
            return "Segoe UI"
        if "trebuchet" in f_clean:
            return "Trebuchet MS"
        if "fraunces" in f_clean:
            return "Fraunces"
        if "chango" in f_clean:
            return "Chango"
        return f
    return str(config_karaoke.get("font", "Chango"))


def write_ass_karaoke(
    entries: List[KaraokeEntry],
    ass_path: Path,
    config: dict,
    effect_cfg: dict,
    video_width: int,
    video_height: int,
    defaults_dir: Optional[Path] = None,
    time_offset: float = 0.0,
    end_time_limit: Optional[float] = None,
) -> None:
    """Generate a rich ASS karaoke subtitle file with multiple configurable animation styles."""
    config_k = config.get("karaoke", {}) if isinstance(config, dict) else {}
    fx_lyrics = effect_cfg.get("lyrics", {}) if isinstance(effect_cfg, dict) else {}

    font = _resolve_font(fx_lyrics, config_k)
    font_size = int(fx_lyrics.get("font_size", config_k.get("font_size", 72)))
    highlight_color = fx_lyrics.get("highlight_color", config_k.get("highlight_color", "&H0000FFFF&"))
    base_color = fx_lyrics.get("base_color", config_k.get("base_color", "&H00FFFFFF&"))
    outline_color = fx_lyrics.get("outline_color", config_k.get("outline_color", "&H00000000&"))
    outline_width = int(fx_lyrics.get("outline_width", config_k.get("outline_width", 5)))
    shadow = bool(fx_lyrics.get("shadow", config_k.get("shadow", True)))
    shadow_depth = int(fx_lyrics.get("shadow_depth", config_k.get("shadow_depth", 3)))
    pos_y_pct = float(fx_lyrics.get("position_y_percent", config_k.get("position_y_percent", 65)))
    border_style = int(fx_lyrics.get("border_style", 1))
    back_color = fx_lyrics.get("box_color", "&H80000000&") if border_style == 3 else "&H00000000&"
    spacing = int(fx_lyrics.get("spacing", 0))
    bold = -1 if fx_lyrics.get("bold", True) else 0
    uppercase = bool(fx_lyrics.get("uppercase", False) or fx_lyrics.get("casing", "") == "upper")
    karaoke_style = str(fx_lyrics.get("karaoke_style", "word_highlight"))

    shadow_val = shadow_depth if shadow else 0

    lines: List[str] = [
        "[Script Info]",
        "Title: TikTok Karaoke",
        "ScriptType: v4.00+",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.601",
        f"PlayResX: {video_width}",
        f"PlayResY: {video_height}",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        (
            f"Style: Default,{font},{font_size},{highlight_color},{base_color},"
            f"{outline_color},{back_color},{bold},0,0,0,100,100,{spacing},0,{border_style},{outline_width},{shadow_val},5,40,40,40,1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    x_px = int(video_width) // 2
    y_px = int(float(video_height) * pos_y_pct / 100.0)
    line_spacing_px = int(font_size * 1.35)

    valid_entries = [e for e in entries if e.words]

    for idx, entry in enumerate(valid_entries):
        e_start = float(entry.start) - float(time_offset)
        e_end = float(entry.end) - float(time_offset)

        if end_time_limit is not None and e_start >= float(end_time_limit):
            continue
        if e_end <= 0.0:
            continue

        clipped_start = max(0.0, e_start)
        clipped_end = (
            min(float(end_time_limit), e_end) if end_time_limit is not None else e_end
        )
        if clipped_end <= clipped_start:
            clipped_end = clipped_start + 0.05

        first_w_start = float(entry.words[0].start) - float(time_offset)
        lead_in_cs = max(0, int(round((first_w_start - clipped_start) * 100.0)))

        # Format line text based on chosen karaoke_style
        if karaoke_style == "clean_static":
            sentence_words = [
                kw.text.upper() if uppercase else kw.text for kw in entry.words
            ]
            line_text = " ".join(sentence_words)
        elif karaoke_style == "word_pop":
            text_parts: List[str] = []
            if lead_in_cs >= 3:
                text_parts.append(f"{{\\k{lead_in_cs}}}")
            prev_w_end = float(entry.words[0].start)
            for i, kw in enumerate(entry.words):
                if i > 0:
                    gap_cs = max(0, int(round((float(kw.start) - prev_w_end) * 100.0)))
                    if gap_cs >= 3:
                        text_parts.append(f" {{\\k{gap_cs}}}")
                    else:
                        text_parts.append(" ")
                w_text = kw.text.upper() if uppercase else kw.text
                dur_cs = max(3, int(kw.duration_cs))
                dur_ms = dur_cs * 10
                pop_tag = f"{{\\k{dur_cs}\\t(0,{dur_ms // 2},\\fscx130\\fscy130)\\t({dur_ms // 2},{dur_ms},\\fscx100\\fscy100)}}"
                text_parts.append(f"{pop_tag}{w_text}")
                prev_w_end = float(kw.end)
            line_text = "".join(text_parts)
        elif karaoke_style == "underline":
            text_parts = []
            if lead_in_cs >= 3:
                text_parts.append(f"{{\\k{lead_in_cs}}}")
            prev_w_end = float(entry.words[0].start)
            for i, kw in enumerate(entry.words):
                if i > 0:
                    gap_cs = max(0, int(round((float(kw.start) - prev_w_end) * 100.0)))
                    if gap_cs >= 3:
                        text_parts.append(f" {{\\k{gap_cs}}}")
                    else:
                        text_parts.append(" ")
                w_text = kw.text.upper() if uppercase else kw.text
                dur_cs = max(3, int(kw.duration_cs))
                text_parts.append(f"{{\\k{dur_cs}\\u1}}{w_text}{{\\u0}}")
                prev_w_end = float(kw.end)
            line_text = "".join(text_parts)
        else:
            # Standard word_highlight
            text_parts = []
            if lead_in_cs >= 3:
                text_parts.append(f"{{\\k{lead_in_cs}}}")
            prev_w_end = float(entry.words[0].start)
            for i, kw in enumerate(entry.words):
                if i > 0:
                    gap_cs = max(0, int(round((float(kw.start) - prev_w_end) * 100.0)))
                    if gap_cs >= 3:
                        text_parts.append(f" {{\\k{gap_cs}}}")
                    else:
                        text_parts.append(" ")
                w_text = kw.text.upper() if uppercase else kw.text
                dur_cs = max(3, int(kw.duration_cs))
                text_parts.append(f"{{\\k{dur_cs}}}{w_text}")
                prev_w_end = float(kw.end)
            line_text = "".join(text_parts)

        pos_tag = f"{{\\an5\\pos({x_px},{y_px})}}"
        lines.append(
            f"Dialogue: 0,{ass_timestamp(clipped_start)},{ass_timestamp(clipped_end)},"
            f"Default,,0,0,0,,{pos_tag}{line_text}"
        )

    ass_path = Path(ass_path)
    ass_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ass_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
