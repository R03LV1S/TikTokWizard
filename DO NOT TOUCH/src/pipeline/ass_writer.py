#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShotCut TikTok Automator — ASS Karaoke Writer

Generates an ASS subtitle file with per-word karaoke highlighting (\\k tags).
Supports custom fonts, styles, highlight colors, outlines, and positions.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .schema import KaraokeEntry, KaraokeWord


def ass_timestamp(seconds: float) -> str:
    """Format seconds into ASS timestamp: H:MM:SS.cc (centiseconds)."""
    total_cs = max(0, int(round(seconds * 100)))
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
        f = lyrics_cfg["font"]
        # Map common filenames/variants to proper font family names if needed
        f_clean = f.lower().replace("-", " ").replace("_", " ").strip()
        if "arial narrow" in f_clean:
            return "Arial Narrow"
        elif "arial" in f_clean and "narrow" not in f_clean:
            return "Arial"
        elif "impact" in f_clean:
            return "Impact"
        elif "segoe" in f_clean:
            return "Segoe UI"
        elif "trebuchet" in f_clean:
            return "Trebuchet MS"
        elif "fraunces" in f_clean:
            return "Fraunces"
        elif "chango" in f_clean:
            return "Chango"
        return f
    return config_karaoke.get("font", "Chango")


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
    """
    Generate an ASS karaoke subtitle file with {\\k<duration>} per-word highlighting.
    Supports uppercase transform, box pill backgrounds, custom Y positions, and spacing.
    """
    config_k = config.get("karaoke", {})
    fx_lyrics = effect_cfg.get("lyrics", {})

    font = _resolve_font(fx_lyrics, config_k)
    font_size = fx_lyrics.get("font_size", config_k.get("font_size", 72))
    highlight_color = fx_lyrics.get("highlight_color", config_k.get("highlight_color", "&H0000FFFF&"))
    base_color = fx_lyrics.get("base_color", config_k.get("base_color", "&H00FFFFFF&"))
    outline_color = fx_lyrics.get("outline_color", config_k.get("outline_color", "&H00000000&"))
    outline_width = fx_lyrics.get("outline_width", config_k.get("outline_width", 5))
    shadow = fx_lyrics.get("shadow", config_k.get("shadow", True))
    shadow_depth = fx_lyrics.get("shadow_depth", config_k.get("shadow_depth", 3))
    pos_y_pct = fx_lyrics.get("position_y_percent", config_k.get("position_y_percent", 70))
    border_style = fx_lyrics.get("border_style", 1)  # 1 = outline + shadow, 3 = opaque box
    back_color = fx_lyrics.get("box_color", "&H80000000&") if border_style == 3 else "&H00000000&"
    spacing = fx_lyrics.get("spacing", 0)
    bold = -1 if fx_lyrics.get("bold", True) else 0
    uppercase = fx_lyrics.get("uppercase", False) or fx_lyrics.get("casing", "") == "upper"

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

    x_px = video_width // 2
    y_px = int(video_height * pos_y_pct / 100)

    for entry in entries:
        if not entry.words:
            continue

        e_start = entry.start - time_offset
        e_end = entry.end - time_offset

        # Filter out entries outside trimmed time range if clipping
        if end_time_limit is not None and e_start >= end_time_limit:
            continue
        if e_end <= 0:
            continue

        # Clamp visible range
        clipped_start = max(0.0, e_start)
        clipped_end = min(end_time_limit, e_end) if end_time_limit is not None else e_end

        if clipped_end <= clipped_start:
            continue

        # Calculate lead-in pause
        first_w_start = entry.words[0].start - time_offset
        lead_in_cs = max(0, int(round((first_w_start - e_start) * 100)))

        # Format words with ASS karaoke tags: {\k<duration_cs>}Word
        text_parts: List[str] = []
        if lead_in_cs > 0:
            text_parts.append(f"{{\\k{lead_in_cs}}}")

        prev_w_end = entry.words[0].start

        for i, kw in enumerate(entry.words):
            # Check for pause before this word
            if i > 0:
                gap_cs = max(0, int(round((kw.start - prev_w_end) * 100)))
                if gap_cs > 5:
                    text_parts.append(f" {{\\k{gap_cs}}}")
                else:
                    text_parts.append(" ")

            w_text = kw.text.upper() if uppercase else kw.text
            text_parts.append(f"{{\\k{kw.duration_cs}}}{w_text}")
            prev_w_end = kw.end

        line_text = "".join(text_parts)

        # Positioning: middle-center (\an5) at (x_px, y_px)
        pos_tag = f"{{\\an5\\pos({x_px},{y_px})}}"

        lines.append(
            f"Dialogue: 0,{ass_timestamp(clipped_start)},{ass_timestamp(clipped_end)},"
            f"Default,,0,0,0,,{pos_tag}{line_text}"
        )

    ass_path = Path(ass_path)
    ass_path.write_text("\n".join(lines), encoding="utf-8")
