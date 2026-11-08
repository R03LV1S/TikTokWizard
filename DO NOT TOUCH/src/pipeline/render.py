#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShotCut TikTok Automator — Render Pipeline

Builds the FFmpeg command for burning TikTok karaoke subtitles (ASS)
over looping/custom backgrounds (videos, pictures, gifs), with optional audio
spectrum/waveform visualizers and custom time-frame clipping.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _escape_ffmpeg_path(path: Path) -> str:
    """
    Escape Windows paths for FFmpeg filtergraph syntax:
      - Use forward slashes
      - Escape colon in drive letter (W: -> W\\:)
      - Escape single quotes
    """
    p_str = Path(path).resolve().as_posix()
    p_str = p_str.replace(":", r"\:")
    p_str = p_str.replace("'", r"\'")
    return p_str


def _find_font_dir(
    font_name: str,
    font_file: Optional[str] = None,
    defaults_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Return directory containing font files for libass fontsdir option."""
    if font_file:
        pf = Path(font_file)
        if pf.exists():
            return pf.parent

    if defaults_dir and defaults_dir.exists():
        font_variants = [
            font_name.lower().replace(" ", "").replace("-", "").replace("_", "")
        ]
        try:
            # Check directory and subdirectories (e.g. _Assets/Fonts)
            for f in defaults_dir.rglob("*"):
                if not f.is_file():
                    continue
                f_clean = f.stem.lower().replace(" ", "").replace("-", "").replace("_", "")
                if f.suffix.lower() in (".ttf", ".otf", ".ttc"):
                    for variant in font_variants:
                        if variant in f_clean or f_clean in variant:
                            return f.parent
        except Exception:
            pass
        return defaults_dir

    win_fonts = Path(r"C:\Windows\Fonts")
    if win_fonts.exists():
        return win_fonts

    return None


def find_default_background(defaults_dir: Path, preferred_file: Optional[str] = None) -> Optional[Path]:
    """
    Find background asset from defaults_dir.
    Backgrounds are strictly loaded from _Assets/Backgrounds/ (never project folders).
    If preferred_file is provided, tries to find that specific filename first.
    """
    defaults_dir = Path(defaults_dir)
    if not defaults_dir.exists():
        return None

    if preferred_file:
        pref = Path(preferred_file)
        if pref.is_absolute() and pref.exists():
            return pref
        direct = defaults_dir / preferred_file
        if direct.exists():
            return direct
        matches = list(defaults_dir.rglob(pref.name))
        if matches:
            return matches[0]

    # Look for files explicitly named 'default.*' in Backgrounds subdirectories
    for ext in (".mp4", ".mov", ".mkv", ".webm", ".jpg", ".jpeg", ".png", ".gif", ".webp"):
        matches = list(defaults_dir.rglob(f"default{ext}"))
        if matches:
            return matches[0]

    # Fallback to any media in Backgrounds
    for ext in (".mp4", ".mov", ".mkv", ".webm", ".jpg", ".jpeg", ".png", ".gif", ".webp"):
        matches = list(defaults_dir.rglob(f"*{ext}"))
        if matches:
            return matches[0]

    return None


def build_ffmpeg_command(
    config: dict,
    effect_cfg: dict,
    audio_path: Path,
    background_path: Optional[Path],
    defaults_dir: Path,
    ass_path: Path,
    output_path: Path,
    duration: float,
    start_time: float = 0.0,
) -> List[str]:
    """
    Build FFmpeg command for vertical TikTok karaoke render.
    Supports audio start_time clipping, animated GIFs, images, videos, and visualizers.
    """
    audio_path = Path(audio_path)
    ass_path = Path(ass_path)
    output_path = Path(output_path)
    defaults_dir = Path(defaults_dir)

    w = config.get("video_width", 1080)
    h = config.get("video_height", 1920)
    fps = config.get("fps", 30)
    ffmpeg_bin = config.get("ffmpeg_path", "ffmpeg")

    visualizer = effect_cfg.get("visualizer", {"type": "none", "color": "#00ffff", "height_pct": 12})
    bg_cfg = effect_cfg.get("background", {})
    fit_mode = bg_cfg.get("fit_mode", "blur_pad")  # blur_pad, crop_fill, or pad

    # Audio input is 0 (with optional start_time seeking)
    inputs: List[str] = []
    if start_time > 0:
        inputs.extend(["-ss", f"{start_time:.3f}"])
    inputs.extend(["-t", f"{duration:.3f}", "-i", str(audio_path.resolve())])

    # Background input is 1
    found_bg = False
    resolved_bg = background_path if background_path and background_path.exists() else find_default_background(defaults_dir)

    if resolved_bg and resolved_bg.exists():
        ext = resolved_bg.suffix.lower()
        is_image = ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        is_gif = ext == ".gif"

        if is_image:
            inputs.extend(["-loop", "1", "-i", str(resolved_bg.resolve())])
        elif is_gif:
            inputs.extend(["-ignore_loop", "0", "-i", str(resolved_bg.resolve())])
        else:
            inputs.extend(["-stream_loop", "-1", "-i", str(resolved_bg.resolve())])
        found_bg = True

    filter_chains: List[str] = []

    # 1. Background Video Filtering
    if found_bg:
        if fit_mode == "crop_fill":
            filter_chains.append(
                f"[1:v]scale={w}:{h}:force_original_aspect_ratio=increase,"
                f"crop={w}:{h},setsar=1,fps={fps}[bg]"
            )
        elif fit_mode == "pad":
            filter_chains.append(
                f"[1:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps={fps}[bg]"
            )
        elif fit_mode == "dark_pad":
            filter_chains.append(
                f"[1:v]split=2[bg_src][fg_src];"
                f"[bg_src]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur=30:5,eq=brightness=-0.32:contrast=1.2[bg_dark];"
                f"[fg_src]scale={w}:{h}:force_original_aspect_ratio=decrease[fg_sharp];"
                f"[bg_dark][fg_sharp]overlay=(W-w)/2:(H-h)/2,setsar=1,fps={fps}[bg]"
            )
        elif fit_mode == "mirror_pad":
            filter_chains.append(
                f"[1:v]split=2[bg_src][fg_src];"
                f"[bg_src]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},hflip,boxblur=25:5[bg_mirror];"
                f"[fg_src]scale={w}:{h}:force_original_aspect_ratio=decrease[fg_sharp];"
                f"[bg_mirror][fg_sharp]overlay=(W-w)/2:(H-h)/2,setsar=1,fps={fps}[bg]"
            )
        else:
            # blur_pad (TikTok professional style: blurred fill with sharp video centered)
            filter_chains.append(
                f"[1:v]split=2[bg_src][fg_src];"
                f"[bg_src]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur=25:5[bg_blur];"
                f"[fg_src]scale={w}:{h}:force_original_aspect_ratio=decrease[fg_sharp];"
                f"[bg_blur][fg_sharp]overlay=(W-w)/2:(H-h)/2,setsar=1,fps={fps}[bg]"
            )
    else:
        # Fallback solid color
        color = bg_cfg.get("colors", ["#0a0a0a"])[0] if bg_cfg.get("colors") else "#0a0a0a"
        filter_chains.append(f"color=c={color}:s={w}x{h}:r={fps}:d={duration}[bg]")

    current_vid = "bg"

    # 2. Visualizer Overlay
    viz_type = visualizer.get("type", "none")
    viz_color = visualizer.get("color", "#00ffff")
    viz_h_pct = visualizer.get("height_pct", 12)
    viz_h = max(60, int(h * viz_h_pct / 100))
    viz_pos = visualizer.get("position", "above_lyrics")

    # Determine Y position of visualizer
    lyrics_y_pct = effect_cfg.get("lyrics", {}).get("position_y_percent", 70)
    lyrics_y_px = int(h * lyrics_y_pct / 100)
    if viz_pos == "top":
        viz_y = int(h * 0.12)
    elif viz_pos == "center":
        viz_y = int((h - viz_h) / 2)
    elif viz_pos == "bottom":
        viz_y = int(h - viz_h - 80)
    elif viz_pos == "below_lyrics":
        viz_y = min(h - viz_h - 40, lyrics_y_px + 80)
    else:  # "above_lyrics" (default)
        viz_y = max(40, lyrics_y_px - viz_h - 60)

    if viz_type == "audio_spectrum":
        # Multi-color or custom colored rainbow spectrum
        filter_chains.append(
            f"[0:a]showspectrum=s={w}x{viz_h}:mode=combined:color=rainbow:scale=cbrt[spec]"
        )
        filter_chains.append(f"[{current_vid}][spec]overlay=0:{viz_y}:format=auto[vid]")
        current_vid = "vid"
    elif viz_type == "audio_waveform":
        filter_chains.append(
            f"[0:a]showwaves=s={w}x{viz_h}:mode=line:colors={viz_color}[wave]"
        )
        filter_chains.append(f"[{current_vid}][wave]overlay=0:{viz_y}:format=auto[vid]")
        current_vid = "vid"
    elif viz_type in ("audio_bars", "neon_glow_bars"):
        filter_chains.append(
            f"[0:a]showfreqs=s={w}x{viz_h}:mode=bar:ascale=log:fscale=log:colors={viz_color}[bars]"
        )
        filter_chains.append(f"[{current_vid}][bars]overlay=0:{viz_y}:format=auto[vid]")
        current_vid = "vid"
    elif viz_type in ("musical_cqt", "showcqt"):
        filter_chains.append(
            f"[0:a]showcqt=s={w}x{viz_h}:bar_g=2:gamma=2:count=1,format=rgba,colorchannelmixer=aa=0.85[cqt]"
        )
        filter_chains.append(f"[{current_vid}][cqt]overlay=0:{viz_y}:format=auto[vid]")
        current_vid = "vid"
    elif viz_type in ("circular_vector", "avectorscope"):
        scope_size = min(w - 40, max(240, viz_h * 2))
        filter_chains.append(
            f"[0:a]asplit=2[a1][a2];[a1][a2]amerge[stereo];"
            f"[stereo]avectorscope=s={scope_size}x{scope_size}:m=lissajous:rc=0:gc=255:bc=255:zoom=1.4,"
            f"format=rgba,colorkey=0x000000:0.1:0.1[scope]"
        )
        filter_chains.append(f"[{current_vid}][scope]overlay=(W-w)/2:{viz_y}:format=auto[vid]")
        current_vid = "vid"
    elif viz_type in ("stereo_wave", "p2p_waveform"):
        filter_chains.append(
            f"[0:a]showwaves=s={w}x{viz_h}:mode=p2p:colors={viz_color}:scale=lin[waves]"
        )
        filter_chains.append(f"[{current_vid}][waves]overlay=0:{viz_y}:format=auto[vid]")
        current_vid = "vid"
    elif viz_type in ("minimal_line", "cline_waveform"):
        filter_chains.append(
            f"[0:a]showwaves=s={w}x{viz_h}:mode=cline:colors={viz_color}[cline]"
        )
        filter_chains.append(f"[{current_vid}][cline]overlay=0:{viz_y}:format=auto[vid]")
        current_vid = "vid"

    # 3. ASS Subtitle Burn-In
    escaped_ass = _escape_ffmpeg_path(ass_path)
    lyrics_font = effect_cfg.get("lyrics", {}).get("font", config.get("karaoke", {}).get("font", "Chango"))
    resolved_font_dir = _find_font_dir(lyrics_font, config.get("karaoke", {}).get("font_file"), defaults_dir)

    fontsdir_opt = ""
    if resolved_font_dir and resolved_font_dir.exists():
        escaped_font_dir = _escape_ffmpeg_path(resolved_font_dir)
        fontsdir_opt = f":fontsdir='{escaped_font_dir}'"

    ass_filter = f"[{current_vid}]ass='{escaped_ass}'{fontsdir_opt}[final]"
    filter_chains.append(ass_filter)

    # Combine into -filter_complex
    filter_complex_str = ";".join(filter_chains)

    cmd = [
        ffmpeg_bin,
        "-y",
        *inputs,
        "-filter_complex", filter_complex_str,
        "-map", "[final]",
        "-map", "0:a",
        "-c:v", config.get("codec", "h264"),
        "-crf", str(config.get("crf", 22)),
        "-preset", "fast",
        "-r", str(fps),
        "-c:a", config.get("audio_codec", "aac"),
        "-b:a", "192k",
        "-t", f"{duration:.3f}",
        str(output_path.resolve()),
    ]

    return cmd
