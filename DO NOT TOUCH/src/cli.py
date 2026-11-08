#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShotCut TikTok Automator — CLI Entry Point (TikTok Wizard)

Interactive CLI for producing TikTok / YouTube Shorts style karaoke videos
with word-by-word highlighting timed precisely to audio speech.

Features:
- Responsive text wrapping for small/unmaximized terminal windows.
- Custom Time-frame clipping (e.g. 01:10 to 01:40).
- Profile system with persistent profiles stored in DO NOT TOUCH/profiles/.
- Create custom profiles: select video/image/gif backgrounds, fonts, colors,
  letter styling, and audio spectrum/waveform visualizers.
- "Last Used" profile quick-selection.
- Delete custom profiles (Default protected).
- Strict Background Rule: Backgrounds are always sourced from _Assets/Backgrounds/.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root and src are importable
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from pipeline.schema import (
    KaraokeEntry,
    KaraokeMeta,
    KaraokeWord,
    LyricLine,
    LyricWord,
)
from pipeline.alignment import align
from pipeline.timing import resolve_with_no_overlap
from pipeline.ass_writer import write_ass_karaoke
from pipeline.render import build_ffmpeg_command, find_default_background
from pipeline.lyrics_editor import (
    export_easy_lyrics_file,
    import_easy_lyrics_to_entries,
    parse_ass_to_easy_lyrics,
)


# ---------------------------------------------------------------------------
# Responsive Terminal Formatting Helpers
# ---------------------------------------------------------------------------

def get_term_width() -> int:
    """Return current terminal column width (minimum 40, default 80)."""
    try:
        cols = shutil.get_terminal_size((80, 24)).columns
        return max(40, cols)
    except Exception:
        return 80


def print_wrapped(text: str, indent: str = "", subsequent_indent: Optional[str] = None) -> None:
    """Print text wrapped cleanly to the current terminal width."""
    width = get_term_width()
    sub_indent = indent if subsequent_indent is None else subsequent_indent
    lines = textwrap.wrap(
        text,
        width=width,
        initial_indent=indent,
        subsequent_indent=sub_indent,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if lines:
        for line in lines:
            print(line)
    else:
        print(indent)


def print_header(title: str) -> None:
    """Print a clean responsive header box."""
    width = min(80, get_term_width())
    border = "=" * width
    print(border)
    print(title.center(width))
    print(border + "\n")


def print_divider() -> None:
    width = min(80, get_term_width())
    print("-" * width)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path | str, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json_atomic(path: Path | str, data: Any) -> None:
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return "sha256:" + h.hexdigest()
    except Exception:
        return ""


def ensure_dir(path: Path | str) -> None:
    os.makedirs(path, exist_ok=True)


def fmt_seconds(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def parse_time_str(val: str) -> Optional[float]:
    """Parse string like '01:10', '1:10', '70', or '00:01:10.50' into seconds."""
    val = val.strip()
    if not val:
        return None
    # Check if pure float/int seconds
    try:
        return float(val)
    except ValueError:
        pass
    parts = val.split(":")
    if len(parts) == 2:
        try:
            return int(parts[0]) * 60 + float(parts[1])
        except ValueError:
            return None
    elif len(parts) == 3:
        try:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        except ValueError:
            return None
    return None


def open_folder_and_select(file_path: Path) -> None:
    """Open system file manager with the newly rendered video file highlighted."""
    try:
        fp = Path(file_path).resolve()
        if sys.platform == "win32":
            # /select,<file> opens Windows Explorer with file selected
            subprocess.Popen(["explorer.exe", f"/select,{str(fp)}"])
        elif sys.platform == "darwin":
            # open -R <file> reveals file in macOS Finder
            subprocess.Popen(["open", "-R", str(fp)])
        else:
            # Linux fallback to xdg-open on parent directory
            subprocess.Popen(["xdg-open", str(fp.parent if fp.is_file() else fp)])
    except Exception:
        pass


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _safe_input(prompt: str = "") -> str:
    try:
        return input(prompt)
    except EOFError:
        print("\n(Exiting)")
        sys.exit(0)


def _is_back(val: str) -> bool:
    """Return True if user input indicates wanting to go back / cancel."""
    return val.strip().lower() in ("b", "back", "cancel", "0")


# ---------------------------------------------------------------------------
# Configuration & Defaults
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "schema": "shotcut_tiktok.config.v1",
    "projects_dir": "../START HERE/_Projects",
    "defaults_dir": "../START HERE/_Projects/_Assets",
    "build_dir": "build",
    "output_dir": "output",
    "history_file": "render-history.json",
    "last_used_profile": "default",
    "auto_open_folder": True,
    "video_width": 1080,
    "video_height": 1920,
    "fps": 30,
    "preset": "tiktok",
    "codec": "h264",
    "audio_codec": "aac",
    "crf": 22,
    "default_effect_preset": "default",
    "lyric_sync_engine": "forced_alignment_dp",
    "whisper_model": "small",
    "whisper_compute_type": "int8",
    "ffmpeg_path": "ffmpeg",
    "melt_path": "melt",
    "safe_mode": True,
    "karaoke": {
        "enabled": True,
        "highlight_color": "&H00B400FF&",
        "base_color": "&H00D900FF&",
        "outline_color": "&H00FFFFFF&",
        "font": "Chango",
        "font_size": 72,
        "outline_width": 4,
        "shadow": True,
        "shadow_depth": 5,
        "position_y_percent": 70,
        "safe_zone_pct": 90,
        "font_file": None,
    },
}


def load_config() -> dict:
    cfg_path = _project_root / "config.json"
    cfg = load_json(cfg_path, None)
    if cfg is None:
        save_json_atomic(cfg_path, DEFAULT_CONFIG)
        return dict(DEFAULT_CONFIG)
    merged = dict(DEFAULT_CONFIG)
    merged.update(cfg)
    if "karaoke" not in merged or not isinstance(merged["karaoke"], dict):
        merged["karaoke"] = dict(DEFAULT_CONFIG["karaoke"])
    else:
        k_defs = dict(DEFAULT_CONFIG["karaoke"])
        k_defs.update(merged["karaoke"])
        merged["karaoke"] = k_defs
    return merged


def save_config(cfg: dict) -> None:
    cfg_path = _project_root / "config.json"
    save_json_atomic(cfg_path, cfg)


# ---------------------------------------------------------------------------
# Profile Management (Stored in DO NOT TOUCH/profiles/)
# ---------------------------------------------------------------------------

PROFILES_DIR = _project_root / "profiles"


def init_default_profiles() -> None:
    """Ensure standard starter profiles exist in DO NOT TOUCH/profiles/."""
    ensure_dir(PROFILES_DIR)
    default_p = PROFILES_DIR / "default.json"
    if not default_p.exists():
        save_json_atomic(default_p, {
            "name": "DEFAULT (default.mp4 + Neon Pink Bubble Lyrics)",
            "description": "Signature style: default background video with bold Chango bubble-letter lyrics, vibrant neon pink font/highlight, crisp white outline, and deep black offset drop-shadow.",
            "is_default": True,
            "background": {"type": "video", "fit_mode": "blur_pad", "file": None},
            "visualizer": {"type": "none", "color": "#00ffff", "height_pct": 12, "position": "above_lyrics"},
            "lyrics": {
                "font": "Chango",
                "font_size": 72,
                "highlight_color": "&H00B400FF&",
                "base_color": "&H00D900FF&",
                "outline_color": "&H00FFFFFF&",
                "outline_width": 4,
                "shadow": True,
                "shadow_depth": 5,
                "position_y_percent": 70,
            },
        })

    spectrum_p = PROFILES_DIR / "neon_spectrum.json"
    if not spectrum_p.exists():
        save_json_atomic(spectrum_p, {
            "name": "Neon Spectrum Visualizer",
            "description": "Default background with a vibrant rainbow audio spectrum visualizer pulsing above bubble lyrics.",
            "is_default": False,
            "background": {"type": "video", "fit_mode": "blur_pad", "file": None},
            "visualizer": {"type": "audio_spectrum", "color": "#ff00ff", "height_pct": 15, "position": "above_lyrics"},
            "lyrics": {
                "font": "Chango",
                "font_size": 72,
                "highlight_color": "&H0000FFFF&",
                "base_color": "&H00FFFFFF&",
                "outline_color": "&H00000000&",
                "outline_width": 5,
                "shadow": True,
                "shadow_depth": 3,
                "position_y_percent": 70,
            },
        })

    wave_p = PROFILES_DIR / "cybernetic_waveform.json"
    if not wave_p.exists():
        save_json_atomic(wave_p, {
            "name": "Cybernetic Waveform",
            "description": "Default background with an energetic cyan waveform visualizer line above bubble lyrics.",
            "is_default": False,
            "background": {"type": "video", "fit_mode": "blur_pad", "file": None},
            "visualizer": {"type": "audio_waveform", "color": "#00ffff", "height_pct": 12, "position": "above_lyrics"},
            "lyrics": {
                "font": "Chango",
                "font_size": 72,
                "highlight_color": "&H0000FFFF&",
                "base_color": "&H00FFFFFF&",
                "outline_color": "&H00000000&",
                "outline_width": 5,
                "shadow": True,
                "shadow_depth": 3,
                "position_y_percent": 70,
            },
        })

    # Additional high-impact pre-built profiles for v2.5
    starter_profiles = {
        "cyberpunk_neon": {
            "name": "Cyberpunk Neon (Impact + CQT Musical Spectrogram)",
            "description": "Ultra-punchy uppercase Impact font with electric cyan sung highlight and hot magenta unsung glow, powered by musical CQT spectrogram and blur-pad background.",
            "is_default": False,
            "background": {"type": "video", "fit_mode": "blur_pad", "file": None},
            "visualizer": {"type": "musical_cqt", "color": "#00ffff", "height_pct": 14, "position": "above_lyrics"},
            "lyrics": {"font": "Impact", "font_size": 76, "uppercase": True, "highlight_color": "&H00FFFF00&", "base_color": "&H00FF00FF&", "outline_color": "&H00000000&", "outline_width": 6, "shadow": True, "shadow_depth": 4, "position_y_percent": 70, "spacing": 1},
        },
        "rap_trap_punch": {
            "name": "808 Trap & Hip-Hop (Impact + Dual Stereo Wave)",
            "description": "High-impact bold uppercase text in Flame Gold and Pure White with peak-to-peak dual stereo soundwave visualizer and cinematic dark pad background.",
            "is_default": False,
            "background": {"type": "video", "fit_mode": "dark_pad", "file": None},
            "visualizer": {"type": "stereo_wave", "color": "#ffcc00", "height_pct": 12, "position": "above_lyrics"},
            "lyrics": {"font": "Impact", "font_size": 80, "uppercase": True, "highlight_color": "&H0000D7FF&", "base_color": "&H00FFFFFF&", "outline_color": "&H00000000&", "outline_width": 6, "shadow": True, "shadow_depth": 5, "position_y_percent": 70, "spacing": 2},
        },
        "retro_synthwave": {
            "name": "80s Retro Synthwave (Trebuchet MS + Neon Equalizer Bars)",
            "description": "Outrun synthwave aesthetic with Sunset Orange & Neon Purple gradient tones, pulsing vertical frequency bars, and mirror-pad reflection.",
            "is_default": False,
            "background": {"type": "video", "fit_mode": "mirror_pad", "file": None},
            "visualizer": {"type": "neon_glow_bars", "color": "#ff00aa", "height_pct": 13, "position": "above_lyrics"},
            "lyrics": {"font": "Trebuchet MS", "font_size": 74, "uppercase": True, "highlight_color": "&H00FF0080&", "base_color": "&H000080FF&", "outline_color": "&H00000000&", "outline_width": 5, "shadow": True, "shadow_depth": 4, "position_y_percent": 68, "spacing": 1},
        },
        "lofi_chill": {
            "name": "Lo-Fi Chill & Coffee (Fraunces + Minimal Soundwave)",
            "description": "Relaxed aesthetic serif lyrics in Soft Peach and Warm Cream, subtle centered line waveform, and warm dark pad background.",
            "is_default": False,
            "background": {"type": "video", "fit_mode": "dark_pad", "file": None},
            "visualizer": {"type": "minimal_line", "color": "#e0d0c0", "height_pct": 8, "position": "below_lyrics"},
            "lyrics": {"font": "Fraunces", "font_size": 64, "uppercase": False, "highlight_color": "&H00B0E0E6&", "base_color": "&H00E8F8F8&", "outline_color": "&H00202020&", "outline_width": 3, "shadow": True, "shadow_depth": 2, "position_y_percent": 72},
        },
        "edm_festival": {
            "name": "EDM Festival Rave (Impact + Circular Vector Scope)",
            "description": "High-voltage rave energy featuring Neon Acid Green and Cyan karaoke lyrics with a spinning circular Lissajous vector scope visualizer.",
            "is_default": False,
            "background": {"type": "video", "fit_mode": "blur_pad", "file": None},
            "visualizer": {"type": "circular_vector", "color": "#00ff66", "height_pct": 15, "position": "above_lyrics"},
            "lyrics": {"font": "Impact", "font_size": 76, "uppercase": True, "highlight_color": "&H0000FF66&", "base_color": "&H00FFFF00&", "outline_color": "&H00000000&", "outline_width": 6, "shadow": True, "shadow_depth": 4, "position_y_percent": 70, "spacing": 1},
        },
        "minimalist_clean": {
            "name": "Minimalist Clean (Segoe UI + Pill Box Highlight)",
            "description": "Sleek modern subtitle styling with translucent dark pill background box, clean white and cyan text, zero visualizer clutter.",
            "is_default": False,
            "background": {"type": "video", "fit_mode": "blur_pad", "file": None},
            "visualizer": {"type": "none", "color": "#00ffff", "height_pct": 10, "position": "above_lyrics"},
            "lyrics": {"font": "Segoe UI", "font_size": 66, "uppercase": False, "highlight_color": "&H0000FFFF&", "base_color": "&H00FFFFFF&", "outline_color": "&H00000000&", "outline_width": 10, "shadow": False, "shadow_depth": 0, "border_style": 3, "box_color": "&H80000000&", "position_y_percent": 72},
        },
        "heavy_metal": {
            "name": "Heavy Metal & Rock (Impact + Peak Soundwave)",
            "description": "Hard rock aggression featuring Blood Crimson and Ghost White uppercase lyrics with high-contrast peak stereo soundwave lines.",
            "is_default": False,
            "background": {"type": "video", "fit_mode": "dark_pad", "file": None},
            "visualizer": {"type": "stereo_wave", "color": "#e60020", "height_pct": 14, "position": "above_lyrics"},
            "lyrics": {"font": "Impact", "font_size": 78, "uppercase": True, "highlight_color": "&H000020E6&", "base_color": "&H00E0E0E0&", "outline_color": "&H00000000&", "outline_width": 7, "shadow": True, "shadow_depth": 5, "position_y_percent": 70, "spacing": 2},
        },
        "condensed_viral": {
            "name": "Viral Shorts Condensed (Arial Narrow + Equalizer Bars)",
            "description": "Modern tall condensed lettering in signature viral TikTok yellow and pure white, designed for maximum words per line on mobile screens.",
            "is_default": False,
            "background": {"type": "video", "fit_mode": "blur_pad", "file": None},
            "visualizer": {"type": "neon_glow_bars", "color": "#ffff00", "height_pct": 11, "position": "above_lyrics"},
            "lyrics": {"font": "Arial Narrow", "font_size": 74, "uppercase": True, "highlight_color": "&H0000FFFF&", "base_color": "&H00FFFFFF&", "outline_color": "&H00000000&", "outline_width": 5, "shadow": True, "shadow_depth": 4, "position_y_percent": 70},
        },
    }

    for sp_key, sp_data in starter_profiles.items():
        sp_path = PROFILES_DIR / f"{sp_key}.json"
        if not sp_path.exists():
            save_json_atomic(sp_path, sp_data)


def list_profiles() -> Dict[str, dict]:
    """Load all profiles from DO NOT TOUCH/profiles/, ensuring 'default' is always first."""
    init_default_profiles()
    profiles: Dict[str, dict] = {}
    for p in sorted(PROFILES_DIR.glob("*.json")):
        data = load_json(p)
        if data and isinstance(data, dict):
            key = p.stem
            profiles[key] = data

    # Ensure 'default' is at the very front
    sorted_profiles: Dict[str, dict] = {}
    if "default" in profiles:
        sorted_profiles["default"] = profiles.pop("default")
    for k, v in profiles.items():
        sorted_profiles[k] = v
    return sorted_profiles


def save_profile(key: str, data: dict) -> None:
    ensure_dir(PROFILES_DIR)
    path = PROFILES_DIR / f"{key}.json"
    save_json_atomic(path, data)


def delete_profile(key: str) -> bool:
    if key.lower() == "default":
        return False
    path = PROFILES_DIR / f"{key}.json"
    if path.exists():
        try:
            path.unlink()
            return True
        except Exception:
            return False
    return False


# Ordered standard resolutions tailored for social media & video formats (1-10)
# Ordered resolutions with target platforms & format benefits (1-9)
STANDARD_RESOLUTIONS: List[Tuple[str, str, int, int]] = [
    (
        "1920x1080 (9:16 Full HD)",
        "TikTok, Shorts, Reels, Snapchat (Standard phone full-screen)",
        1080,
        1920,
    ),
    (
        "2560x1440 (9:16 2K Vertical HD)",
        "High-bitrate social uploads (Sharper text, avoids platform blur)",
        1440,
        2560,
    ),
    (
        "3840x2160 (9:16 4K Ultra Vertical)",
        "4K mobile displays, master archives & studio-quality exports",
        2160,
        3840,
    ),
    (
        "1280x720  (9:16 Fast Mobile Render)",
        "Quick draft previews, lightweight test renders & low-spec PCs",
        720,
        1280,
    ),
    (
        "1080x1080 (1:1 Instagram Square)",
        "Instagram Feed, Facebook, X (Twitter), Threads, Bluesky",
        1080,
        1080,
    ),
    (
        "1920x1080 (16:9 YouTube Landscape)",
        "YouTube long-form, Vimeo, desktop monitors, TV displays",
        1920,
        1080,
    ),
    (
        "2560x1440 (16:9 2K Landscape)",
        "2K YouTube widescreen (Unlocks YouTube's premium VP9/AV1 codec)",
        2560,
        1440,
    ),
    (
        "3840x2160 (16:9 4K Landscape)",
        "4K UHD YouTube, cinematic presentations & master releases",
        3840,
        2160,
    ),
]

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
LYRIC_EXTS = {".txt", ".lrc", ".srt"}


def find_projects(projects_dir: Path) -> List[Path]:
    """
    Find all user song project folders in _Projects.
    Always ignores '_Assets', 'Defaults', and template folders.
    Lists all user song folders, displaying status of audio and lyrics files.
    """
    if not projects_dir.is_dir():
        return []
    candidates = []
    for p in sorted(projects_dir.iterdir()):
        if not p.is_dir():
            continue
        norm_name = p.name.strip().lower()
        # Always strictly ignore _Assets, Defaults, and template/example-only folders
        if norm_name in ("_assets", "assets", "defaults", "_defaults"):
            continue
        if norm_name.startswith(("_artist_song", "artist_song")):
            continue
        if norm_name.startswith("."):
            continue
        candidates.append(p)
    return candidates


def find_audio(folder: Path) -> Optional[Path]:
    if not folder.is_dir():
        return None
    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
            return f
    return None


def find_lyrics(folder: Path) -> Optional[Path]:
    if not folder.is_dir():
        return None
    # Prioritize files containing 'lyric' in the name, otherwise take first text file (ignoring READMEs)
    candidates = []
    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() in LYRIC_EXTS:
            if "readme" in f.name.lower():
                continue
            if "lyric" in f.name.lower():
                return f
            candidates.append(f)
    return candidates[0] if candidates else None


def list_background_assets(defaults_dir: Path) -> Dict[str, List[Path]]:
    """Scan _Assets/Backgrounds/ for Videos, Pictures, and Gifs."""
    bg_dir = defaults_dir / "Backgrounds"
    categories = {"Videos": [], "Pictures": [], "Gifs": []}
    if not bg_dir.exists():
        return categories

    for f in bg_dir.rglob("*"):
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        if ext in (".mp4", ".mov", ".mkv", ".webm", ".avi"):
            categories["Videos"].append(f)
        elif ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            categories["Pictures"].append(f)
        elif ext == ".gif":
            categories["Gifs"].append(f)
    return categories


def list_font_assets(defaults_dir: Path) -> List[str]:
    """Scan _Assets/Fonts/ and return prioritized fonts for viral TikTok subtitles."""
    fonts: List[str] = []
    priority_order = ["Impact", "Arial Narrow", "Chango", "Fraunces", "Segoe UI", "Trebuchet MS", "Arial Bold"]
    fonts_dir = defaults_dir / "Fonts"
    if fonts_dir.exists():
        for f in sorted(fonts_dir.rglob("*")):
            if f.is_file() and f.suffix.lower() in (".ttf", ".otf"):
                stem = f.stem.lower()
                if "fraunces" in stem:
                    name = "Fraunces"
                elif "chango" in stem:
                    name = "Chango"
                elif "impact" in stem:
                    name = "Impact"
                elif "arial-narrow" in stem or "arialnb" in stem:
                    name = "Arial Narrow"
                elif "arial" in stem and "narrow" not in stem:
                    name = "Arial Bold"
                elif "segoe" in stem:
                    name = "Segoe UI"
                elif "trebuc" in stem:
                    name = "Trebuchet MS"
                else:
                    name = f.stem.split("-")[0].split("_")[0]
                if name not in fonts:
                    fonts.append(name)
    # Add priority order
    ordered = [pf for pf in priority_order if pf in fonts] + [f for f in fonts if f not in priority_order]
    for pf in priority_order:
        if pf not in ordered:
            ordered.append(pf)
    return ordered


# ---------------------------------------------------------------------------
# History & Iterations
# ---------------------------------------------------------------------------

def load_history(history_path: Path) -> dict:
    data = load_json(history_path, None)
    if data is None:
        data = {
            "schema": "shotcut_tiktok.history.v1",
            "created_at": now_iso(),
            "last_updated": now_iso(),
            "total_renders": 0,
            "entries": [],
        }
        save_json_atomic(history_path, data)
    return data


def append_history(history_path: Path, entry: dict) -> None:
    data = load_history(history_path)
    data["entries"].append(entry)
    data["total_renders"] = len(data["entries"])
    data["last_updated"] = now_iso()
    save_json_atomic(history_path, data)


def get_project_data_dir(project_folder: Path) -> Path:
    """
    Get the internal .data cache directory located inside DO NOT TOUCH/_Projects/<name>/.data.
    Safely migrates any existing .data folder from START HERE/_Projects/<name>/.data.
    """
    dnt_data = _project_root / "_Projects" / project_folder.name / ".data"
    old_data = project_folder / ".data"

    if old_data.exists() and not dnt_data.exists():
        try:
            dnt_data.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_data), str(dnt_data))
        except Exception:
            pass

    dnt_data.mkdir(parents=True, exist_ok=True)
    return dnt_data


def load_project_completion(project_folder: Path) -> dict:
    dot_data = get_project_data_dir(project_folder)
    marker = dot_data / ".completed.json"
    legacy_marker = project_folder / ".completed.json"
    effective_marker = marker if marker.exists() else (legacy_marker if legacy_marker.exists() else marker)
    data = load_json(effective_marker, None)
    if data is None:
        data = {
            "schema": "shotcut_tiktok.completed.v1",
            "slug": project_folder.name,
            "status": "completed",
            "first_completed_at": None,
            "last_completed_at": None,
            "total_iterations": 0,
            "iterations": [],
        }
    return data


def record_iteration(
    project_folder: Path,
    iteration: int,
    effect_preset: str,
    output_file: Path,
    render_seconds: float,
    duration_seconds: float,
    resolution: str,
    fps: int,
    source_checksums: dict,
    notes: str = "",
) -> None:
    comp = load_project_completion(project_folder)
    if comp["first_completed_at"] is None:
        comp["first_completed_at"] = now_iso()
    comp["last_completed_at"] = now_iso()
    comp["total_iterations"] = iteration

    comp["iterations"].append({
        "iteration": iteration,
        "effect_preset": effect_preset,
        "output_file": str(output_file),
        "completed_at": now_iso(),
        "render_seconds": round(render_seconds, 2),
        "duration_seconds": round(duration_seconds, 2),
        "resolution": resolution,
        "fps": fps,
        "source_checksums": source_checksums,
        "notes": notes,
    })
    dot_data = get_project_data_dir(project_folder)
    dot_data.mkdir(parents=True, exist_ok=True)
    save_json_atomic(dot_data / ".completed.json", comp)


# ---------------------------------------------------------------------------
# FFmpeg Execution with Animated Responsive Progress Bar
# ---------------------------------------------------------------------------

def run_render_with_progress(cmd: List[str], duration: float) -> Tuple[bool, float, str]:
    start_t = time.time()
    time_re = re.compile(r"time=(\d{2}):(\d{2}):(\d{2})\.(\d{2,3})")
    stderr_lines: List[str] = []

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )

        while True:
            line = proc.stderr.readline()
            if not line and proc.poll() is not None:
                remaining = proc.stderr.read()
                if remaining:
                    stderr_lines.append(remaining)
                break
            if not line:
                continue
            stderr_lines.append(line)
            m = time_re.search(line)
            if m:
                hh, mm, ss, ms = m.groups()
                t = int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / (10 ** len(ms))
                pct = min(100, int((t / duration) * 100)) if duration > 0 else 0

                # Responsive progress bar width adapting to terminal size
                cols = get_term_width()
                avail_bar_len = max(10, min(40, cols - 32))
                filled = int(avail_bar_len * (pct / 100.0))
                bar = "=" * filled + "-" * (avail_bar_len - filled)

                status_str = f"[{bar}] {pct:3d}%  {fmt_seconds(t)} / {fmt_seconds(duration)}"
                print(f"\r{status_str[:cols-1]}", end="", flush=True)

        proc.wait()
        print()
        return proc.returncode == 0, time.time() - start_t, "".join(stderr_lines)
    except KeyboardInterrupt:
        proc.kill()
        proc.wait()
        print("\nRender cancelled by user.")
        return False, time.time() - start_t, ""


def probe_audio_duration(ffmpeg_path: str, audio_path: Path) -> float:
    try:
        res = subprocess.run(
            [ffmpeg_path, "-i", str(audio_path)],
            capture_output=True,
            text=True,
            timeout=15,
            encoding="utf-8",
            errors="replace",
        )
        m = re.search(r"Duration:\s*(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)", res.stderr)
        if m:
            hh, mm, ss = m.group(1), m.group(2), m.group(3)
            return int(hh) * 3600 + int(mm) * 60 + float(ss)
    except Exception:
        pass
    return 60.0


# ---------------------------------------------------------------------------
# Custom Profile Wizard
# ---------------------------------------------------------------------------

def create_custom_profile_wizard(defaults_dir: Path) -> Optional[Tuple[str, dict]]:
    """Interactive step-by-step wizard to create a new custom profile."""
    clear_screen()
    print_header("Create Custom Video Profile")
    print_wrapped("Customize every aspect of your video: background, visualizer, fonts, casing, styles, and colors.")
    print()

    # 1. Background Source
    print("1. Background Media Source:")
    bg_assets = list_background_assets(defaults_dir)
    print("   1. Default Video (default.mp4 in _Assets/Backgrounds/Videos) [Recommended]")
    print("   2. Choose another Video from _Assets/Backgrounds/Videos")
    print("   3. Choose a Picture from _Assets/Backgrounds/Pictures")
    print("   4. Choose an Animated GIF from _Assets/Backgrounds/Gifs")
    print("   0. Cancel & Back")

    bg_choice = _safe_input("\nSelect background type [1-4, 0/B to cancel, Enter for 1]: ").strip()
    if _is_back(bg_choice):
        return None

    selected_bg_file = None
    bg_type = "video"

    if bg_choice == "2" and bg_assets["Videos"]:
        print("\nAvailable Videos:")
        for i, vf in enumerate(bg_assets["Videos"], 1):
            print(f"   {i}. {vf.name}")
        print("   0. Cancel & Back")
        v_idx = _safe_input(f"\nSelect video [1-{len(bg_assets['Videos'])}, 0/B to cancel]: ").strip()
        if _is_back(v_idx):
            return None
        if v_idx.isdigit() and 1 <= int(v_idx) <= len(bg_assets["Videos"]):
            selected_bg_file = str(bg_assets["Videos"][int(v_idx) - 1].relative_to(defaults_dir))
    elif bg_choice == "3" and bg_assets["Pictures"]:
        bg_type = "picture"
        print("\nAvailable Pictures:")
        for i, pf in enumerate(bg_assets["Pictures"], 1):
            print(f"   {i}. {pf.name}")
        print("   0. Cancel & Back")
        p_idx = _safe_input(f"\nSelect picture [1-{len(bg_assets['Pictures'])}, 0/B to cancel]: ").strip()
        if _is_back(p_idx):
            return None
        if p_idx.isdigit() and 1 <= int(p_idx) <= len(bg_assets["Pictures"]):
            selected_bg_file = str(bg_assets["Pictures"][int(p_idx) - 1].relative_to(defaults_dir))
    elif bg_choice == "4" and bg_assets["Gifs"]:
        bg_type = "gif"
        print("\nAvailable Animated GIFs:")
        for i, gf in enumerate(bg_assets["Gifs"], 1):
            print(f"   {i}. {gf.name}")
        print("   0. Cancel & Back")
        g_idx = _safe_input(f"\nSelect gif [1-{len(bg_assets['Gifs'])}, 0/B to cancel]: ").strip()
        if _is_back(g_idx):
            return None
        if g_idx.isdigit() and 1 <= int(g_idx) <= len(bg_assets["Gifs"]):
            selected_bg_file = str(bg_assets["Gifs"][int(g_idx) - 1].relative_to(defaults_dir))

    # 2. Fit Mode
    print("\n2. Background Scaling / Fit Mode:")
    print("   1. Blur Pad (Recommended for TikTok: blurred 9:16 background with sharp video centered)")
    print("   2. Dark Pad (Cinematic dimmed & contrast-boosted dark background)")
    print("   3. Mirror Pad (Horizontally mirrored & blurred ambient reflection)")
    print("   4. Crop Fill (Fills entire screen, cropped to canvas)")
    print("   5. Black Pad (Adds black bars on top and bottom)")
    print("   0. Cancel & Back")
    fit_choice = _safe_input("Select fit mode [1-5, 0/B to cancel, Enter for 1]: ").strip()
    if _is_back(fit_choice):
        return None
    fit_mode = "blur_pad"
    if fit_choice == "2":
        fit_mode = "dark_pad"
    elif fit_choice == "3":
        fit_mode = "mirror_pad"
    elif fit_choice == "4":
        fit_mode = "crop_fill"
    elif fit_choice == "5":
        fit_mode = "pad"

    # 3. Audio Spectrum / Visualizer Overlay
    print("\n3. Audio Spectrum & Soundwave Visualizer Overlay:")
    print("   1. None (Clean background - no visualizer) [Default]")
    print("   2. Constant-Q Musical Note Spectrogram (Colorful reactive musical notes)")
    print("   3. Circular Vector Scope Orb (Spinning Lissajous rave / EDM scope)")
    print("   4. Dual Stereo Waveform (Peak-to-peak energetic soundwaves)")
    print("   5. Frequency Equalizer Bars (Vertical pulsing frequency bars)")
    print("   6. Centered Clean Soundwave Line (Subtle minimal lo-fi wave)")
    print("   7. Rainbow Audio Spectrum (Continuous multi-color spectrum)")
    print("   8. Single Audio Waveform Line (Classic soundwave line)")
    print("   0. Cancel & Back")
    viz_choice = _safe_input("Select visualizer [1-8, 0/B to cancel, Enter for 1]: ").strip()
    if _is_back(viz_choice):
        return None
    viz_type = "none"
    viz_color = "#00ffff"
    viz_h_pct = 12
    if viz_choice == "2":
        viz_type = "musical_cqt"
        viz_color = "#00ffff"
        viz_h_pct = 14
    elif viz_choice == "3":
        viz_type = "circular_vector"
        viz_color = "#00ff66"
        viz_h_pct = 15
    elif viz_choice == "4":
        viz_type = "stereo_wave"
        viz_color = "#ffcc00"
        viz_h_pct = 12
    elif viz_choice == "5":
        viz_type = "neon_glow_bars"
        viz_color = "#00ffcc"
        viz_h_pct = 12
    elif viz_choice == "6":
        viz_type = "minimal_line"
        viz_color = "#e0e0e0"
        viz_h_pct = 8
    elif viz_choice == "7":
        viz_type = "audio_spectrum"
        viz_color = "#ff00ff"
        viz_h_pct = 12
    elif viz_choice == "8":
        viz_type = "audio_waveform"
        viz_color = "#00ffff"
        viz_h_pct = 10

    # 4. Visualizer Placement (if visualizer active)
    viz_position = "above_lyrics"
    if viz_type != "none":
        print("\n4. Visualizer Placement & Location:")
        print("   1. Above Lyrics (Recommended - sits directly above subtitle text)")
        print("   2. Below Lyrics (Sits beneath subtitle text)")
        print("   3. Bottom of Screen (Resting at the lower edge)")
        print("   4. Center of Screen (True center canvas placement)")
        print("   5. Top of Screen (Upper third placement)")
        print("   0. Cancel & Back")
        viz_pos_choice = _safe_input("Select visualizer position [1-5, 0/B to cancel, Enter for 1]: ").strip()
        if _is_back(viz_pos_choice):
            return None
        if viz_pos_choice == "2":
            viz_position = "below_lyrics"
        elif viz_pos_choice == "3":
            viz_position = "bottom"
        elif viz_pos_choice == "4":
            viz_position = "center"
        elif viz_pos_choice == "5":
            viz_position = "top"

    # 5. Font Selection
    print("\n5. Subtitle Font:")
    fonts = list_font_assets(defaults_dir)
    for i, font_name in enumerate(fonts, 1):
        rec = " [Recommended / Viral TikTok Headline]" if font_name == "Impact" else ""
        print(f"   {i}. {font_name}{rec}")
    print("   0. Cancel & Back")
    font_choice = _safe_input(f"Select font [1-{len(fonts)}, 0/B to cancel, Enter for 1 ({fonts[0]})]: ").strip()
    if _is_back(font_choice):
        return None
    chosen_font = fonts[0]
    if font_choice.isdigit() and 1 <= int(font_choice) <= len(fonts):
        chosen_font = fonts[int(font_choice) - 1]

    # 6. Text Casing Transformation
    print("\n6. Text Casing & Transformation:")
    print("   1. UPPERCASE (Punchy TikTok / Shorts viral headline style) [Recommended]")
    print("   2. Normal / As-is (Preserve original casing from lyric file)")
    print("   0. Cancel & Back")
    casing_choice = _safe_input("Select casing [1-2, 0/B to cancel, Enter for 1]: ").strip()
    if _is_back(casing_choice):
        return None
    uppercase = True if casing_choice != "2" else False

    # 7. Subtitle Vertical Placement (Y-Position)
    print("\n7. Subtitle Screen Position (Vertical Placement):")
    print("   1. Center-Low 70% (TikTok / Reels Recommended - above platform controls & captions)")
    print("   2. True Center 50% (Centered on canvas)")
    print("   3. Center-High 35% (Clear of body / face in portrait videos)")
    print("   4. Top 15% (Headliner banner)")
    print("   5. Bottom 85% (Cinema lower-third subtitle)")
    print("   0. Cancel & Back")
    y_pos_choice = _safe_input("Select vertical position [1-5, 0/B to cancel, Enter for 1]: ").strip()
    if _is_back(y_pos_choice):
        return None
    pos_y_pct = 70
    if y_pos_choice == "2":
        pos_y_pct = 50
    elif y_pos_choice == "3":
        pos_y_pct = 35
    elif y_pos_choice == "4":
        pos_y_pct = 15
    elif y_pos_choice == "5":
        pos_y_pct = 85

    # 8. Subtitle Styling & Box Effects
    print("\n8. Subtitle Border & Box Background Style:")
    print("   1. Punchy Bold Outline & Drop Shadow (Classic TikTok Style) [Recommended]")
    print("   2. Modern Translucent Box / Pill Highlight Background")
    print("   3. Heavy Heavyweight 7px Outline (Maximum contrast for fast backgrounds)")
    print("   4. Clean Minimal 3px Outline (Subtle aesthetic)")
    print("   0. Cancel & Back")
    style_choice = _safe_input("Select subtitle style [1-4, 0/B to cancel, Enter for 1]: ").strip()
    if _is_back(style_choice):
        return None
    border_style = 1
    outline_width = 5
    shadow = True
    shadow_depth = 4
    box_color = "&H80000000&"
    if style_choice == "2":
        border_style = 3
        outline_width = 10
        shadow = False
        shadow_depth = 0
    elif style_choice == "3":
        outline_width = 7
        shadow_depth = 5
    elif style_choice == "4":
        outline_width = 3
        shadow_depth = 2

    # 9. Font Size
    print("\n9. Font Size:")
    print("   1. 72pt (Default TikTok bubble size - punchy & legible)")
    print("   2. 80pt (Extra Large - maximum headline impact)")
    print("   3. 64pt (Medium - great for multi-word lines)")
    print("   0. Cancel & Back")
    fsize_choice = _safe_input("Select font size [1-3, 0/B to cancel, Enter for 1]: ").strip()
    if _is_back(fsize_choice):
        return None
    font_size = 72
    if fsize_choice == "2":
        font_size = 80
    elif fsize_choice == "3":
        font_size = 64

    # 10. Color Scheme
    print("\n10. Lyrics Color Scheme:")
    print("   1. Signature Yellow Sung / Crisp White Unsung (Default TikTok / Shorts)")
    print("   2. Cyberpunk Cyan Sung / Hot Magenta Unsung")
    print("   3. 808 Flame Gold Sung / Pure White Unsung")
    print("   4. EDM Acid Green Sung / Electric Cyan Unsung")
    print("   5. 80s Synthwave Sunset Orange Sung / Neon Purple Unsung")
    print("   6. Lo-Fi Soft Peach Sung / Warm Cream Unsung")
    print("   7. Heavy Metal Blood Crimson Sung / Ghost White Unsung")
    print("   8. Custom Colors (Enter custom Hex codes)")
    print("   0. Cancel & Back")
    color_choice = _safe_input("Select color scheme [1-8, 0/B to cancel, Enter for 1]: ").strip()
    if _is_back(color_choice):
        return None
    highlight_color = "&H0000FFFF&"  # Yellow (BGR: 00, FF, FF)
    base_color = "&H00FFFFFF&"       # White
    if color_choice == "2":
        highlight_color = "&H00FFFF00&"  # Cyan
        base_color = "&H00FF00FF&"       # Magenta
    elif color_choice == "3":
        highlight_color = "&H0000D7FF&"  # Gold
        base_color = "&H00FFFFFF&"       # White
    elif color_choice == "4":
        highlight_color = "&H0000FF66&"  # Acid Green
        base_color = "&H00FFFF00&"       # Cyan
    elif color_choice == "5":
        highlight_color = "&H000080FF&"  # Sunset Orange
        base_color = "&H00FF0080&"       # Neon Purple
    elif color_choice == "6":
        highlight_color = "&H00B0E0E6&"  # Soft Peach
        base_color = "&H00E8F8F8&"       # Warm Cream
    elif color_choice == "7":
        highlight_color = "&H000020E6&"  # Blood Crimson
        base_color = "&H00E0E0E0&"       # Ghost White
    elif color_choice == "8":
        c_sung = _safe_input("Enter Sung Highlight Hex (e.g. #00ffff or Enter for Yellow): ").strip()
        if _is_back(c_sung):
            return None
        if c_sung.startswith("#") and len(c_sung) == 7:
            r, g, b = c_sung[1:3], c_sung[3:5], c_sung[5:7]
            highlight_color = f"&H00{b}{g}{r}&"

    # 11. Resolution Selection (1-9)
    print("\nStep 3/3: Video Resolution & Canvas Format:")
    for r_idx, (r_name, r_target, rw, rh) in enumerate(STANDARD_RESOLUTIONS, 1):
        def_tag = " [Recommended / Default]" if rw == 1080 and rh == 1920 else ""
        print(f"  {r_idx}. {r_name}{def_tag}")
        print(f"     Target: {r_target}")
    print("  9. Custom (Enter custom Width x Height)")
    print("     Target: Specialized canvas sizes (e.g. 4:5 vertical, 21:9 ultrawide)")
    print("  0. Back to Profile Selection")

    res_choice = _safe_input("\nSelect resolution [1-9, 0/B to cancel, Enter for 1]: ").strip()
    if _is_back(res_choice):
        return None
    p_w, p_h = 1080, 1920
    if res_choice.isdigit():
        choice_num = int(res_choice)
        if 1 <= choice_num <= len(STANDARD_RESOLUTIONS):
            _, _, p_w, p_h = STANDARD_RESOLUTIONS[choice_num - 1]
        elif choice_num == 9:
            while True:
                c_res = _safe_input("Enter custom resolution (e.g. 1080x1920, 0/B to cancel): ").strip().lower()
                if _is_back(c_res):
                    return None
                m = re.match(r"^(\d{3,4})[xX:](\d{3,4})$", c_res)
                if m:
                    p_w, p_h = int(m.group(1)), int(m.group(2))
                    break
                print("Invalid format. Please enter e.g. 1080x1920 or 1920x1080.")
    elif res_choice == "":
        p_w, p_h = 1080, 1920

    # 12. Profile Name
    print("\n12. Profile Name:")
    p_name = _safe_input("Enter a name for this profile (e.g. 'My Viral Beat', 0/B to cancel): ").strip()
    if _is_back(p_name):
        return None
    if not p_name:
        p_name = f"Custom Profile ({datetime.now().strftime('%b %d %H%M')})"

    profile_key = re.sub(r"[^a-zA-Z0-9_-]", "_", p_name.lower()).strip("_")
    if not profile_key or profile_key == "default":
        profile_key = f"custom_{int(time.time())}"

    new_profile = {
        "name": p_name,
        "description": f"Custom profile using {chosen_font} font, {viz_type} visualizer, {p_w}x{p_h} resolution, and {fit_mode} background.",
        "is_default": False,
        "resolution": {"width": p_w, "height": p_h},
        "background": {
            "type": bg_type,
            "fit_mode": fit_mode,
            "file": selected_bg_file,
        },
        "visualizer": {
            "type": viz_type,
            "color": viz_color,
            "height_pct": viz_h_pct,
            "position": viz_position,
        },
        "lyrics": {
            "font": chosen_font,
            "font_size": font_size,
            "uppercase": uppercase,
            "highlight_color": highlight_color,
            "base_color": base_color,
            "outline_color": "&H00000000&",
            "outline_width": outline_width,
            "shadow": shadow,
            "shadow_depth": shadow_depth,
            "border_style": border_style,
            "box_color": box_color,
            "position_y_percent": pos_y_pct,
        },
    }

    save_profile(profile_key, new_profile)
    print(f"\n[OK] Profile '{p_name}' successfully created and saved!")
    time.sleep(1.5)
    return profile_key, new_profile

def settings_menu(config: dict) -> None:
    """Interactive settings menu to configure global preferences."""
    while True:
        clear_screen()
        print_header("Settings & Preferences")
        auto_open = config.get("auto_open_folder", True)
        auto_str = "ENABLED (Auto-open in File Explorer)" if auto_open else "DISABLED"
        crf = config.get("crf", 22)
        fps = config.get("fps", 30)

        print(f"  1. Auto-Open Output Folder:   [{auto_str}]")
        print("     (Automatically opens the song folder with new video selected when render finishes)\n")
        print(f"  2. Video Render Quality (CRF): [{crf}] (Lower = higher quality / larger file)")
        print(f"  3. Default Frame Rate (FPS):   [{fps} FPS]")
        print("  4. Reset All Settings to Default\n")
        print("  0. Back to Main Menu\n")

        choice = _safe_input("Select an option to toggle/change [0-4, Enter to return]: ").strip().lower()
        if choice in ("0", "b", "back", "q", "exit", ""):
            break
        elif choice == "1":
            config["auto_open_folder"] = not auto_open
            save_config(config)
            st = "ENABLED" if config["auto_open_folder"] else "DISABLED"
            print(f"\n[OK] Auto-Open Output Folder is now: {st}")
            time.sleep(1)
        elif choice == "2":
            new_crf = _safe_input("Enter video CRF quality [16-28, Default 22, 0/B to cancel]: ").strip()
            if not _is_back(new_crf) and new_crf.isdigit() and 14 <= int(new_crf) <= 32:
                config["crf"] = int(new_crf)
                save_config(config)
                print(f"\n[OK] Video CRF set to {config['crf']}")
                time.sleep(1)
        elif choice == "3":
            print("\nSelect Frame Rate:")
            print("  1. 30 FPS (Standard social media format - recommended)")
            print("  2. 60 FPS (Ultra-smooth high-framerate motion)")
            print("  0. Cancel")
            fps_c = _safe_input("Select [1-2, 0/B to cancel]: ").strip()
            if fps_c == "1":
                config["fps"] = 30
                save_config(config)
                print("\n[OK] Frame rate set to 30 FPS.")
                time.sleep(1)
            elif fps_c == "2":
                config["fps"] = 60
                save_config(config)
                print("\n[OK] Frame rate set to 60 FPS.")
                time.sleep(1)
        elif choice == "4":
            confirm = _safe_input("Reset all settings to factory default? [y/N]: ").strip().lower()
            if confirm in ("y", "yes"):
                config.clear()
                config.update(DEFAULT_CONFIG)
                save_config(config)
                print("\n[OK] Settings reset to factory defaults.")
                time.sleep(1)


def delete_profile_menu() -> None:
    """Menu to delete custom profiles (protecting default)."""
    clear_screen()
    print_header("Delete a Custom Profile")
    profiles = list_profiles()
    custom_keys = [k for k in profiles.keys() if k != "default"]

    if not custom_keys:
        print_wrapped("No custom profiles found to delete. The 'default' profile is protected.", indent="  ")
        _safe_input("\nPress Enter to return...")
        return

    print("Select a profile to permanently remove:\n")
    for i, k in enumerate(custom_keys, 1):
        print(f"  {i}. {profiles[k].get('name', k)} ({k}.json)")
    print("\n  0. Cancel & return\n")

    choice = _safe_input(f"Select [1-{len(custom_keys)}, 0 to cancel]: ").strip()
    if choice in ("0", "", "q"):
        return
    if choice.isdigit() and 1 <= int(choice) <= len(custom_keys):
        del_key = custom_keys[int(choice) - 1]
        confirm = _safe_input(f"Are you sure you want to delete '{del_key}'? [y/N]: ").strip().lower()
        if confirm in ("y", "yes"):
            if delete_profile(del_key):
                print(f"\n[OK] Profile '{del_key}' deleted.")
            else:
                print(f"\n[FAIL] Could not delete '{del_key}'.")
            time.sleep(1.2)


# ---------------------------------------------------------------------------
# Main Interactive Flow
# ---------------------------------------------------------------------------

def main() -> int:
    config = load_config()

    projects_dir = _project_root / config["projects_dir"]
    defaults_dir = _project_root / config["defaults_dir"]
    build_dir = _project_root / config["build_dir"]
    history_path = _project_root / config["history_file"]

    ensure_dir(projects_dir)
    ensure_dir(defaults_dir)
    ensure_dir(build_dir)
    init_default_profiles()

    while True:
        clear_screen()
        print_header("TikTok Wizard v1.0.2")
        print_wrapped("Turns your song + lyrics.txt into viral TikTok/Shorts karaoke videos.")
        print()

        projects = find_projects(projects_dir)
        if not projects:
            print_wrapped("No project folders found.", indent="  ")
            print_wrapped(f"Please add a song folder inside: {projects_dir}", indent="  ")
            print_wrapped("Each folder needs:", indent="  ")
            print_wrapped("- Music file (.mp3 or .wav)", indent="    ")
            print_wrapped("- Lyrics file (.txt) formatted line-by-line", indent="    ")
            print()
            _safe_input("Press Enter to refresh...")
            continue

        print("Select a Project:\n")
        for i, p in enumerate(projects, 1):
            audio = find_audio(p)
            lyrics = find_lyrics(p)
            wc = 0
            if lyrics:
                try:
                    wc = len(lyrics.read_text(encoding="utf-8").split())
                except Exception:
                    pass
            print(f"  {i}. {p.name}")
            print(f"     Audio:  {audio.name if audio else '[MISSING]'}")
            print(f"     Lyrics: {lyrics.name if lyrics else '[MISSING]'} ({wc} words)")
        print("\n  S. Settings & Preferences (Toggle Auto-Open Folder, Quality, FPS)")
        print("  0. Exit\n")

        choice = _safe_input("Select [1-{}, S for Settings, 0 to exit]: ".format(len(projects))).strip()
        if choice in ("0", "exit", "q"):
            return 0
        if choice.lower() in ("s", "settings", "pref", "preferences"):
            settings_menu(config)
            continue

        try:
            p_idx = int(choice) - 1
            if p_idx < 0 or p_idx >= len(projects):
                raise ValueError
        except ValueError:
            print("Invalid selection.")
            time.sleep(1)
            continue

        project_folder = projects[p_idx]
        slug = project_folder.name
        audio_path = find_audio(project_folder)
        lyrics_path = find_lyrics(project_folder)

        # Pre-flight Check
        clear_screen()
        print_header(f"Project: {slug}")
        ok = True

        if audio_path:
            print(f"  [OK] Audio:      {audio_path.name}")
        else:
            print("  [FAIL] Missing audio file (.mp3 / .wav)")
            ok = False

        if lyrics_path:
            print(f"  [OK] Lyrics:     {lyrics_path.name}")
        else:
            print("  [FAIL] Missing lyrics file (.txt)")
            ok = False

        ffmpeg_bin = config.get("ffmpeg_path", "ffmpeg")
        try:
            subprocess.run([ffmpeg_bin, "-version"], capture_output=True, timeout=5)
            print(f"  [OK] FFmpeg:     {ffmpeg_bin} detected")
        except Exception:
            print(f"  [FAIL] FFmpeg not found on PATH ('{ffmpeg_bin}')")
            ok = False

        if not ok:
            print("\nPlease fix the missing files and run again.")
            _safe_input("\nPress Enter to return to menu...")
            continue

        total_audio_duration = probe_audio_duration(ffmpeg_bin, audio_path)
        print(f"  [OK] Audio Duration: {fmt_seconds(total_audio_duration)} ({total_audio_duration:.2f}s)\n")

        # -------------------------------------------------------------------
        # Time-Frame Selection (Full song or Custom Start/End)
        # -------------------------------------------------------------------
        print_divider()
        print_wrapped("Clip Time-Frame Selection", indent="  ")
        print_divider()
        print(f"  1. Full Song (0:00 to {fmt_seconds(total_audio_duration)}) (Default)")
        ex_start = fmt_seconds(min(30.0, total_audio_duration * 0.25))
        ex_end = fmt_seconds(min(60.0, total_audio_duration * 0.5))
        print(f"  2. Custom Time-Frame (e.g. {ex_start} to {ex_end})")
        print("  0. Back to Project Selection")
        time_choice = _safe_input("\nSelect [1/2, 0/B to cancel, Enter for Full Song]: ").strip()
        if _is_back(time_choice):
            continue

        start_time = 0.0
        render_duration = total_audio_duration

        if time_choice == "2":
            time_cancelled = False
            while True:
                s_inp = _safe_input(f"Enter START time [0:00 - {fmt_seconds(total_audio_duration)}, 0/B to cancel, Enter for 0:00]: ").strip()
                if _is_back(s_inp):
                    time_cancelled = True
                    break
                if not s_inp:
                    start_time = 0.0
                    break
                parsed_s = parse_time_str(s_inp)
                if parsed_s is not None and 0 <= parsed_s < total_audio_duration:
                    start_time = parsed_s
                    break
                print(f"Invalid start time. Must be between 0:00 and {fmt_seconds(total_audio_duration)}.")

            if time_cancelled:
                continue

            while True:
                e_inp = _safe_input(f"Enter END time [{fmt_seconds(start_time)} - {fmt_seconds(total_audio_duration)}, 0/B to cancel, Enter for end ({fmt_seconds(total_audio_duration)})]: ").strip()
                if _is_back(e_inp):
                    time_cancelled = True
                    break
                if not e_inp:
                    render_duration = total_audio_duration - start_time
                    break
                parsed_e = parse_time_str(e_inp)
                if parsed_e is not None and parsed_e > start_time:
                    render_duration = min(total_audio_duration, parsed_e) - start_time
                    break
                print(f"Invalid end time (must be greater than {fmt_seconds(start_time)} and up to {fmt_seconds(total_audio_duration)}). Please try again.")

            if time_cancelled:
                continue

            print(f"\n[OK] Clip set: {fmt_seconds(start_time)} to {fmt_seconds(start_time + render_duration)} (Duration: {fmt_seconds(render_duration)})\n")

        # -------------------------------------------------------------------
        # Step 1: Subtitles & Timing Alignment
        # -------------------------------------------------------------------
        dot_data = get_project_data_dir(project_folder)
        ass_path = dot_data / "karaoke.ass"
        meta_path = dot_data / "karaoke_meta.json"
        user_bak_path = dot_data / "karaoke.ass.user_edits.bak"
        easy_lyrics_path = project_folder / "lyrics_easy.txt"

        # Check if user edited lyrics_easy.txt
        easy_edits_detected = False
        if easy_lyrics_path.exists():
            curr_easy_hash = sha256_file(easy_lyrics_path)
            last_easy_hash = None
            if meta_path.exists():
                try:
                    last_easy_hash = json.loads(meta_path.read_text(encoding="utf-8")).get("easy_lyrics_checksum")
                except Exception:
                    pass
            if last_easy_hash and curr_easy_hash != last_easy_hash:
                easy_edits_detected = True

        use_custom_ass = False
        if easy_edits_detected:
            print_divider()
            print("  ✨ [EASY LYRICS EDITS DETECTED] Found changes in lyrics_easy.txt!")
            print("     TikTok Wizard will automatically apply your word & timing updates to the video.\n")
            print("  1. Apply edits from lyrics_easy.txt [Default]")
            print("  2. Re-align fresh with Whisper AI (Overwrites lyrics_easy.txt)")
            ez_choice = _safe_input("\nSelect option [1-2, Enter for 1]: ").strip()
            if ez_choice != "2":
                use_custom_ass = True
            else:
                use_custom_ass = False
        elif ass_path.exists():
            is_custom = False
            if user_bak_path.exists():
                is_custom = True
            if meta_path.exists():
                try:
                    m_obj = json.loads(meta_path.read_text(encoding="utf-8"))
                    c_hash = m_obj.get("ass_checksum")
                    if c_hash and sha256_file(ass_path) != c_hash:
                        is_custom = True
                except Exception:
                    pass

            print_divider()
            if is_custom:
                print("  ✨ [CUSTOM SUBTITLES DETECTED] Hand-tuned subtitle edits detected!")
                print("     Your custom timestamps will be preserved for this video render.\n")
            else:
                print("  📄 [EXISTING SUBTITLES DETECTED] Found existing synced subtitles.\n")
            print("  1. Keep existing subtitles (Preserve manual edits & bypass re-align) [Default]")
            print("  2. Re-align with Whisper AI (Fresh alignment; automatically backs up current file)")
            sub_choice = _safe_input("\nSelect subtitle source [1-2, Enter for 1]: ").strip()
            if sub_choice != "2":
                use_custom_ass = True
                print("\n  -> [OK] Keeping existing subtitles (manual edits preserved).")
            else:
                use_custom_ass = False
                ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                bak_dest = dot_data / f"karaoke.ass.bak_{ts_str}"
                try:
                    shutil.copy2(ass_path, bak_dest)
                    print(f"\n  -> [BACKUP] Existing subtitles safely backed up to .data/{bak_dest.name}")
                except Exception:
                    pass

        raw_lines = [l for l in lyrics_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        karaoke_entries = []
        karaoke_meta = {}

        def _run_alignment():
            nonlocal karaoke_entries, karaoke_meta
            print_divider()
            print_wrapped("Timing Alignment (Whisper AI)", indent="  ")
            print_divider()
            whisper_model = config.get("whisper_model", "small")
            compute_type = config.get("whisper_compute_type", "int8")
            res = align(
                audio_path=audio_path,
                lyrics_path=lyrics_path,
                total_duration=total_audio_duration,
                model_size=whisper_model,
                compute_type=compute_type,
            )
            if not res:
                return False
            w_entries, k_meta = res
            karaoke_meta = k_meta
            karaoke_entries = resolve_with_no_overlap(w_entries, line_count=len(raw_lines))
            return True

        if not use_custom_ass:
            if not _run_alignment():
                print("\n[ERROR] Alignment failed. Please check audio and lyrics.")
                _safe_input("\nPress Enter to return...")
                continue
        else:
            if meta_path.exists():
                try:
                    karaoke_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    karaoke_meta = {"word_count": len(raw_lines)}
            else:
                karaoke_meta = {"word_count": len(raw_lines)}

        # -------------------------------------------------------------------
        # Step 2: Visual Style / Profile Selection & Step 3: Resolution
        # -------------------------------------------------------------------
        profile_flow_cancelled = False
        while True:
            selected_profile_key = "default"
            selected_fx = None

            clear_screen()
            print_header("Step 2/3: Video Profile & Style")
            profiles = list_profiles()
            last_used_key = config.get("last_used_profile", "default")
            if last_used_key not in profiles:
                last_used_key = "default"

            print(f"Last Used Profile: [{profiles[last_used_key].get('name', last_used_key)}]\n")
            print("Available Profiles:\n")

            profile_keys = list(profiles.keys())
            for idx_k, k in enumerate(profile_keys, 1):
                p_data = profiles[k]
                tags = []
                if k == "default":
                    tags.append("Option 1 / Default")
                if k == last_used_key:
                    tags.append("Last Used")
                tag_str = f" ({', '.join(tags)})" if tags else ""

                print(f"  {idx_k}. {p_data.get('name', k)}{tag_str}")
                desc = p_data.get("description", "")
                if desc:
                    print_wrapped(desc, indent="     ", subsequent_indent="     ")
                print()

            print(f"  C. + Create New Custom Profile (Select Video/Image/GIF, Font, Colors, Spectrum)")
            print(f"  D. - Delete a Custom Profile")
            print(f"  L. Quick-select 'Last Used' [{profiles[last_used_key].get('name', last_used_key)}]")
            auto_st = "ON" if config.get("auto_open_folder", True) else "OFF"
            print(f"  S. Settings & Preferences (Auto-Open Folder: [{auto_st}])")
            print(f"  0. Back to Project Selection\n")

            choice_str = _safe_input(f"Select profile [1-{len(profile_keys)}, 0/B to cancel, Enter for 1 (Default)]: ").strip()

            if _is_back(choice_str):
                profile_flow_cancelled = True
                break
            elif choice_str.upper() == "C":
                res = create_custom_profile_wizard(defaults_dir)
                if res:
                    selected_profile_key, selected_fx = res
                else:
                    continue
            elif choice_str.upper() == "D":
                delete_profile_menu()
                continue
            elif choice_str.upper() == "L":
                selected_profile_key = last_used_key
                selected_fx = profiles[selected_profile_key]
            elif choice_str.upper() == "S":
                settings_menu(config)
                continue
            elif choice_str == "" or choice_str == "1":
                selected_profile_key = "default"
                selected_fx = profiles["default"]
            elif choice_str.isdigit():
                p_num = int(choice_str) - 1
                if 0 <= p_num < len(profile_keys):
                    selected_profile_key = profile_keys[p_num]
                    selected_fx = profiles[selected_profile_key]
                else:
                    print("Invalid choice, please select again.")
                    time.sleep(1)
                    continue
            else:
                print("Invalid choice, please select again.")
                time.sleep(1)
                continue

            # Update last used profile in config
            config["last_used_profile"] = selected_profile_key
            save_config(config)

            # -------------------------------------------------------------------
            # Resolution Selection (Exact 1-9 structure with Target Platforms)
            # -------------------------------------------------------------------
            print_divider()
            print_wrapped("Step 3/3: Video Resolution & Canvas Format", indent="  ")
            print_divider()
            for r_idx, (r_name, r_target, rw, rh) in enumerate(STANDARD_RESOLUTIONS, 1):
                def_tag = " [Recommended / Default]" if rw == 1080 and rh == 1920 else ""
                print(f"  {r_idx}. {r_name}{def_tag}")
                print(f"     Target: {r_target}")
            print("  9. Custom (Enter custom Width x Height)")
            print("     Target: Specialized canvas sizes (e.g. 4:5 vertical, 21:9 ultrawide)")
            print("  0. Back to Profile Selection")

            res_opt = _safe_input("\nSelect resolution [1-9, 0/B to go back, Enter for 1]: ").strip()
            if _is_back(res_opt):
                # Loops back to Step 2: Video Profile & Style
                continue

            vid_w, vid_h = 1080, 1920
            res_back = False
            if res_opt.isdigit():
                opt_num = int(res_opt)
                if 1 <= opt_num <= len(STANDARD_RESOLUTIONS):
                    _, _, vid_w, vid_h = STANDARD_RESOLUTIONS[opt_num - 1]
                elif opt_num == 9:
                    while True:
                        c_inp = _safe_input("Enter custom resolution (e.g. 1080x1920, 0/B to go back): ").strip().lower()
                        if _is_back(c_inp):
                            res_back = True
                            break
                        m = re.match(r"^(\d{3,4})[xX:](\d{3,4})$", c_inp)
                        if m:
                            vid_w, vid_h = int(m.group(1)), int(m.group(2))
                            break
                        print("Invalid format. Please enter e.g. 1080x1920 or 1920x1080.")
            elif res_opt == "":
                vid_w, vid_h = 1080, 1920

            if res_back:
                continue

            # Completed both Step 2 and Step 3
            break

        if profile_flow_cancelled:
            continue

        render_config = dict(config)
        render_config["video_width"] = vid_w
        render_config["video_height"] = vid_h

        # Strictly locate background from _Assets/Backgrounds
        pref_bg = selected_fx.get("background", {}).get("file")
        background_path = find_default_background(defaults_dir, preferred_file=pref_bg)

        # -------------------------------------------------------------------
        # Step 4: Review & Render
        # -------------------------------------------------------------------
        clear_screen()
        print_header("Review & Confirm")
        print(f"  Project:    {slug}")
        print(f"  Audio:      {audio_path.name}")
        print(f"  Lyrics:     {lyrics_path.name} ({karaoke_meta.get('word_count', len(raw_lines))} words)")
        sub_desc = "CUSTOM / MANUAL EDITS (Preserved)" if use_custom_ass else "AUTO-ALIGNED (Whisper AI)"
        print(f"  Subtitles:  {sub_desc}")
        print(f"  Profile:    {selected_fx.get('name', selected_profile_key)}")
        print(f"  Background: {background_path.name if background_path else 'Solid Dark'}")
        viz_name = selected_fx.get('visualizer', {}).get('type', 'none')
        print(f"  Visualizer: {viz_name.upper() if viz_name != 'none' else 'None'}")
        print(f"  Font:       {selected_fx.get('lyrics', {}).get('font', 'Chango')}")
        print(f"  Resolution: {vid_w}x{vid_h}")
        auto_lbl = "ENABLED (Folder will open automatically)" if config.get("auto_open_folder", True) else "DISABLED"
        print(f"  Auto-Open:  {auto_lbl}")
        if start_time > 0 or render_duration < total_audio_duration:
            print(f"  Clip Range: {fmt_seconds(start_time)} to {fmt_seconds(start_time + render_duration)} (Duration: {fmt_seconds(render_duration)})")
        else:
            print(f"  Duration:   {fmt_seconds(render_duration)} (Full song)")

        print("\nStart rendering?")
        print("  1. Yes, render video now (Default)")
        print("  E. Edit Lyrics (Open lyrics_easy.txt to edit words/timing)")
        print("  K. Subtitle Source (Toggle Custom vs Fresh Whisper AI)")
        print("  S. Settings & Preferences (Toggle Auto-Open Folder, Quality, FPS)")
        print("  2. Cancel & return to menu")
        print("  0. Back to Project Selection")

        confirm = _safe_input("\nSelect [1/2, E to Edit Lyrics, K for Subs, S for Settings, Enter for Yes]: ").strip()
        if confirm.lower() in ("e", "edit"):
            # Ensure lyrics_easy.txt exists
            if not easy_lyrics_path.exists() and ass_path.exists():
                export_easy_lyrics_file(ass_path, easy_lyrics_path)
            if easy_lyrics_path.exists():
                try:
                    if sys.platform == "win32":
                        os.startfile(str(easy_lyrics_path))
                    elif sys.platform == "darwin":
                        subprocess.run(["open", str(easy_lyrics_path)])
                    else:
                        subprocess.run(["xdg-open", str(easy_lyrics_path)])
                    print("\n[OK] Opened lyrics_easy.txt. Make your changes, save the file, and return here!")
                except Exception as e:
                    print(f"\n[!] Please open and edit: {easy_lyrics_path}")
            else:
                print("\n[!] Please run alignment first before editing lyrics.")
            _safe_input("\nPress Enter once you have saved your edits...")
            # If edited, flag for custom compile
            if easy_lyrics_path.exists():
                use_custom_ass = True
            continue
        if confirm.lower() in ("s", "settings"):
            settings_menu(config)
            continue
        if confirm.lower() in ("k", "subtitles", "subs"):
            if use_custom_ass:
                use_custom_ass = False
                if not karaoke_entries:
                    if not _run_alignment():
                        print("\n[ERROR] Alignment failed.")
                        use_custom_ass = True
            else:
                if ass_path.exists() or (easy_lyrics_path.exists()):
                    use_custom_ass = True
                else:
                    print("\n[!] No existing subtitles found on disk to preserve.")
                    time.sleep(1.5)
            continue
        if confirm in ("2", "0", "b", "back", "cancel", "q", "exit", "n", "no"):
            continue
        if confirm not in ("", "1", "y", "yes"):
            continue

        # -------------------------------------------------------------------
        # Write ASS Subtitles (Compile easy lyrics or fresh alignment)
        # -------------------------------------------------------------------
        dot_data = get_project_data_dir(project_folder)
        dot_data.mkdir(parents=True, exist_ok=True)
        ass_path = dot_data / "karaoke.ass"

        # Check if lyrics_easy.txt was modified or should be imported
        if easy_lyrics_path.exists():
            curr_easy_hash = sha256_file(easy_lyrics_path)
            last_easy_hash = None
            if meta_path.exists():
                try:
                    last_easy_hash = json.loads(meta_path.read_text(encoding="utf-8")).get("easy_lyrics_checksum")
                except Exception:
                    pass
            if curr_easy_hash != last_easy_hash or not ass_path.exists():
                print("\n  -> [EASY LYRICS] Transcribing user edits from lyrics_easy.txt into video subtitles...")
                try:
                    ez_entries = import_easy_lyrics_to_entries(easy_lyrics_path)
                    if ez_entries:
                        karaoke_entries = ez_entries
                        if ass_path.exists():
                            ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            bak_dest = dot_data / f"karaoke.ass.bak_{ts_str}"
                            try:
                                shutil.copy2(ass_path, bak_dest)
                            except Exception:
                                pass
                        write_ass_karaoke(
                            entries=karaoke_entries,
                            ass_path=ass_path,
                            config=render_config,
                            effect_cfg=selected_fx,
                            video_width=vid_w,
                            video_height=vid_h,
                            defaults_dir=defaults_dir,
                            time_offset=start_time,
                            end_time_limit=render_duration if start_time > 0 or render_duration < total_audio_duration else None,
                        )
                        karaoke_meta["ass_checksum"] = sha256_file(ass_path)
                        karaoke_meta["easy_lyrics_checksum"] = curr_easy_hash
                        karaoke_meta["lyrics_checksum"] = sha256_file(lyrics_path)
                        karaoke_meta["generated_at"] = now_iso()
                        save_json_atomic(dot_data / "karaoke_meta.json", karaoke_meta)
                        use_custom_ass = True
                except Exception as e:
                    print(f"  [!] Notice: could not recompile easy lyrics ({e}), falling back.")

        if not use_custom_ass:
            if ass_path.exists():
                ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                bak_dest = dot_data / f"karaoke.ass.bak_{ts_str}"
                try:
                    shutil.copy2(ass_path, bak_dest)
                except Exception:
                    pass
            write_ass_karaoke(
                entries=karaoke_entries,
                ass_path=ass_path,
                config=render_config,
                effect_cfg=selected_fx,
                video_width=vid_w,
                video_height=vid_h,
                defaults_dir=defaults_dir,
                time_offset=start_time,
                end_time_limit=render_duration if start_time > 0 or render_duration < total_audio_duration else None,
            )
            # Also generate easy-to-edit lyrics file in START HERE/_Projects/<name>
            try:
                export_easy_lyrics_file(ass_path, easy_lyrics_path)
                karaoke_meta["easy_lyrics_checksum"] = sha256_file(easy_lyrics_path)
            except Exception:
                pass
            karaoke_meta["ass_checksum"] = sha256_file(ass_path)
            karaoke_meta["lyrics_checksum"] = sha256_file(lyrics_path)
            karaoke_meta["generated_at"] = now_iso()
            save_json_atomic(dot_data / "karaoke_meta.json", karaoke_meta)
        else:
            print("\n  [INFO] Preserving existing custom karaoke.ass for this render.")
            # Ensure lyrics_easy.txt is exported if missing
            if ass_path.exists() and not easy_lyrics_path.exists():
                try:
                    export_easy_lyrics_file(ass_path, easy_lyrics_path)
                    karaoke_meta["easy_lyrics_checksum"] = sha256_file(easy_lyrics_path)
                    save_json_atomic(dot_data / "karaoke_meta.json", karaoke_meta)
                except Exception:
                    pass

        # Determine output versioned filename (strictly inspect all existing files to prevent overwriting)
        clip_tag = f"_clip_{int(start_time)}s-{int(start_time+render_duration)}s" if start_time > 0 or render_duration < total_audio_duration else ""
        target_prefix = f"{slug}{clip_tag}"
        existing_versions = [0]
        try:
            for f in project_folder.glob("*.mp4"):
                m_ver = re.search(r"\bv(\d+)\.mp4$", f.name, re.IGNORECASE)
                if m_ver:
                    existing_versions.append(int(m_ver.group(1)))
                elif f.name.lower().startswith(slug.lower()):
                    existing_versions.append(1)
        except Exception:
            pass

        ver = max(existing_versions) + 1
        while (project_folder / f"{target_prefix} v{ver}.mp4").resolve().exists():
            ver += 1
        output_path = project_folder / f"{target_prefix} v{ver}.mp4"

        print(f"\nRendering '{output_path.name}'...")
        print_wrapped("Please wait while FFmpeg composites your video with karaoke subtitles...")
        print()

        cmd = build_ffmpeg_command(
            config=render_config,
            effect_cfg=selected_fx,
            audio_path=audio_path,
            background_path=background_path,
            defaults_dir=defaults_dir,
            ass_path=ass_path,
            output_path=output_path,
            duration=render_duration,
            start_time=start_time,
        )

        success, render_time, stderr = run_render_with_progress(cmd, render_duration)

        if success:
            print("\n" + "=" * 50)
            print_wrapped("SUCCESS: Video rendered successfully!", indent="  ")
            sep = "\\" if os.name == "nt" else "/"
            print(f"  Saved to:      ...{sep}_Projects{sep}{slug}{sep}{output_path.name}")
            print(f"  Render time:   {fmt_seconds(render_time)} ({render_time:.1f}s)")
            print(f"  Output folder: Opened automatically")
            print("=" * 50)

            # Automatically open the folder and highlight the new video file
            if config.get("auto_open_folder", True):
                open_folder_and_select(output_path)

            checksums = {
                audio_path.name: sha256_file(audio_path),
                lyrics_path.name: sha256_file(lyrics_path),
            }
            if background_path:
                checksums[background_path.name] = sha256_file(background_path)

            record_iteration(
                project_folder=project_folder,
                iteration=ver,
                effect_preset=selected_profile_key,
                output_file=output_path,
                render_seconds=render_time,
                duration_seconds=render_duration,
                resolution=f"{vid_w}x{vid_h}",
                fps=config.get("fps", 30),
                source_checksums=checksums,
                notes=f"TikTok Karaoke Profile '{selected_profile_key}' ({karaoke_meta['word_count']} words, DP alignment)",
            )

            append_history(history_path, {
                "entry_id": str(uuid.uuid4())[:12],
                "slug": slug,
                "project_folder": str(project_folder.relative_to(_project_root)),
                "iteration": ver,
                "effect_preset": selected_profile_key,
                "output_file": str(output_path),
                "render_seconds": round(render_time, 2),
                "duration_seconds": round(render_duration, 2),
                "resolution": f"{vid_w}x{vid_h}",
                "fps": config.get("fps", 30),
                "completed_at": now_iso(),
            })

            try:
                import winsound
                winsound.MessageBeep(winsound.MB_ICONASTERISK)
            except Exception:
                pass

        else:
            print("\n[ERROR] FFmpeg rendering failed.")
            if stderr:
                print("\n---- FFmpeg Error Log ----")
                print(stderr[-2000:])
                print("--------------------------")

        _safe_input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nExiting.")
        sys.exit(0)
    except Exception as e:
        print("\n" + "=" * 60)
        print("ERROR: An unexpected error occurred.")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        print("=" * 60)
        _safe_input("\nPress Enter to exit...")
        sys.exit(1)
