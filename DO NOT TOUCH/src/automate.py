#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shotcut TikTok Automator
========================
Double-click this file or run:  python automate.py

Workflow:
  1. Pick a project folder (must contain MP3 + TXT lyrics)
  2. Pick a visual style (Default / Random / Custom / Saved)
  3. Review choices
  4. Approve execution
  5. Watch progress bar
  6. Choose: another project, same project different look, or exit

Features:
  - Word-by-word karaoke highlighting (ASS karaoke subtitles)
  - Cybernetic WMP XP dark theme presets
  - Version-stacked outputs (never overwrites)
  - Machine-readable JSON history (render-history.json)
  - Per-project completion markers (.completed.json)

All settings live in config.json and effects.json next to this script.
Only files inside the TikTok Wizard root directory are touched.
"""

import json
import os
import sys
import subprocess
import winsound
import hashlib
import shutil
import uuid
import random
import re
from bisect import bisect_right

try:
    import numpy as np
except ImportError:
    np = None

# Whisper forced-alignment stack (optional; preferred timing engine when present)
try:
    from faster_whisper import WhisperModel as _WhisperModel
    WHISPER_AVAILABLE = True
except Exception:
    _WhisperModel = None
    WHISPER_AVAILABLE = False
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ---------------------------------------------------------------------------
# Paths & constants (always relative to THIS script's folder)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"
EFFECTS_PATH = ROOT / "effects.json"
HISTORY_PATH = ROOT / "render-history.json"

DEFAULT_CONFIG = {
    "schema": "shotcut_tiktok.config.v1",
    "projects_dir": "Projects",
    "defaults_dir": "Projects/defaults",
    "build_dir": "build",
    "output_dir": "output",
    "history_file": "render-history.json",
    "video_width": 1080,
    "video_height": 1920,
    "fps": 30,
    "preset": "youtube",
    "codec": "h264",
    "audio_codec": "aac",
    "crf": 23,
    "default_effect_preset": "cybernetic_wmp_xp_dark",
    "lyric_sync_engine": "simple",
    "ffmpeg_path": "ffmpeg",
    "melt_path": "melt",
    "safe_mode": True,
    "karaoke": {
        "enabled": True,
        "highlight_color": "&H0000FFFF&",
        "base_color": "&H00FFFFFF&",
        "outline_color": "&H00000000&",
        "font": "Arial Black",
        "font_size": 48,
        "outline_width": 3,
        "shadow": True,
        "position_y_percent": 85,
        "safe_zone_pct": 90
    }
}

DEFAULT_EFFECTS = {
    "schema": "shotcut_tiktok.effects.v1",
    "presets": {
        "cybernetic_wmp_xp_dark": {
            "name": "Cybernetic WMP XP Dark",
            "description": "Dark neon grid background with scanlines, cyan/magenta accents.",
            "background": {
                "type": "generated",
                "theme": "wmp_xp_dark",
                "colors": ["#0a0a0a", "#1a1a2e"],
                "elements": ["grid", "scanlines", "corner_brackets"]
            },
            "blur": {"type": "gaussian", "amount": 12},
            "visualizer": {
                "type": "audio_spectrum",
                "color": "#00ffff",
                "position": "center_bottom",
                "height_pct": 25
            },
            "lyrics": {
                "style": "tiktok_outline",
                "position": "bottom_center",
                "font": "Arial Black",
                "size_pct": 7,
                "fill_color": "#ffffff",
                "outline_color": "#000000",
                "outline_width_px": 3,
                "shadow": True,
                "background_pill": True,
                "pill_color": "#00000088",
                "fade_in_sec": 0.25,
                "fade_out_sec": 0.25,
                "align": "center",
                "safe_zone_pct": 90
            }
        },
        "tiktok_standard": {
            "name": "TikTok Standard",
            "description": "Clean white text on semi-black pill, subtle background blur.",
            "background": {
                "type": "source",
                "theme": "source_blur",
                "colors": [],
                "elements": ["blur"]
            },
            "blur": {"type": "box", "amount": 20},
            "visualizer": {
                "type": "audio_waveform",
                "color": "#ffffff",
                "position": "center_bottom",
                "height_pct": 20
            },
            "lyrics": {
                "style": "tiktok_pill",
                "position": "bottom_center",
                "font": "Arial Black",
                "size_pct": 7,
                "fill_color": "#ffffff",
                "outline_color": "#000000",
                "outline_width_px": 2,
                "shadow": True,
                "background_pill": True,
                "pill_color": "#000000aa",
                "fade_in_sec": 0.2,
                "fade_out_sec": 0.2,
                "align": "center",
                "safe_zone_pct": 90
            }
        },
        "neon_cyberpunk": {
            "name": "Neon Cyberpunk",
            "description": "Dark magenta/cyan grid, glitch visualizer, outlined lyrics.",
            "background": {
                "type": "generated",
                "theme": "cyber_grid",
                "colors": ["#1a001a", "#001a1a"],
                "elements": ["grid", "scanlines", "glow"]
            },
            "blur": {"type": "gaussian", "amount": 8},
            "visualizer": {
                "type": "audio_spectrum",
                "color": "#ff00ff",
                "position": "center_bottom",
                "height_pct": 30
            },
            "lyrics": {
                "style": "tiktok_outline",
                "position": "bottom_center",
                "font": "Impact",
                "size_pct": 8,
                "fill_color": "#ffffff",
                "outline_color": "#ff00ff",
                "outline_width_px": 4,
                "shadow": True,
                "background_pill": False,
                "pill_color": "#00000000",
                "fade_in_sec": 0.15,
                "fade_out_sec": 0.15,
                "align": "center",
                "safe_zone_pct": 90
            }
        },
        "dark_minimal": {
            "name": "Dark Minimal",
            "description": "Pure black background, white minimal lyrics, no visualizer.",
            "background": {
                "type": "generated",
                "theme": "solid_dark",
                "colors": ["#000000", "#000000"],
                "elements": []
            },
            "blur": {"type": "none", "amount": 0},
            "visualizer": {"type": "none"},
            "lyrics": {
                "style": "subtitles_plain",
                "position": "bottom_center",
                "font": "Arial",
                "size_pct": 6,
                "fill_color": "#ffffff",
                "outline_color": "#000000",
                "outline_width_px": 2,
                "shadow": False,
                "background_pill": False,
                "pill_color": "#00000000",
                "fade_in_sec": 0.1,
                "fade_out_sec": 0.1,
                "align": "center",
                "safe_zone_pct": 90
            }
        }
    },
    "random_pool": [
        "cybernetic_wmp_xp_dark",
        "tiktok_standard",
        "neon_cyberpunk",
        "dark_minimal"
    ],
    "components": {
        "visualizers": ["audio_spectrum", "audio_waveform", "audio_light", "none"],
        "blurs": ["gaussian", "box", "none"],
        "lyric_styles": ["tiktok_outline", "tiktok_pill", "subtitles_plain"],
        "backgrounds": ["wmp_xp_dark", "cyber_grid", "solid_dark", "source_blur"]
    }
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json_atomic(path: Path, data: Any) -> None:
    """Write JSON atomically (temp file + rename) so crashes don't corrupt it."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fmt_seconds(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def srt_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    h = total_ms // 3_600_000
    total_ms %= 3_600_000
    m = total_ms // 60_000
    total_ms %= 60_000
    s = total_ms // 1000
    ms = total_ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def ass_timestamp(seconds: float) -> str:
    """ASS timestamp format: H:MM:SS.cc (centiseconds)"""
    total_cs = int(round(seconds * 100))
    h = total_cs // 360000
    total_cs %= 360000
    m = total_cs // 6000
    total_cs %= 6000
    s = total_cs // 100
    cs = total_cs % 100
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def pause(message: str = "\nPress Enter to continue...") -> None:
    _safe_input(message)


# ---------------------------------------------------------------------------
# Config / effects loading
# ---------------------------------------------------------------------------
def load_config() -> Dict:
    cfg = load_json(CONFIG_PATH, None)
    if cfg is None:
        save_json_atomic(CONFIG_PATH, DEFAULT_CONFIG)
        return dict(DEFAULT_CONFIG)
    merged = dict(DEFAULT_CONFIG)
    merged.update(cfg)
    # ensure nested karaoke dict exists
    if "karaoke" not in merged or not isinstance(merged["karaoke"], dict):
        merged["karaoke"] = dict(DEFAULT_CONFIG["karaoke"])
    else:
        karaoke_defaults = dict(DEFAULT_CONFIG["karaoke"])
        karaoke_defaults.update(merged["karaoke"])
        merged["karaoke"] = karaoke_defaults
    return merged


def load_effects() -> Dict:
    fx = load_json(EFFECTS_PATH, None)
    if fx is None:
        save_json_atomic(EFFECTS_PATH, DEFAULT_EFFECTS)
        return dict(DEFAULT_EFFECTS)
    return fx


# ---------------------------------------------------------------------------
# Project discovery
# ---------------------------------------------------------------------------
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac"}
LYRIC_EXTS = {".txt", ".lrc", ".srt"}
BACKGROUND_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".jpg", ".jpeg", ".png", ".gif", ".webp"}


def find_projects(projects_dir: Path) -> List[Path]:
    if not projects_dir.exists():
        return []
    candidates = []
    for folder in sorted(projects_dir.iterdir()):
        if not folder.is_dir():
            continue
        # skip defaults folder if it's inside projects
        if folder.name.lower() == "defaults":
            continue
        files = {p.suffix.lower() for p in folder.iterdir() if p.is_file()}
        if files & AUDIO_EXTS and files & LYRIC_EXTS:
            candidates.append(folder)
    return candidates


# Versioned render outputs produced by this script look like "<slug> v<N>.mp4".
_RENDER_OUTPUT_RE = re.compile(r"\bv\d+\s*$", re.IGNORECASE)


def find_background(folder: Path) -> Optional[Path]:
    """Return a user-supplied background asset, ignoring our own render outputs.

    IMPORTANT: previous karaoke renders live in the same project folder (e.g.
    "Song v1.mp4") and already have lyrics burned in. Reusing one as the
    background makes OLD captions appear in the new video, so skip them.
    When nothing else is found the caller falls back to the default background.
    """
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.suffix.lower() not in BACKGROUND_EXTS:
            continue
        # Skip versioned render outputs like "PaoPao - Grax v1", "... v12"
        if _RENDER_OUTPUT_RE.search(p.stem):
            continue
        return p
    return None


def find_audio(folder: Path) -> Optional[Path]:
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            return p
    return None


def find_lyrics(folder: Path) -> Optional[Path]:
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in LYRIC_EXTS:
            return p
    return None


# ---------------------------------------------------------------------------
# Defaults folder (one-time background assets)
# ---------------------------------------------------------------------------
def find_default_background(defaults_dir: Path) -> Optional[Path]:
    if not defaults_dir.exists():
        return None
    for ext in (".mp4", ".mov", ".jpg", ".jpeg", ".png", ".gif"):
        matches = list(defaults_dir.glob(f"*{ext}"))
        if matches:
            return matches[0]
    return None


# ---------------------------------------------------------------------------
# History (global render-history.json)
# ---------------------------------------------------------------------------
def load_history(path: Path) -> Dict:
    data = load_json(path, None)
    if data is None:
        data = {
            "schema": "shotcut_tiktok.history.v1",
            "created_at": now_iso(),
            "last_updated": now_iso(),
            "total_renders": 0,
            "entries": [],
        }
        save_json_atomic(path, data)
    return data


def append_history(path: Path, entry: Dict) -> None:
    data = load_history(path)
    data["entries"].append(entry)
    data["total_renders"] = len(data["entries"])
    data["last_updated"] = now_iso()
    save_json_atomic(path, data)


# ---------------------------------------------------------------------------
# Per-project completion marker (.completed.json) - history, not a gate
# ---------------------------------------------------------------------------
def load_project_completion(project_folder: Path) -> Dict:
    dot_data = project_folder / ".data"
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


def save_project_completion(project_folder: Path, data: Dict) -> None:
    dot_data = project_folder / ".data"
    dot_data.mkdir(parents=True, exist_ok=True)
    save_json_atomic(dot_data / ".completed.json", data)


def record_iteration(
    project_folder: Path,
    iteration: int,
    effect_preset: str,
    resolved_preset: Optional[str],
    effect_config_file: str,
    output_file: str,
    render_seconds: float,
    duration_seconds: float,
    resolution: str,
    fps: int,
    source_checksums: Dict[str, str],
    notes: str = "",
) -> None:
    comp = load_project_completion(project_folder)
    if comp["first_completed_at"] is None:
        comp["first_completed_at"] = now_iso()
    comp["last_completed_at"] = now_iso()
    comp["total_iterations"] = iteration

    entry = {
        "iteration": iteration,
        "effect_preset": effect_preset,
        "resolved_preset": resolved_preset,
        "effect_config_file": effect_config_file,
        "output_file": output_file,
        "completed_at": now_iso(),
        "render_seconds": round(render_seconds, 2),
        "duration_seconds": round(duration_seconds, 2),
        "resolution": resolution,
        "fps": fps,
        "source_checksums": source_checksums,
        "notes": notes,
    }
    comp["iterations"].append(entry)
    save_project_completion(project_folder, comp)


# ---------------------------------------------------------------------------
# State (per-build state.json) - resume friendly
# ---------------------------------------------------------------------------
def init_state(build_dir: Path, slug: str, stage: str = "pending") -> Path:
    ensure_dir(build_dir)
    state_path = build_dir / "state.json"
    state = {
        "schema": "shotcut_tiktok.state.v1",
        "slug": slug,
        "stage": stage,
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "attempt": 1,
        "artifacts": {},
        "error": None,
    }
    save_json_atomic(state_path, state)
    return state_path


def update_state(state_path: Path, **fields) -> None:
    state = load_json(state_path, {})
    state.update(fields)
    state["updated_at"] = now_iso()
    save_json_atomic(state_path, state)


# ---------------------------------------------------------------------------
# Lyrics parsing -> word-level timing
# ---------------------------------------------------------------------------
def read_lyrics_raw(lyrics_path: Path) -> str:
    return lyrics_path.read_text(encoding="utf-8")


# Inline timestamp regexes. Supported formats (all handled here):
#   [SS.mmm]          -> e.g. [05.500]Hola            (seconds.millis)
#   [MM:SS.mmm]       -> e.g. [00:05.500]Hola          (minutes.seconds.millis)
#   [H:MM:SS.mmm]     -> e.g. [01:12:34.5]Hola         (hours:minutes:seconds.millis)
TIMESTAMP_RE = re.compile(
    r'^\s*\[?(?P<ts>\d{1,2}:\d{2}(?::\d{2})?[.,]\d{1,3}|\d{1,2}[.,]\d{1,3})\]?\s*(?P<rest>.*)$'
)
WORD_TIMESTAMP_RE = re.compile(
    r'\[(?P<ts>\d{1,2}:\d{2}(?::\d{2})?[.,]\d{1,3}|\d{1,2}[.,]\d{1,3})\]\s*(?P<word>\S+)'
)


def _ts_to_seconds(ts: str) -> float:
    """Convert a timestamp string like '05.5', '00:05.5', or '01:12:34.5' -> seconds."""
    if "." in ts:
        head, frac = ts.split(".", 1)
        frac = (frac + "000")[:3]
        ms = int(frac)
    else:
        head, ms = ts, 0
    parts = [p for p in head.split(":") if p != ""]
    nums = [int(p) for p in parts]
    if len(nums) == 1:
        h, m, s = 0, 0, nums[0]
    elif len(nums) == 2:
        h, m, s = 0, nums[0], nums[1]
    else:
        h, m, s = nums[0], nums[1], nums[2]
    return h * 3600 + m * 60 + s + ms / 1000.0


def parse_time_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")[:3]) / 1000.0


def parse_lyrics_with_words(
    lyrics_path: Path,
    total_duration: float,
    offset_seconds: float = 0.0,
    vocal_regions: Optional[List[Tuple[float, float]]] = None,
) -> List[Dict]:
    """
    Parse lyrics.txt into word-level timing entries.

    Supports two formats:
    1. Plain text (one phrase per line):
       Llevo días que no salgo de la casa
       Que no salgo de mi mente
    2. Inline word timestamps:
       [00:00.50]Llevo [00:01.20]días [00:01.80]que

    Returns list of dicts: {"word": str, "start": float, "end": float, "line": int}
    """
    raw = read_lyrics_raw(lyrics_path)
    lines = [ln.rstrip("\r") for ln in raw.split("\n")]
    # Remove empty trailing lines but keep structure
    non_empty = [(i, ln) for i, ln in enumerate(lines) if ln.strip()]
    if not non_empty:
        return []

    total_words = 0
    for _, ln in non_empty:
        total_words += len(WORD_TIMESTAMP_RE.findall(ln)) or len(ln.split())

    # If any line has inline timestamps, use them; otherwise equal distribution
    has_timestamps = any(WORD_TIMESTAMP_RE.search(ln) for _, ln in non_empty)

    entries: List[Dict] = []
    mapped_mode = False
    if has_timestamps:
        # Collect all timestamped words globally, then chain end-times
        # (each word ends when the next word starts; last ends at total_duration).
        timed: List[Dict] = []
        for line_idx, ln in non_empty:
            matches = list(WORD_TIMESTAMP_RE.finditer(ln))
            if not matches:
                continue
            for m in matches:
                start = _ts_to_seconds(m.group("ts"))
                timed.append({
                    "word": m.group("word"),
                    "start": start,
                    "end": start + 0.5,
                    "line": line_idx,
                })
        timed.sort(key=lambda e: e["start"])
        for j, ev in enumerate(timed):
            if j + 1 < len(timed):
                ev["end"] = timed[j + 1]["start"]
            else:
                ev["end"] = total_duration
        entries.extend(timed)
    else:
        # Equal duration per line, then equal duration per word within line.
        # If vocal regions were detected, distribute words ONLY across those
        # regions (active seconds), mapping back into real audio time, so the
        # caption never advances during instrumental sections.
        line_count = len(non_empty)
        effective_duration = total_duration
        mapper = None
        total_active = 0.0
        if vocal_regions:
            mapper, total_active = _map_active_to_real(vocal_regions)
            if mapper is not None and total_active > 0:
                effective_duration = total_active
                mapped_mode = True

        if not mapped_mode:
            # Reserve trailing padding so words don't extend into end-of-song instrumental
            END_PADDING = 2.0  # seconds of instrumental silence at end of song
            available_duration = max(effective_duration - 2.0, effective_duration * 0.9) - offset_seconds
            if available_duration <= 0:
                available_duration = effective_duration
            effective_duration = available_duration + offset_seconds

        single_map = mapped_mode and total_active > 0
        per_line = (total_active if single_map else effective_duration - offset_seconds) / max(1, line_count)
        current_a = offset_seconds if not single_map else 0.0  # active-domain cursor

        for line_idx, (orig_idx, ln) in enumerate(non_empty):
            words = ln.split()
            if not words:
                current_a += per_line
                continue
            per_word = per_line / len(words)
            for w in words:
                w_start_a = current_a
                w_end_a = current_a + per_word
                if single_map:
                    w_start = mapper(w_start_a)
                    w_end = mapper(w_end_a)
                else:
                    w_start = w_start_a
                    w_end = w_end_a
                if w_start >= w_end:
                    w_end = w_start + max(0.05, per_word * 0.5)
                entries.append({
                    "word": w,
                    "start": w_start,
                    "end": w_end,
                    "line": line_idx,
                })
                current_a = w_end_a

        if single_map and entries:
            # Clamp placements into the real audio timeline
            for e in entries:
                if e["start"] > total_duration:
                    e["start"] = total_duration

    # Clamp ends safely within meaningful audio duration
    for e in entries:
        if e["end"] > total_duration + 0.5:
            e["end"] = total_duration + 0.5
        if e["start"] >= e["end"]:
            e["end"] = e["start"] + 0.1
        # Mapped mode: never let a word's end extend past the end of the vocal
        # region it was placed in — this prevents captions into instrumentals.
        if mapped_mode and vocal_regions:
            for (rs, re_) in vocal_regions:
                if rs - 0.5 <= e["start"] <= re_:
                    e["end"] = min(e["end"], re_)
                    break

    return entries


def _find_font_dir(font_name: str, font_file: Optional[str] = None, defaults_dir: Optional[Path] = None) -> Optional[Path]:
    """Return the directory containing the matching font file, or None.

    Used to build the fontsdir= option for ffmpeg's ass filter so libass
    can resolve the font family name.
    """
    # 1. Explicit font_file from config
    if font_file:
        pf = Path(font_file)
        if pf.exists():
            return pf.parent

    # 2. Scan defaults folder for font files
    if defaults_dir and defaults_dir.exists():
        font_variants = [font_name, font_name.replace(" ", ""), font_name.lower().replace(" ", "")]
        try:
            for f in defaults_dir.iterdir():
                if not f.is_file():
                    continue
                f_lower = f.name.lower().replace(" ", "").replace("-", "").replace("_", "")
                if not f_lower.endswith((".ttf", ".ttc", ".otf")):
                    continue
                for variant in font_variants:
                    v_lower = variant.lower().replace(" ", "").replace("-", "").replace("_", "")
                    if f_lower.startswith(v_lower) or v_lower in f_lower:
                        return defaults_dir
        except Exception:
            pass

    # 3. Check Windows system font directories
    font_dirs = [
        r"C:\Windows\Fonts",
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Windows\Fonts"),
    ]
    font_variants = [font_name, font_name.replace(" ", ""), font_name.lower().replace(" ", "")]
    for fdir in font_dirs:
        if not os.path.isdir(fdir):
            continue
        try:
            for fname in os.listdir(fdir):
                f_lower = fname.lower().replace(" ", "").replace("-", "").replace("_", "")
                for variant in font_variants:
                    v_lower = variant.lower().replace(" ", "").replace("-", "").replace("_", "")
                    if f_lower.startswith(v_lower) and f_lower.endswith((".ttf", ".ttc", ".otf")):
                        return Path(fdir)
        except Exception:
            pass
    return None


def _resolve_font(font_name: str, font_file: Optional[str] = None, defaults_dir: Optional[Path] = None) -> str:
    """Return the font family name for use in the ASS style."""
    return font_name


def write_ass_karaoke(
    word_entries: List[Dict],
    ass_path: Path,
    config: Dict,
    effect_cfg: Dict,
    video_width: int,
    video_height: int,
    defaults_dir: Optional[Path] = None,
) -> None:
    """
    Generate an ASS karaoke subtitle file with per-line visibility.

    All words for each sentence appear together when the first word is sung (-0.2s before).
    The sentence disappears when the last word finishes singing.
    \k tags highlight each word individually as it's sung.
    Fully dynamic — works for any audio duration, no hardcoded values.
    """
    karaoke_cfg = config.get("karaoke", {})
    font = _resolve_font(karaoke_cfg.get("font", "Arial Black"), karaoke_cfg.get("font_file"), defaults_dir)
    font_size = karaoke_cfg.get("font_size", 80)
    base_color = karaoke_cfg.get("base_color", "&H00808000&")   # dim cyan — visible but not highlighted
    highlight_color = karaoke_cfg.get("highlight_color", "&H00FFFFFF&")  # white — highlighted when sung
    outline_color = karaoke_cfg.get("outline_color", "&H00FFFFFF&")
    outline_width = karaoke_cfg.get("outline_width", 6)   # big outline so text pops
    shadow = karaoke_cfg.get("shadow", True)
    shadow_depth = karaoke_cfg.get("shadow_depth", 4)     # big shadow depth
    pos_y_pct = karaoke_cfg.get("position_y_percent", 85)

    # Build ASS header
    lines = []
    lines.append("[Script Info]")
    lines.append("Title: TikTok Karaoke")
    lines.append("ScriptType: v4.00+")
    lines.append("WrapStyle: 0")
    lines.append("ScaledBorderAndShadow: yes")
    lines.append("YCbCr Matrix: TV.601")
    lines.append(f"PlayResX: {video_width}")
    lines.append(f"PlayResY: {video_height}")
    lines.append("")
    lines.append("[V4+ Styles]")
    lines.append("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding")
    shadow_val = shadow_depth if shadow else 0
    style_line = (
        # PrimaryColour = sung word (highlight), SecondaryColour = unsung word (base).
        # libass \k transitions SecondaryColour -> PrimaryColour over the word duration.
        f"Style: Default,{font},{font_size},{highlight_color},{base_color},"
        f"{outline_color},&H00000000&,-1,0,0,0,100,100,0,0,1,{outline_width},{shadow_val},2,20,20,20,1"
    )
    lines.append(style_line)
    lines.append("")
    lines.append("[Events]")
    lines.append("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text")

    # Group word entries by line
    line_groups: Dict[int, List[Dict]] = {}
    for entry in word_entries:
        line_idx = entry["line"]
        if line_idx not in line_groups:
            line_groups[line_idx] = []
        line_groups[line_idx].append(entry)

    # Sort line groups by line index
    sorted_line_indices = sorted(line_groups.keys())

    # Build one Dialogue event per contiguous word run (a line whose words span
    # a vocal-region gap becomes multiple events; also prevents line overlap).
    prev_line_end: Optional[float] = None
    for line_idx in sorted_line_indices:
        entries = line_groups[line_idx]
        # Sort entries within line by start time
        entries.sort(key=lambda e: e["start"])

        # Split entries into contiguous runs (gap > 0.4s starts a new run)
        runs: List[List[Dict]] = []
        current_run: List[Dict] = []
        prev_end: Optional[float] = None
        for e in entries:
            if current_run and prev_end is not None and (e["start"] - prev_end) > 0.4:
                runs.append(current_run)
                current_run = []
            current_run.append(e)
            prev_end = e["end"]
        if current_run:
            runs.append(current_run)

        for run in runs:
            line_start = max(0.0, run[0]["start"] - 0.2)
            line_end = run[-1]["end"]

            # Never let consecutive lines overlap: line N+1 must not start before
            # line N ends (this was causing lyrics to stack over each other).
            if prev_line_end is not None and line_start < prev_line_end:
                line_start = prev_line_end
            if line_end < line_start:
                line_end = line_start
            prev_line_end = line_end

            # Build text with \k karaoke tags for each word
            text_parts = []
            for i, entry in enumerate(run):
                word = entry["word"]
                duration_cs = max(1, int(round((entry["end"] - entry["start"]) * 100)))
                if i > 0:
                    text_parts.append(" ")
                text_parts.append(f"{{\\k{duration_cs}}}{word}")

            text = "".join(text_parts)

            # Position: bottom center
            y_px = int(video_height * pos_y_pct / 100)
            x_px = video_width // 2

            lines.append(
                f"Dialogue: 0,{ass_timestamp(line_start)},{ass_timestamp(line_end)},"
                f"Default,,0,0,0,,{{\\an8\\pos({x_px},{y_px})}}{text}"
            )

    ass_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# ffmpeg probe
# ---------------------------------------------------------------------------
def probe_duration(ffmpeg_path: str, audio_path: Path) -> Optional[float]:
    try:
        result = subprocess.run(
            [
                ffmpeg_path.replace("ffmpeg", "ffprobe"),
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        val = result.stdout.strip()
        return float(val) if val else None
    except Exception:
        return None


def detect_vocal_onset(ffmpeg_path: str, audio_path: Path) -> float:
    """
    Detect the first vocal onset using ffmpeg.
    Uses silencedetect first, then falls back to amplitude-based detection
    for songs that start with instruments (no silence at the beginning).
    Returns time in seconds when vocals/speech first start.
    """
    try:
        # --- Attempt 1: silencedetect (catches quiet intros) ---
        cmd = [
            ffmpeg_path,
            "-i", str(audio_path),
            "-af", "silencedetect=noise=-40dB:d=0.3",
            "-f", "null", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        silence_end_times = []
        for line in result.stderr.split("\n"):
            if "silence_end:" in line:
                match = re.search(r"silence_end:\s*([\d.]+)", line)
                if match:
                    silence_end_times.append(float(match.group(1)))

        if silence_end_times:
            # The FIRST silence_end marks when the leading silence ends — position of sound onset
            return silence_end_times[0]

        # --- Attempt 2: Amplitude fallback ---
        # If silencedetect found nothing, the song likely starts with sound.
        # Check first 10 seconds for any non-silent region.
        try:
            cmd3 = [
                ffmpeg_path,
                "-i", str(audio_path),
                "-ar", "44100",
                "-ac", "1",
                "-t", "10.0",
                "-f", "s16le",
                "-",
            ]
            result3 = subprocess.run(cmd3, capture_output=True, timeout=45)
            if result3.stdout:
                samples = len(result3.stdout) // 2
                if samples > 0:
                    min_amp = 0.015  # ~ -36dBFS
                    slice_size = max(1, samples // 10)
                    for i in range(0, min(slice_size * 10, samples), slice_size):
                        chunk = result3.stdout[i:i + slice_size * 2]
                        if len(chunk) < slice_size * 2:
                            continue
                        max_val = max(
                            abs(int.from_bytes(chunk[j:j + 2], "little", signed=True))
                            for j in range(0, len(chunk), 2)
                        )
                        if max_val > min_amp * 32768:
                            return _precise_onset_from_energy(
                                audio_path, ffmpeg_path, min_amp, result3.stdout, samples
                            )
        except Exception:
            pass

        # --- Attempt 3: Very aggressive silencedetect (last resort) ---
        try:
            cmd4 = [
                ffmpeg_path,
                "-i", str(audio_path),
                "-af", "silencedetect=noise=-60dB:d=0.1",
                "-f", "null", "-",
            ]
            result4 = subprocess.run(cmd4, capture_output=True, text=True, timeout=60)
            for line in result4.stderr.split("\n"):
                if "silence_end:" in line:
                    match = re.search(r"silence_end:\s*([\d.]+)", line)
                    if match:
                        t = float(match.group(1))
                        if t > 0.1:
                            return t
        except Exception:
            pass

        return 0.0
    except Exception:
        return 0.0


def _precise_onset_from_energy(
    audio_path: Path,
    ffmpeg_path: str,
    min_amp: float,
    stdout_data: bytes,
    samples: int,
) -> float:
    """Binary-search precise onset time from amplitude data (10s window)."""
    try:
        frame_duration = 1.0 / 44100.0
        slice_size = max(1, samples // 10)
        for i in range(0, min(slice_size * 10, samples), slice_size):
            chunk = stdout_data[i:i + slice_size * 2]
            if len(chunk) < slice_size * 2:
                continue
            max_val = max(
                abs(int.from_bytes(chunk[j:j + 2], "little", signed=True))
                for j in range(0, len(chunk), 2)
            )
            if max_val > min_amp * 32768:
                # Binary search within this slice for onset precision
                lo = i * frame_duration
                hi = min((i + slice_size) * frame_duration, 10.0)
                for _ in range(8):
                    mid = (lo + hi) / 2
                    pos = int(mid * 44100.0)
                    if pos >= samples:
                        hi = mid
                        continue
                    sub = stdout_data[pos:pos + 2]
                    if len(sub) < 2:
                        hi = mid
                        continue
                    val = abs(int.from_bytes(sub, "little", signed=True))
                    if val > min_amp * 32768:
                        hi = mid
                    else:
                        lo = mid
                return max(lo, 0.01)
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Hybrid vocal-region detection (local, adaptive, for karaoke gating)
# ---------------------------------------------------------------------------

def detect_vocal_regions(ffmpeg_path: str, audio_path: Path) -> List[Tuple[float, float]]:
    """
    Detect time windows where singing is active, fully locally and adaptively.

    Strategy (no ML models, no hardcoded song structure):
      - Decode mono 16kHz PCM
      - Window 1024 / hop 512
      - Vocal frames: vocal-band (150-3000Hz) energy dominates the mix,
        bass band (20-150Hz) is subdued, and overall level is above an
        adaptive RMS floor (percentile-based, not hardcoded for any song)
      - Morphological smoothing merges micro-gaps (vocal chops) and drops
        isolated misclassified blips

    Returns list of (start_sec, end_sec) vocal regions.
    Empty list means "detection unreliable / no vocals" — caller falls back.
    """
    if np is None:
        return []

    try:
        SR = 16000
        WINDOW = 1024
        HOP = 512

        cmd = [
            ffmpeg_path,
            "-v", "quiet",
            "-i", str(audio_path),
            "-ar", str(SR),
            "-ac", "1",
            "-f", "s16le",
            "-",
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        raw = result.stdout
        if not raw:
            return []
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        n = len(samples)
        if n < WINDOW * 4:
            return []

        freqs = np.fft.rfftfreq(WINDOW, 1.0 / SR)
        low = (freqs >= 20) & (freqs <= 150)
        mid = (freqs >= 150) & (freqs <= 3000)

        nf = (n - WINDOW) // HOP + 1
        ratio = np.empty(nf, dtype=np.float64)   # mid / total
        low_mid = np.empty(nf, dtype=np.float64) # low / mid
        rms = np.empty(nf, dtype=np.float64)
        window = np.hanning(WINDOW)

        for i in range(nf):
            frame = samples[i * HOP: i * HOP + WINDOW] * window
            spec = np.abs(np.fft.rfft(frame)) ** 2
            tot = spec.sum() + 1e-12
            mid_e = spec[mid].sum() + 1e-12
            ratio[i] = spec[mid].sum() / tot
            low_mid[i] = spec[low].sum() / mid_e
            rms[i] = np.sqrt(np.mean(frame ** 2)) + 1e-12

        # Adaptive RMS floor: everything below p15 is silence/atmospheric pad
        rms_floor = max(np.percentile(rms, 15) * 1.15, 0.006)
        # Vocal frames: mid-band dominant, subdued bass, above quiet floor
        vocal = (ratio > 0.60) & (low_mid < 2.0) & (rms > rms_floor)

        # --- Sliding 1-second density window ---
        # Sparse misclassified blips (synth stabs) can't activate a second;
        # only dense vocal passages register as active. Adaptive threshold
        # relative to this song's own density distribution.
        frames_per_sec = max(1, int(SR / HOP))
        n_sec = nf // frames_per_sec
        if n_sec < 8:
            # Too little audio for reliable segmentation — treat as instrumental
            return []

        sec_density = vocal[: n_sec * frames_per_sec].reshape(n_sec, frames_per_sec).mean(axis=1)
        # Adaptive threshold: clearly instrumental seconds sit far below dense ones.
        # Take the midpoint of the two dominant density clusters (bimodal valley).
        # Floor raised to 0.35 so sparse instrumental stretches aren't misread as vocals.
        p_low = np.percentile(sec_density, 20)
        p_high = np.percentile(sec_density, 80)
        thr = min(0.55, max(0.35, (p_low + p_high) / 2.0))
        active_sec = sec_density >= thr

        # Build contiguous active-second segments, merging short gaps (<= 2s)
        segs: List[Tuple[int, int]] = []
        i = 0
        while i < n_sec:
            if not active_sec[i]:
                i += 1
                continue
            j = i
            last_active = i
            while j < n_sec:
                if active_sec[j]:
                    last_active = j
                    j += 1
                elif j - last_active <= 2:
                    j += 1
                else:
                    break
            segs.append((i, last_active + 1))
            i = last_active + 1

        # Drop segments that are too short to be real singing passages, and
        # segments whose average vocal density is too low (sparse misclassifications).
        min_seg_sec = 2.5
        min_seg_density = 0.40
        filtered_segs: List[Tuple[int, int]] = []
        for s, e in segs:
            if (e - s) < min_seg_sec:
                continue
            if sec_density[s:e].mean() < min_seg_density:
                continue
            filtered_segs.append((s, e))
        segs = filtered_segs
        if not segs:
            return []

        regions = [(float(s), float(e)) for s, e in segs]
        return regions

    except Exception:
        return []


def _map_active_to_real(regions: List[Tuple[float, float]]):
    """
    Build an interpolation mapping active-domain seconds -> real audio seconds,
    and return (map_func, total_active). Used to schedule lyrics/words only
    across the detected vocal regions.
    """
    starts = [s for s, _ in regions]
    durs = [e - s for s, e in regions]
    cum = [0.0]
    for d in durs:
        cum.append(cum[-1] + d)
    total_active = cum[-1]

    def map_t(a: float) -> float:
        a = max(0.0, min(a, total_active))
        # Bisect: find segment idx with cum[idx] <= a < cum[idx+1]
        idx = bisect_right(cum, a) - 1
        if idx < 0:
            idx = 0
        if idx >= len(regions):
            idx = len(regions) - 1
        return regions[idx][0] + (a - cum[idx])

    return map_t, total_active


# ---------------------------------------------------------------------------
# Whisper forced alignment (optional, preferred when available)
# ---------------------------------------------------------------------------

def transcribe_words(
    audio_path: Path,
    language: str = "es",
    model_size: str = "large-v3",
) -> Optional[List[Dict]]:
    """
    Run Whisper (faster-whisper backend) to get word-level timestamps.

    Returns list of {"word": str, "start": float, "end": float} or None if the
    stack is unavailable / fails.
    """
    if not WHISPER_AVAILABLE or _WhisperModel is None:
        return None
    try:
        model = _WhisperModel(model_size, device="auto", compute_type="default")
        segments, _info = model.transcribe(str(audio_path), language=language, condition_on_previous_text=False)
        words: List[Dict] = []
        for seg in segments:
            for w in seg.words:
                words.append({
                    "word": str(w.word),
                    "start": float(w.start),
                    "end": float(w.end),
                })
        return words if words else None
    except Exception:
        return None


def _norm_word(s: str) -> str:
    """Lowercase + strip punctuation for fuzzy lyric/word matching."""
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def assign_whisper_words_to_lines(
    whisper_words: List[Dict],
    lyric_lines: List[List[str]],
) -> List[Dict]:
    """
    Map Whisper word timestamps onto lyric lines (karaoke entries).

    whisper_words: ordered [{"word","start","end"}] from Whisper.
    lyric_lines: ordered list of word-lists (one per lyric line, non-empty lines only).

    Returns entries shaped like parse_lyrics_with_words output:
    [{"word","start","end","line"}]
    """
    entries: List[Dict] = []
    wi = 0
    n_w = len(whisper_words)
    for line_idx, words in enumerate(lyric_lines):
        if not words:
            continue
        line_ws: List[Dict] = []
        for w in words:
            key = _norm_word(w)
            found = None
            while wi < n_w:
                gw = whisper_words[wi]
                if _norm_word(gw["word"]) == key:
                    found = gw
                    wi += 1
                    break
                wi += 1
            if found is not None:
                line_ws.append(found)
        for j, gw in enumerate(line_ws):
            end = line_ws[j + 1]["start"] if (j + 1) < len(line_ws) else float(gw["end"])
            end = max(end, float(gw["start"]) + 0.05)
            entries.append({
                "word": words[j] if j < len(words) else str(gw["word"]),
                "start": float(gw["start"]),
                "end": float(end),
                "line": line_idx,
            })
    return entries


# ---------------------------------------------------------------------------
# Effect selection UI
# ---------------------------------------------------------------------------
def pick_effect_preset(effects: Dict, config: Dict) -> Tuple[str, Optional[str], Dict]:
    presets = effects["presets"]
    random_pool = effects.get("random_pool", list(presets.keys()))
    default_key = config.get("default_effect_preset", list(presets.keys())[0])

    while True:
        clear_screen()
        print("=== Choose a Visual Style ===\n")
        print(f"1. DEFAULT  ({presets[default_key]['name']})")
        print(f"   {presets[default_key]['description']}")
        print("2. Random   (script picks a style for you)")
        print("3. Custom   (choose each piece yourself)")
        print("4. Saved    (use my last choice)")
        print("0. Back")
        choice = _safe_input("\nSelect: ").strip()

        if choice == "0":
            return None, None, {}
        elif choice == "1":
            cfg = json.loads(json.dumps(presets[default_key]))
            return default_key, default_key, cfg
        elif choice == "2":
            resolved = random.choice(random_pool)
            cfg = json.loads(json.dumps(presets[resolved]))
            cfg["_randomized"] = True
            return "random", resolved, cfg
        elif choice == "3":
            return pick_custom_effects(presets, effects)
        elif choice == "4":
            last = load_json(HISTORY_PATH, None)
            if last and last.get("entries"):
                last_entry = last["entries"][-1]
                prev_preset = last_entry.get("effect_preset")
                if prev_preset and prev_preset in presets:
                    cfg = json.loads(json.dumps(presets[prev_preset]))
                    return prev_preset, prev_preset, cfg
            print("\nNo saved style found. Using Default.")
            cfg = json.loads(json.dumps(presets[default_key]))
            return default_key, default_key, cfg
        else:
            print("Invalid choice.")
            pause()


def pick_custom_effects(presets: Dict, effects: Dict) -> Tuple[str, Optional[str], Dict]:
    keys = list(presets.keys())
    base = random.choice(keys)
    cfg = json.loads(json.dumps(presets[base]))
    cfg["_randomized"] = False

    while True:
        clear_screen()
        print("=== Custom Style Builder ===\n")
        print(f"Current: {json.dumps(cfg, indent=2)[:800]}...\n")
        print("1. Background theme")
        print("2. Blur type")
        print("3. Visualizer")
        print("4. Lyric style")
        print("5. Keep and continue")
        print("0. Cancel")
        choice = _safe_input("\nSelect: ").strip()

        if choice == "0":
            return None, None, {}
        elif choice == "5":
            return "custom", None, cfg
        elif choice == "1":
            print("\nBackgrounds:", ", ".join(effects["components"]["backgrounds"]))
            sel = _safe_input("Pick one (or Enter to keep): ").strip()
            if sel:
                cfg["background"]["theme"] = sel
        elif choice == "2":
            print("\nBlurs:", ", ".join(effects["components"]["blurs"]))
            sel = _safe_input("Pick one (or Enter to keep): ").strip()
            if sel:
                cfg["blur"]["type"] = sel
        elif choice == "3":
            print("\nVisualizers:", ", ".join(effects["components"]["visualizers"]))
            sel = _safe_input("Pick one (or Enter to keep): ").strip()
            if sel:
                cfg["visualizer"]["type"] = sel
        elif choice == "4":
            print("\nLyric styles:", ", ".join(effects["components"]["lyric_styles"]))
            sel = _safe_input("Pick one (or Enter to keep): ").strip()
            if sel:
                cfg["lyrics"]["style"] = sel
        else:
            print("Invalid choice.")


# ---------------------------------------------------------------------------
# Review / confirm
# ---------------------------------------------------------------------------
def show_summary(
    project_folder: Path,
    audio_path: Path,
    lyrics_path: Path,
    background_path: Optional[Path],
    effect_preset: str,
    effect_cfg: Dict,
    duration: Optional[float],
    word_count: int,
) -> bool:
    clear_screen()
    print("=== Review Your Choices ===\n")
    print(f"Project: {project_folder.name}")
    print(f"Music:   {audio_path.name}")
    print(f"Lyrics:  {lyrics_path.name} ({word_count} words)")
    print(f"Background: {background_path.name if background_path else '(none - will use default)'}")
    print(f"Style:   {effect_preset}")
    print(f"Lyrics:  {effect_cfg['lyrics']['style']} at {effect_cfg['lyrics']['position']}")
    print(f"Karaoke: Word-by-word highlighting (ASS)")
    print(f"Visuals: {effect_cfg['visualizer']['type']}")
    print(f"Blur:    {effect_cfg['blur']['type']}")
    if duration:
        print(f"Duration: {fmt_seconds(duration)}")
    print()
    print("Is this correct?")
    print("1. Yes, start rendering")
    print("2. No, go back")
    print("3. Cancel")
    choice = _safe_input("\nSelect: ").strip()
    return choice == "1"


def approve_execution() -> bool:
    clear_screen()
    print("=== Approve Execution ===\n")
    print("This will create a video file. It may take a few minutes.\n")
    print("1. Start rendering")
    print("2. Cancel")
    choice = _safe_input("\nSelect: ").strip()
    return choice == "1"


# ---------------------------------------------------------------------------
# Render pipeline (ffmpeg-based with ASS karaoke)
# ---------------------------------------------------------------------------
def next_version(output_dir: Path, slug: str) -> str:
    n = 1
    while True:
        candidate = output_dir / f"{slug} v{n}.mp4"
        if not candidate.exists():
            return f"{slug} v{n}.mp4"
        n += 1


def hex_to_ass_color(hex_color: str) -> str:
    """Convert #RRGGBB to ASS &HBBGGRR&"""
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = h[0:2], h[2:4], h[4:6]
        return f"&H00{b}{g}{r}&"
    return hex_color


# showspectrum filter color index (0-14 per FFmpeg source)
_HEX_TO_SPECTRUM_INDEX = {
    "#00ffff": 4,   # cyan
    "#ffffff": 0,   # white
    "#ff00ff": 6,   # magenta
    "#ff0000": 2,   # red
    "#00ff00": 3,   # green
    "#0000ff": 5,   # blue
    "#ffff00": 1,   # yellow
    "#ffa500": 8,   # orange (approx)
    "#800080": 7,   # purple (approx)
    "#000000": 7,   # black
}

_HEX_TO_COLOR_NAME = {
    "#000000": "black",
    "#ffffff": "white",
    "#00ffff": "cyan",
    "#ff00ff": "magenta",
    "#ff0000": "red",
    "#00ff00": "green",
    "#0000ff": "blue",
    "#ffff00": "yellow",
    "#ffa500": "orange",
    "#800080": "purple",
    "#ffc0cb": "pink",
    "#008080": "teal",
    "#808080": "gray",
}


def hex_to_spectrum_index(hex_color: str, fallback: int = 4) -> int:
    """Convert #RRGGBB to showspectrum color index (0-14)."""
    return _HEX_TO_SPECTRUM_INDEX.get(hex_color.lower(), fallback)


def hex_to_color_name(hex_color: str, fallback: str = "white") -> str:
    """Convert #RRGGBB to a FFmpeg color name for filters like showwaves."""
    return _HEX_TO_COLOR_NAME.get(hex_color.lower(), fallback)


def hex_to_ffmpeg_color(hex_color: str) -> str:
    """Convert #RRGGBB to 0xRRGGBB for FFmpeg filter options (color source, drawtext, etc.)."""
    h = hex_color.lstrip("#")
    if len(h) == 6:
        return f"0x{h}"
    return hex_color


def build_ffmpeg_command(
    config: Dict,
    effect_cfg: Dict,
    audio_path: Path,
    background_path: Optional[Path],
    defaults_dir: Path,
    ass_path: Path,
    output_path: Path,
    duration: float,
) -> List[str]:
    """
    Build ffmpeg command:
      - 1080x1920 vertical
      - Background (source or generated default or solid color)
      - Blur on background
      - Audio visualizer overlay
      - ASS karaoke subtitles burned in
      - H.264/AAC MP4
    """
    w = config["video_width"]
    h = config["video_height"]
    fps = config["fps"]
    ffmpeg = config["ffmpeg_path"]

    visualizer = effect_cfg["visualizer"]
    blur = effect_cfg["blur"]
    bg = effect_cfg["background"]

    inputs = ["-i", str(audio_path)]
    filter_parts = []

    # --- Background ---
    # [bg] is the base video stream.
    # Scale to cover the full 1080x1920 frame without stretching.
    # Top and bottom are always filled; sides may be cropped for wide images.
    bg_source_index = 1  # audio is input 0, background would be input 1
    found_bg = False

    if background_path and background_path.exists():
        is_bg_image = background_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
        if is_bg_image:
            inputs.extend(["-i", str(background_path)])
        else:
            # Loop the entire MP4 video indefinitely so it never ends before the audio
            inputs.extend(["-stream_loop", "-1", "-i", str(background_path)])
        found_bg = True
    else:
        default_bg = find_default_background(defaults_dir)
        if default_bg:
            is_bg_image = default_bg.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
            if is_bg_image:
                inputs.extend(["-i", str(default_bg)])
            else:
                inputs.extend(["-stream_loop", "-1", "-i", str(default_bg)])
            found_bg = True
            background_path = default_bg  # track for state/logging

    if found_bg:
        is_bg_image = background_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
        if is_bg_image:
            # Loop the single image frame indefinitely using the loop filter
            # Fit image inside frame without cropping/zooming; pad the rest black
            filter_parts.append(
                f"[{bg_source_index}:v]loop=loop=-1:size=1:start=0,"
                f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black[bg]"
            )
        else:
            # Fit video inside frame without cropping/zooming; pad the rest black
            filter_parts.append(
                f"[{bg_source_index}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black[bg]"
            )
    else:
        # No background at all: solid color
        color = hex_to_ffmpeg_color(bg.get("colors", ["#0a0a0a"])[0])
        filter_parts.append(
            f"color=c={color}:size={w}x{h}:duration={duration}:r={fps}[bg]"
        )

    # --- Audio Visualizer ---
    viz = visualizer.get("type", "none")
    viz_hex = visualizer.get("color", "#00ffff")
    viz_spectrum_idx = hex_to_spectrum_index(viz_hex)
    viz_color_name = hex_to_color_name(viz_hex)
    viz_color_ffmpeg = hex_to_ffmpeg_color(viz_hex)
    viz_height_pct = visualizer.get("height_pct", 15)
    viz_h = int(h * viz_height_pct / 100)
    # Position visualizer in the middle-lower area, above lyrics
    viz_y = int(h * 0.55)

    # [vid_label] tracks the current video stream label
    vid_label = "bg"

    if viz == "audio_spectrum":
        filter_parts.append(
            f"[0:a]showspectrum=s={w}x{viz_h}:mode=combined:color={viz_spectrum_idx}:scale=cbrt[spec]"
        )
        filter_parts.append(f"[{vid_label}][spec]overlay=0:{viz_y}[vid]")
        vid_label = "vid"
    elif viz == "audio_waveform":
        filter_parts.append(
            f"[0:a]showwaves=s={w}x{viz_h}:mode=line:colors={viz_color_name}[wave]"
        )
        filter_parts.append(f"[{vid_label}][wave]overlay=0:{viz_y}[vid]")
        vid_label = "vid"
    elif viz == "audio_light":
        light_h = viz_h
        filter_parts.append(
            f"color=c={viz_color_ffmpeg}:size={light_h}x{light_h}:duration={duration}:r={fps}[light]"
        )
        filter_parts.append(f"[{vid_label}][light]overlay={w//2 - light_h//2}:{viz_y}[vid]")
        vid_label = "vid"

    # --- ASS Karaoke Subtitles ---
    # ass filter takes the current video as input and outputs video with subtitles burned in as [final]
    try:
        ass_path_filter = ass_path.relative_to(ROOT).as_posix()
    except ValueError:
        ass_path_filter = ass_path.as_posix()
    # Escape characters that FFmpeg filter graph treats as special:
    #   - backslash first (so we don't double-escape)
    #   - single quotes (to safely wrap the path)
    #   - colons (drive letters / time expressions)
    #   - spaces (separate filter arguments)
    ass_path_filter = (
        ass_path_filter.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace(":", "\\:")
    )
    # Escape the resolved font directory the same way and pass fontsdir= so libass
    # can resolve the font family name from wherever the font lives.
    fontsdir_arg = ""
    karaoke_cfg = config.get("karaoke", {})
    resolved_font_dir = _find_font_dir(
        karaoke_cfg.get("font", "Chango"),
        karaoke_cfg.get("font_file"),
        defaults_dir,
    )
    if resolved_font_dir is not None:
        try:
            fonts_rel = resolved_font_dir.relative_to(ROOT).as_posix()
        except ValueError:
            fonts_rel = resolved_font_dir.as_posix()
        fonts_rel = fonts_rel.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
        fontsdir_arg = f":fontsdir='{fonts_rel}'"
    if not fontsdir_arg and defaults_dir and defaults_dir.exists():
        # Fall back to the defaults dir if font wasn't resolved elsewhere
        try:
            fonts_rel = defaults_dir.relative_to(ROOT).as_posix()
        except ValueError:
            fonts_rel = defaults_dir.as_posix()
        fonts_rel = fonts_rel.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
        fontsdir_arg = f":fontsdir='{fonts_rel}'"
    ass_filter = f"[{vid_label}]ass='{ass_path_filter}'{fontsdir_arg}[final]"
    filter_parts.append(ass_filter)

    cmd = [
        ffmpeg,
        "-y",
        *inputs,
        "-filter_complex", ",".join(filter_parts),
        "-map", "[final]",
        "-map", "0:a",
        "-c:v", config["codec"],
        "-crf", str(config["crf"]),
        "-preset", "fast",
        "-r", str(fps),
        "-c:a", config["audio_codec"],
        "-b:a", "192k",
        "-t", str(duration),
        str(output_path),
    ]
    return cmd


def run_render_with_progress(cmd: List[str], duration: float, cwd: Optional[Path] = None) -> Tuple[bool, float, str]:
    import time

    start = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=str(cwd) if cwd else None,
    )

    time_re = re.compile(r"time=(\d{2}):(\d{2}):(\d{2})\.(\d{2,3})")
    stderr_lines: List[str] = []

    try:
        while True:
            line = proc.stderr.readline()
            if not line and proc.poll() is not None:
                # drain remaining stderr
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
                pct = min(100, int((t / duration) * 100)) if duration else 0
                bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
                print(f"\r[{bar}] {pct}%  {fmt_seconds(t)} / {fmt_seconds(duration)}", end="", flush=True)
        proc.wait()
        print()
        stderr_text = "".join(stderr_lines)
        return proc.returncode == 0, time.time() - start, stderr_text
    except KeyboardInterrupt:
        proc.kill()
        proc.wait()
        print("\nCancelled by user.")
        return False, time.time() - start, ""


# ---------------------------------------------------------------------------
# Main interactive flow
# ---------------------------------------------------------------------------
def main() -> int:
    config = load_config()
    effects = load_effects()

    projects_dir = ROOT / config["projects_dir"]
    defaults_dir = ROOT / config["defaults_dir"]
    build_dir = ROOT / config["build_dir"]
    history_path = ROOT / config["history_file"]

    ensure_dir(projects_dir)
    ensure_dir(defaults_dir)
    ensure_dir(build_dir)

    while True:
        clear_screen()
        print("=== Shotcut TikTok Automator ===\n")
        print("This script turns one song + lyrics into a TikTok-style vertical video.\n")

        # --- Step 1: Pick project ---
        projects = find_projects(projects_dir)
        if not projects:
            print("No project folders found.")
            print(f"Please add folders to: {projects_dir}")
            print("Each folder needs: an MP3 and a TXT lyrics file.")
            pause()
            return 1

        print("Available projects:\n")
        for i, p in enumerate(projects, 1):
            audio = find_audio(p)
            lyrics = find_lyrics(p)
            wc = 0
            if lyrics:
                try:
                    wc = len(lyrics.read_text(encoding="utf-8").split())
                except Exception:
                    wc = 0
            print(f"{i}. {p.name}")
            print(f"   Music: {audio.name if audio else 'MISSING'}")
            print(f"   Lyrics: {lyrics.name if lyrics else 'MISSING'} ({wc} words)")
        print("0. Exit")

        choice = _safe_input("\nSelect a project: ").strip()
        if choice == "0":
            return 0
        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(projects):
                raise ValueError
        except ValueError:
            print("Invalid choice.")
            pause()
            continue

        project_folder = projects[idx]
        slug = project_folder.name
        audio_path = find_audio(project_folder)
        lyrics_path = find_lyrics(project_folder)
        background_path = find_background(project_folder)

        # Pre-flight checklist
        clear_screen()
        print("=== Pre-Flight Checklist ===\n")
        ok = True
        if not audio_path:
            print("✗ Missing music file (MP3/WAV/M4A)")
            ok = False
        else:
            print(f"✓ Music: {audio_path.name}")
        if not lyrics_path:
            print("✗ Missing lyrics file (TXT/LRC/SRT)")
            ok = False
        else:
            print(f"✓ Lyrics: {lyrics_path.name}")
        if not background_path:
            dbg = find_default_background(defaults_dir)
            if dbg:
                print(f"✓ Background: (none in project) → using default {dbg.name}")
                background_path = dbg
            else:
                print("⚠ No background file (will use solid color)")
        else:
            print(f"✓ Background: {background_path.name}")

        ffmpeg_path = config["ffmpeg_path"]
        try:
            subprocess.run([ffmpeg_path, "-version"], capture_output=True, timeout=10)
            print(f"✓ ffmpeg found: {ffmpeg_path}")
        except Exception:
            print(f"⚠ ffmpeg not found ('{ffmpeg_path}'). Install it to render real videos.")
            ok = False

        if not ok:
            print("\nFix the items above, then run this script again.")
            pause()
            continue

        # Probe audio duration
        duration = probe_duration(ffmpeg_path, audio_path)
        if duration:
            print(f"✓ Duration: {fmt_seconds(duration)}")

        # Hybrid vocal-region detection (local, adaptive) — the engine that keeps
        # captions from showing during instrumental sections
        print("🎵 Analyzing vocal regions (local detection)...")
        vocal_regions = detect_vocal_regions(ffmpeg_path, audio_path)
        if vocal_regions:
            total_vocal = sum(e - s for s, e in vocal_regions)
            print(f"✓ Detected {len(vocal_regions)} vocal regions ({total_vocal:.1f}s of singing)")
            print("  Captions will only appear while vocals are active.")
        else:
            print("⚠ Could not detect distinct vocal regions — using equal distribution.")
            vocal_regions = None

        # Auto-detect vocal onset from the audio file (only mode)
        print("🔍 Auto-detecting vocal onset from audio...")
        lyric_offset = detect_vocal_onset(ffmpeg_path, audio_path)
        if vocal_regions:
            print(f"✓ Onset ignored — first vocal region starts at {vocal_regions[0][0]:.2f}s")
        elif lyric_offset <= 0.0:
            print("⚠ Warning: Vocal onset could not be auto-detected. Using 0.0s offset.")
            print("  Words may appear during instrumental intro.")
            print("  This typically happens with songs that start with loud instruments.")
            print("  Check your MP3 or provide a silent intro segment.")
        else:
            print(f"✓ Auto-detected vocal onset at: {lyric_offset:.2f}s")

        # --- Whisper forced alignment (preferred timing engine when available) ---
        word_entries = None
        whisper_words = None
        try:
            whisper_words = transcribe_words(audio_path, language="es")
        except Exception:
            whisper_words = None

        if whisper_words:
            try:
                raw_lyr = read_lyrics_raw(lyrics_path)
                lyr_lines = [ln.rstrip("\r").split() for ln in raw_lyr.split("\n") if ln.strip()]
                word_entries = assign_whisper_words_to_lines(whisper_words, lyr_lines)
            except Exception:
                word_entries = None

        if word_entries:
            print(f"✓ Words aligned via Whisper forced alignment ({len(whisper_words)} words detected)")
        else:
            print("⚠ Whisper unavailable/failed — using local hybrid timing")
            word_entries = parse_lyrics_with_words(lyrics_path, duration or 60.0, lyric_offset, vocal_regions)

        word_count = len(word_entries)
        if not whisper_words and vocal_regions:
            print(f"✓ Words detected: {word_count} (karaoke-ready, timed to {len(vocal_regions)} vocal regions)")
        elif not whisper_words:
            print(f"✓ Words detected: {word_count} (karaoke-ready, offset={lyric_offset:.2f}s)")

        pause("\nPress Enter to continue...")

        # --- Step 2: Pick style ---
        effect_preset, resolved_preset, effect_cfg = pick_effect_preset(effects, config)
        if effect_preset is None:
            continue

        # --- Step 3: Review ---
        if not show_summary(project_folder, audio_path, lyrics_path, background_path, effect_preset, effect_cfg, duration, word_count):
            continue

        # --- Step 4: Approve ---
        if not approve_execution():
            continue

        # --- Step 5: Render ---
        clear_screen()
        print("=== Rendering ===\n")

        iteration_build = build_dir / slug
        ensure_dir(iteration_build)

        comp = load_project_completion(project_folder)
        iteration = comp.get("total_iterations", 0) + 1

        state_path = init_state(iteration_build, slug, stage="syncing")
        update_state(state_path, stage="syncing")

        # Generate ASS karaoke file — saved in project folder for easy editing
        ass_path = (project_folder / ".data" / "karaoke.ass")
        ensure_dir(ass_path.parent)
        write_ass_karaoke(word_entries, ass_path, config, effect_cfg, config["video_width"], config["video_height"], defaults_dir)
        # Save metadata for reference
        save_json_atomic(project_folder / ".data" / "karaoke_meta.json", {
            "schema": "shotcut_tiktok.karaoke_meta.v1",
            "lyrics_checksum": sha256_file(lyrics_path),
            "generated_at": now_iso(),
            "word_count": len(word_entries),
            "vocal_regions_detected": bool(vocal_regions),
            "vocal_regions": [[round(s, 2), round(e, 2)] for s, e in vocal_regions] if vocal_regions else [],
            "total_vocal_seconds": round(sum(e - s for s, e in vocal_regions), 2) if vocal_regions else 0.0,
        })
        update_state(state_path, stage="syncing", artifacts={"ass": str(ass_path)})

        # Write effect config
        effect_cfg["slug"] = slug
        effect_cfg["iteration"] = iteration
        effect_cfg["audio_file"] = str(audio_path)
        effect_cfg["lyrics_file"] = str(lyrics_path)
        effect_cfg["background_file"] = str(background_path) if background_path else None
        effect_cfg["ass_file"] = str(ass_path)
        effect_cfg["word_count"] = word_count
        effect_config_path = iteration_build / f"iteration_{iteration:02d}" / "effect_config.json"
        ensure_dir(effect_config_path.parent)
        save_json_atomic(effect_config_path, effect_cfg)

        # Output filename (versioned) — saved directly in project folder
        out_name = next_version(project_folder, slug)
        output_path = project_folder / out_name

        update_state(state_path, stage="rendering", artifacts={
            "ass": str(ass_path),
            "effect_config": str(effect_config_path),
            "output": str(output_path),
        })

        # Build & run ffmpeg command
        cmd = build_ffmpeg_command(
            config=config,
            effect_cfg=effect_cfg,
            audio_path=audio_path,
            background_path=background_path,
            defaults_dir=defaults_dir,
            ass_path=ass_path,
            output_path=output_path,
            duration=duration or 60.0,
        )

        success, render_seconds, stderr_text = run_render_with_progress(cmd, duration or 60.0, cwd=ROOT)

        if success:
            update_state(state_path, stage="completed")

            checksums = {
                audio_path.name: sha256_file(audio_path),
                lyrics_path.name: sha256_file(lyrics_path),
            }
            if background_path:
                checksums[background_path.name] = sha256_file(background_path)

            record_iteration(
                project_folder=project_folder,
                iteration=iteration,
                effect_preset=effect_preset,
                resolved_preset=resolved_preset,
                effect_config_file=str(effect_config_path.relative_to(ROOT)),
                output_file=str(output_path.relative_to(ROOT)),
                render_seconds=render_seconds,
                duration_seconds=duration or 0.0,
                resolution=f"{config['video_width']}x{config['video_height']}",
                fps=config["fps"],
                source_checksums=checksums,
                notes=f"Karaoke render ({word_count} words)",
            )

            append_history(history_path, {
                "entry_id": str(uuid.uuid4())[:12],
                "slug": slug,
                "project_folder": str(project_folder.relative_to(ROOT)),
                "iteration": iteration,
                "effect_preset": effect_preset,
                "resolved_preset": resolved_preset,
                "effect_config_file": str(effect_config_path.relative_to(ROOT)),
                "output_file": str(output_path.relative_to(ROOT)),
                "output_exists": output_path.exists(),
                "render_seconds": round(render_seconds, 2),
                "duration_seconds": round(duration or 0.0, 2),
                "resolution": f"{config['video_width']}x{config['video_height']}",
                "fps": config["fps"],
                "completed_at": now_iso(),
                "source_files": checksums,
                "notes": f"Iteration {iteration} - karaoke ({word_count} words)",
            })

            print(f"\n✓ Render complete!")
            print(f"  Output: {output_path}")
            print(f"  Time: {fmt_seconds(render_seconds)}")

            # Notification sound — render complete
            try:
                winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
            except Exception:
                try:
                    winsound.Beep(1000, 400)
                except Exception:
                    pass
        else:
            update_state(state_path, stage="failed", error="ffmpeg returned non-zero exit code")
            print("\n✗ Render failed.")
            if stderr_text:
                print("---- ffmpeg error output ----")
                print(stderr_text[-3000:])
                print("-----------------------------")
            pause()
            continue

        # --- Step 6: What's next? ---
        while True:
            clear_screen()
            print("=== Done! ===\n")
            print(f"Video saved: {output_path}\n")
            print("What would you like to do?")
            print("1. Work on another project")
            print(f"2. Render '{slug}' again with a different look")
            print("3. Exit")
            choice = _safe_input("\nSelect: ").strip()
            if choice == "1":
                break
            elif choice == "2":
                continue_inner = True
                while continue_inner:
                    effect_preset2, resolved_preset2, effect_cfg2 = pick_effect_preset(effects, config)
                    if effect_preset2 is None:
                        break
                    clear_screen()
                    print("=== Rendering (Iteration) ===\n")
                    iteration += 1
                    state_path = init_state(iteration_build, slug, stage="rendering")
                    # Reuse existing karaoke.ass from .data folder (lyrics unchanged)
                    ass_path = project_folder / ".data" / "karaoke.ass"
                    if not ass_path.exists() and (project_folder / "karaoke.ass").exists():
                        ass_path = project_folder / "karaoke.ass"
                    effect_cfg2["slug"] = slug
                    effect_cfg2["iteration"] = iteration
                    effect_cfg2["audio_file"] = str(audio_path)
                    effect_cfg2["lyrics_file"] = str(lyrics_path)
                    effect_cfg2["background_file"] = str(background_path) if background_path else None
                    effect_cfg2["ass_file"] = str(ass_path)
                    effect_cfg2["word_count"] = word_count
                    effect_config_path = iteration_build / f"iteration_{iteration:02d}" / "effect_config.json"
                    ensure_dir(effect_config_path.parent)
                    save_json_atomic(effect_config_path, effect_cfg2)
                    # Output filename (versioned) — saved directly in project folder
                    out_name = next_version(project_folder, slug)
                    output_path = project_folder / out_name
                    cmd = build_ffmpeg_command(
                        config=config,
                        effect_cfg=effect_cfg2,
                        audio_path=audio_path,
                        background_path=background_path,
                        defaults_dir=defaults_dir,
                        ass_path=ass_path,
                        output_path=output_path,
                        duration=duration or 60.0,
                    )
                    success, render_seconds, stderr_text = run_render_with_progress(cmd, duration or 60.0, cwd=ROOT)
                    if success:
                        checksums = {
                            audio_path.name: sha256_file(audio_path),
                            lyrics_path.name: sha256_file(lyrics_path),
                        }
                        if background_path:
                            checksums[background_path.name] = sha256_file(background_path)
                        record_iteration(
                            project_folder=project_folder,
                            iteration=iteration,
                            effect_preset=effect_preset2,
                            resolved_preset=resolved_preset2,
                            effect_config_file=str(effect_config_path.relative_to(ROOT)),
                            output_file=str(output_path.relative_to(ROOT)),
                            render_seconds=render_seconds,
                            duration_seconds=duration or 0.0,
                            resolution=f"{config['video_width']}x{config['video_height']}",
                            fps=config["fps"],
                            source_checksums=checksums,
                        )
                        append_history(history_path, {
                            "entry_id": str(uuid.uuid4())[:12],
                            "slug": slug,
                            "project_folder": str(project_folder.relative_to(ROOT)),
                            "iteration": iteration,
                            "effect_preset": effect_preset2,
                            "resolved_preset": resolved_preset2,
                            "effect_config_file": str(effect_config_path.relative_to(ROOT)),
                            "output_file": str(output_path.relative_to(ROOT)),
                            "output_exists": output_path.exists(),
                            "render_seconds": round(render_seconds, 2),
                            "duration_seconds": round(duration or 0.0, 2),
                            "resolution": f"{config['video_width']}x{config['video_height']}",
                            "fps": config["fps"],
                            "completed_at": now_iso(),
                            "source_files": checksums,
                            "notes": f"Iteration {iteration} - karaoke ({word_count} words)",
                        })
                        print(f"\n✓ Render complete: {output_path}")
                    else:
                        print("\n✗ Render failed.")
                        if stderr_text:
                            print("---- ffmpeg error output ----")
                            print(stderr_text[-3000:])
                            print("-----------------------------")
                    pause("\nPress Enter to continue...")
                    continue_inner = False
                break
            elif choice == "3":
                return 0
            else:
                print("Invalid choice.")


def _safe_input(prompt: str) -> str:
    """Wrapper around input() that handles non-terminal environments gracefully."""
    try:
        return input(prompt)
    except EOFError:
        print("(no input available, exiting)")
        sys.exit(0)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nExiting.")
        sys.exit(0)
    except Exception as e:
        print("\n" + "=" * 60)
        print("ERROR: The script crashed.")
        print("=" * 60)
        print(str(e))
        print("=" * 60)
        import traceback
        traceback.print_exc()
        print("=" * 60)
        try:
            _safe_input("\nPress Enter to close this window...")
        except EOFError:
            pass
        sys.exit(1)
