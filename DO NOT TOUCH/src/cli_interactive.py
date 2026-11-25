#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShotCut TikTok Automator — Interactive CLI Entry Point (TikTok Wizard v5.4)

Interactive CLI for producing TikTok / YouTube Shorts style karaoke videos
with word-by-word highlighting timed via the Greenfield MMS_FA Engine,
studio-grade atmospheric particle effects, and permanent settings manifests.

Strict Filesystem Boundary Invariant:
- `START HERE/` is treated as strictly READ-ONLY for reading source audio,
  source lyrics, and `_Assets/`.
- All generated machine files (`karaoke.ass`, `alignment_manifest.json`,
  `karaoke_meta.json`, `lyrics_synced.txt`, stems, and completion markers)
  are stored exclusively inside `DO NOT TOUCH/data/<project_slug>/`.
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

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from src.align.pipeline import UniversalEngine
from src.core.hardware import (
    ComputeProfile,
    HardwareReport,
    build_compute_profile,
    detect_hardware,
    install_gpu_packages_for_vendor,
    load_compute_profile,
    resolve_active_profile,
    save_compute_profile,
)
from src.pipeline.ass_writer import write_ass_karaoke
from src.pipeline.lyrics_editor import (
    export_easy_lyrics_file,
    import_easy_lyrics_to_entries,
)
from src.pipeline.render import build_ffmpeg_command, find_default_background
from src.pipeline.manifest_writer import write_video_render_manifest
from src.pipeline.schema import KaraokeEntry, KaraokeWord
from src.narration import (
    NarrationError,
    check_narration_backend,
    check_narration_backends,
    detect_persian_script,
    ensure_narration_dirs,
    generate_narration,
    generate_narration_subprocess,
    get_narration_paths,
    list_narration_jobs,
    list_scripts,
    list_voice_samples,
)


# ---------------------------------------------------------------------------
# Responsive Terminal Formatting Helpers
# ---------------------------------------------------------------------------


def get_term_width() -> int:
    try:
        cols = shutil.get_terminal_size((80, 24)).columns
        return max(40, cols)
    except Exception:
        return 80


def print_wrapped(
    text: str, indent: str = "", subsequent_indent: Optional[str] = None
) -> None:
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
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
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
    val = val.strip()
    if not val:
        return None
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
    try:
        fp = Path(file_path).resolve()
        if sys.platform == "win32":
            if fp.is_dir():
                os.startfile(str(fp))
            elif fp.is_file():
                subprocess.Popen(f'explorer.exe /select,"{str(fp)}"')
            else:
                parent_dir = fp.parent if fp.parent.exists() else Path.cwd()
                os.startfile(str(parent_dir))
        elif sys.platform == "darwin":
            if fp.is_file():
                subprocess.Popen(["open", "-R", str(fp)])
            else:
                subprocess.Popen(["open", str(fp if fp.is_dir() else fp.parent)])
        else:
            subprocess.Popen(["xdg-open", str(fp if fp.is_dir() else fp.parent)])
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
    return val.strip().lower() in ("b", "back", "cancel", "0")


# ---------------------------------------------------------------------------
# Configuration & Defaults
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "schema": "shotcut_tiktok.config.v4_1",
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
    "lyric_sync_engine": "greenfield_v4_1_real_mms_fa",
    "whisper_model": "small",
    "whisper_compute_type": "int8",
    "tau_song": 0.23,
    "word_target": 0.95,
    "ffmpeg_path": "ffmpeg",
    "melt_path": "melt",
    "safe_mode": True,
    "karaoke": {
        "enabled": True,
        "highlights_enabled": True,
        "highlight_color": "&H00B400FF&",
        "base_color": "&H00D900FF&",
        "outline_color": "&H00FFFFFF&",
        "font": "Chango",
        "font_size": 72,
        "outline_width": 4,
        "shadow": True,
        "shadow_depth": 5,
        "position_y_percent": 72.0,
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


def _safe_relocate_file(source_path: Path, label: str = "relocated") -> None:
    """
    Safely relocate a file or temporary artifact to a dedicated scratch subfolder
    with a mandatory manifest and restore script, adhering strictly to the
    File Retention & Safe-Relocation Policy. Never unlinks or deletes.
    """
    if not source_path.exists():
        return
    import tempfile

    now = datetime.now()
    ts_folder = now.strftime("%Y%m%d_%H%M%S")
    ts_iso = now.isoformat()

    scratch_base = Path("X:/scratch")
    if not scratch_base.exists():
        scratch_base = Path(tempfile.gettempdir()) / "tiktokwizard_scratch"
        scratch_base.mkdir(parents=True, exist_ok=True)

    dest_dir = scratch_base / f"{label}_{ts_folder}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        size = source_path.stat().st_size
        h = hashlib.sha256()
        with open(source_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        sha = h.hexdigest()
    except Exception:
        size = 0
        sha = "unknown"

    dest_file = dest_dir / source_path.name
    try:
        shutil.move(str(source_path), str(dest_file))
    except Exception:
        return

    readme = dest_dir / "README.md"
    readme_text = (
        f"# Relocation Manifest\n"
        f"- **Original Source**: `{source_path.resolve()}`\n"
        f"- **Timestamp**: {ts_iso}\n"
        f"- **Purpose**: Safe relocation of {label}\n"
        f"- **Size**: {size} bytes\n"
        f"- **SHA-256**: `{sha}`\n\n"
        f"## Reinstatement\n"
        f"Run `restore.ps1` or:\n"
        f"```powershell\n"
        f'Copy-Item "{dest_file.resolve()}" "{source_path.resolve()}"\n'
        f"```\n"
    )
    try:
        readme.write_text(readme_text, encoding="utf-8")
    except Exception:
        pass

    restore_ps1 = dest_dir / "restore.ps1"
    try:
        restore_ps1.write_text(
            f'Copy-Item "{dest_file.resolve()}" "{source_path.resolve()}" -Force\n',
            encoding="utf-8",
        )
    except Exception:
        pass

    manifest_log = scratch_base / "MANIFEST.jsonl"
    try:
        entry = json.dumps(
            {
                "timestamp": ts_iso,
                "folder": str(dest_dir.resolve()),
                "source": str(source_path.resolve()),
                "reason": label,
                "files_count": 1,
            }
        )
        with open(manifest_log, "a", encoding="utf-8") as ml:
            ml.write(entry + "\n")
    except Exception:
        pass


def init_default_profiles() -> None:
    ensure_dir(PROFILES_DIR)
    default_p = PROFILES_DIR / "default.json"
    if not default_p.exists():
        save_json_atomic(
            default_p,
            {
                "name": "DEFAULT (default.mp4 + Neon Pink Bubble Lyrics)",
                "description": "Signature style: default background video with bold Chango bubble-letter lyrics, vibrant neon pink font/highlight, crisp white outline, and deep black offset drop-shadow.",
                "is_default": True,
                "background": {"type": "video", "fit_mode": "blur_pad", "file": None},
                "visualizer": {
                    "type": "spectrum_bars",
                    "color": "#00ffff",
                    "color2": "#ff007f",
                    "height_pct": 14,
                    "position": "above_pad",
                },
                "particles": {
                    "type": "bokeh_dust",
                    "density": "medium",
                    "speed": "normal",
                    "audio_reactive": "pulse",
                    "color": "#ffd700",
                },
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
                    "highlights_enabled": True,
                },
            },
        )


def list_profiles() -> Dict[str, dict]:
    init_default_profiles()
    profiles: Dict[str, dict] = {}
    for p in sorted(PROFILES_DIR.glob("*.json")):
        data = load_json(p)
        if data and isinstance(data, dict):
            profiles[p.stem] = data

    sorted_profiles: Dict[str, dict] = {}
    if "default" in profiles:
        sorted_profiles["default"] = profiles.pop("default")
    for k, v in profiles.items():
        sorted_profiles[k] = v
    return sorted_profiles


def save_profile(key: str, data: dict) -> None:
    ensure_dir(PROFILES_DIR)
    save_json_atomic(PROFILES_DIR / f"{key}.json", data)


def delete_profile(key: str) -> bool:
    if key.lower() == "default":
        return False
    path = PROFILES_DIR / f"{key}.json"
    if path.exists():
        try:
            _safe_relocate_file(path, f"deleted_profile_{key}")
            return True
        except Exception:
            return False
    return False


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
_BANNED_LYRIC_NAMES = {"lyrics_easy.txt", "lyrics_synced.txt", "readme.txt"}


def find_projects(projects_dir: Path) -> List[Path]:
    if not projects_dir.is_dir():
        return []
    candidates = []
    for p in sorted(projects_dir.iterdir()):
        if not p.is_dir():
            continue
        norm_name = p.name.strip().lower()
        if norm_name in ("_assets", "assets", "defaults", "_defaults"):
            continue
        if norm_name.startswith(("_artist_song", "artist_song", ".")):
            continue
        candidates.append(p)
    return candidates


def find_audio(folder: Path) -> Optional[Path]:
    if not folder.is_dir():
        return None
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
            return f
    return None


def find_lyrics(folder: Path) -> Optional[Path]:
    """Find the user's raw Holy-Grail lyric file (never selecting lyrics_easy.txt or lyrics_synced.txt)."""
    if not folder.is_dir():
        return None
    candidates = []
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix.lower() in LYRIC_EXTS:
            low = f.name.lower()
            if low in _BANNED_LYRIC_NAMES or "readme" in low:
                continue
            candidates.append(f)
    return candidates[0] if candidates else None


def get_project_data_dir(project_folder: Path) -> Path:
    """Return `DO NOT TOUCH/data/<project_slug>` without ever touching `START HERE/`."""
    dnt_data = _project_root / "data" / project_folder.name
    dnt_data.mkdir(parents=True, exist_ok=True)
    return dnt_data


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


def load_project_completion(project_folder: Path) -> dict:
    dot_data = get_project_data_dir(project_folder)
    marker = dot_data / ".completed.json"
    data = load_json(marker, None)
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
    comp["iterations"].append(
        {
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
        }
    )
    dot_data = get_project_data_dir(project_folder)
    save_json_atomic(dot_data / ".completed.json", comp)


def run_render_with_progress(
    cmd: List[str], duration: float
) -> Tuple[bool, float, str]:
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
            assert proc.stderr is not None
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
                cols = get_term_width()
                avail_bar_len = max(10, min(40, cols - 32))
                filled = int(avail_bar_len * (pct / 100.0))
                bar = "=" * filled + "-" * (avail_bar_len - filled)
                status_str = (
                    f"[{bar}] {pct:3d}%  {fmt_seconds(t)} / {fmt_seconds(duration)}"
                )
                print(f"\r{status_str[: cols - 1]}", end="", flush=True)

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


def select_profile_menu(current_profile_key: str, profiles: Dict[str, dict]) -> str:
    while True:
        clear_screen()
        print_header("Select Visual Profile Preset")
        print_wrapped(
            "Choose a curated visual style preset for subtitles, colors, and layout:\n"
        )

        keys = list(profiles.keys())
        for idx, k in enumerate(keys, 1):
            p = profiles[k]
            name = p.get("name", k)
            marker = " (CURRENT)" if k == current_profile_key else ""
            desc = p.get("description", "")
            print(f"  {idx}. {k.upper()}{marker}")
            if desc:
                print_wrapped(f"     {desc}", indent="     ")
            print()

        print("  0. Keep Current Profile and Return\n")
        choice = _safe_input(f"Select [1-{len(keys)}, 0 to return]: ").strip()
        if choice in ("0", "b", "back", ""):
            return current_profile_key

        try:
            sel_idx = int(choice) - 1
            if 0 <= sel_idx < len(keys):
                return keys[sel_idx]
        except ValueError:
            pass
        print("Invalid selection.")
        time.sleep(1)


def color_picker_menu(current_color: str, color_role: str) -> str:
    """Friendly color picker submenu converting between human names, RGB HEX, and ASS color format."""
    palette = [
        ("Electric Yellow", "#FFE600", "&H0000E6FF&"),
        ("Neon Cyan", "#00FFFF", "&H00FFFF00&"),
        ("Hot Pink / Magenta", "#FF007F", "&H007F00FF&"),
        ("Electric Violet", "#B400FF", "&H00FF00B4&"),
        ("Vibrant Red", "#FF2200", "&H000022FF&"),
        ("Acid Green", "#00FF66", "&H0066FF00&"),
        ("Pure Gold", "#FFD700", "&H0000D7FF&"),
        ("Crisp White", "#FFFFFF", "&H00FFFFFF&"),
        ("Jet Black", "#000000", "&H00000000&"),
    ]

    while True:
        clear_screen()
        print_header(f"Select Color: {color_role}")
        print(f"Current Value: {current_color}\n")
        print("Choose from curated palette presets:")
        for idx, (c_name, c_hex, c_ass) in enumerate(palette, 1):
            marker = " (ACTIVE)" if current_color in (c_ass, c_hex) else ""
            print(f"  {idx}. {c_name:<20} {c_hex}  [{c_ass}]{marker}")

        print(f"  {len(palette) + 1}. Custom RGB Hex (e.g. #FF00AA or #00FFCC)")
        print("\n  0. Keep Current and Return\n")

        choice = _safe_input(f"Select [1-{len(palette) + 1}, 0 to cancel]: ").strip()
        if choice in ("0", "b", "back", ""):
            return current_color

        try:
            sel_idx = int(choice) - 1
            if 0 <= sel_idx < len(palette):
                _, _, ass_val = palette[sel_idx]
                return ass_val
            elif sel_idx == len(palette):
                hex_input = (
                    _safe_input("Enter 6-digit RGB Hex (e.g. #FF1493 or FF1493): ")
                    .strip()
                    .lstrip("#")
                )
                if len(hex_input) == 6:
                    r, g, b = hex_input[0:2], hex_input[2:4], hex_input[4:6]
                    ass_val = f"&H00{b}{g}{r}&".upper()
                    return ass_val
                else:
                    print("Invalid Hex format. Please provide 6 hex characters.")
                    time.sleep(1.2)
        except ValueError:
            pass
        print("Invalid selection.")
        time.sleep(1)


def customize_visuals_menu(selected_fx: dict, defaults_dir: Path) -> dict:
    fx = json.loads(json.dumps(selected_fx))
    lyrics_cfg = fx.setdefault("lyrics", {})
    pad_cfg = fx.setdefault("tiktok_pad", {})
    handle_cfg = fx.setdefault("creator_handle", {})
    viz_cfg = fx.setdefault("visualizer", {})
    particle_cfg = fx.setdefault("particles", {})

    fonts = [
        "Chango",
        "Montserrat",
        "Arial",
        "Arial Narrow",
        "Impact",
        "Segoe UI",
        "Trebuchet MS",
        "Fraunces",
    ]
    karaoke_styles = [
        ("word_highlight", "Word Highlight (Standard 1:1 karaoke wipe)"),
        ("word_pop", "Word Pop (+30% dynamic size pop on active word)"),
        ("underline", "Underline Accent (Dynamic underline on active word)"),
        ("clean_static", "Clean Static (Clear subtitle line without wipe)"),
    ]
    visualizer_types = [
        ("spectrum_bars", "1. Modern Multi-Band Spectrum Equalizer Bars [DEFAULT]"),
        ("radial_bass", "2. Modern Radial Bass Pulse (Circle kick on bass)"),
        ("stereo_mirror", "3. Modern Mirrored Stereo Equalizer Bars"),
        ("particle_dots", "4. Modern Floating Particle Dots (Transient reactive)"),
        ("spectrum_wave", "5. Modern Neon Spectrum Ribbon Wave"),
        ("radial_ring", "6. Modern Neon Radial Ring Wave"),
        ("volt_radial", "7. Volt Radial Flower (volt-l18 256-band blooming rays)"),
        ("segmented_bars", "8. Discrete Segmented LED Equalizer Bars"),
        ("peak_dots", "9. Spectrum Bars with Floating Peak Transient Dots"),
        ("dual_wave", "10. Dual Sinusoidal Phase Soundwave Lines"),
        ("radial_rays", "11. High-Density Radial Laser Ray Burst"),
        ("polygon_mesh", "12. Geometric Cyber Lattice Polygon Mesh"),
        ("crosshair_bars", "13. Symmetrical Center-Out Crosshair Equalizer"),
        ("center_flower", "14. Blooming Center Harmonic Petal Flower"),
        ("starburst", "15. Dynamic Frequency Starburst Galaxy"),
        ("sound_pillows", "16. Floating Rounded Sound Capsule Pillows"),
        ("curved_ribbon", "17. Triple Curved Neon Audio Ribbon"),
        ("audio_waveform", "18. Minimal Hardware Audio Waveform"),
        ("stereo_wave", "19. Dual Peak Stereo Oscilloscope Waves"),
        ("none", "20. None / Clean Background (Disabled)"),
    ]
    visualizer_positions = [
        (
            "above_pad",
            "Above Pad (Bottom of visualizer rests flush on top of TikTok pad) [DEFAULT]",
        ),
        ("below_lyrics", "Below Lyrics (Directly under active lyric lines)"),
        ("center", "Center Screen"),
        ("top", "Top Screen"),
        ("bottom", "Absolute Bottom"),
    ]
    part_type_names = {
        "bokeh_dust": "Bokeh Dust (Luminous golden floating orbs)",
        "rising_embers": "Rising Embers (Fiery upward cinders & sparks)",
        "cinematic_snow": "Cinematic Snow (Parallax multi-layer snowfall)",
        "radial_burst": "Radial Burst (Audio-reactive spark shockwaves)",
        "neon_stars": "Neon Stars (Twinkling diamond cross stardust)",
        "none": "None / Clean Background (Disabled)",
    }

    while True:
        clear_screen()
        print_header("Customize Visuals, Typography & Overlays")
        font_name = lyrics_cfg.get("font", "Chango")
        font_size = lyrics_cfg.get("font_size", 72)
        pos_y = lyrics_cfg.get("position_y_percent", 65)
        h_color = lyrics_cfg.get("highlight_color", "&H00B400FF&")
        b_color = lyrics_cfg.get("base_color", "&H00D900FF&")
        o_color = lyrics_cfg.get("outline_color", "&H00FFFFFF&")
        o_width = lyrics_cfg.get("outline_width", 4)
        shadow_on = lyrics_cfg.get("shadow", True)
        shadow_depth = lyrics_cfg.get("shadow_depth", 5)
        curr_style = lyrics_cfg.get("karaoke_style", "word_highlight")
        pad_enabled = pad_cfg.get("enabled", True)
        pad_opacity = float(pad_cfg.get("opacity", 0.90))
        pad_h_pct = float(pad_cfg.get("height_pct", 12.5))
        h_text = handle_cfg.get("text", "")
        h_pos = handle_cfg.get("position", "bottom_right")
        h_font = handle_cfg.get("font", "Montserrat")
        h_size = handle_cfg.get("font_size", 28)
        h_color = handle_cfg.get("color", "&H00FFFFFF&")
        h_outline = handle_cfg.get("outline_color", "&H00000000&")
        h_out_w = handle_cfg.get("outline_width", 2)
        h_op = handle_cfg.get("opacity", 0.85)

        curr_viz_type = viz_cfg.get("type", "none")
        curr_viz_pos = viz_cfg.get("position", "above_pad")
        curr_viz_color = viz_cfg.get("color", "#00ffff")
        curr_viz_h_pct = float(viz_cfg.get("height_pct", 12))

        curr_part_type = particle_cfg.get("type", "none")
        curr_part_density = particle_cfg.get("density", "medium")
        curr_part_speed = particle_cfg.get("speed", "normal")
        curr_part_audio = particle_cfg.get("audio_reactive", "pulse")
        curr_part_color = particle_cfg.get("color", "auto")
        part_label = part_type_names.get(
            curr_part_type, curr_part_type.replace("_", " ").title()
        )

        print("--- Typography & Positioning ---")
        print(f"  1. Font Family:            {font_name}")
        print(f"  2. Font Size:              {font_size} pt")
        print(f"  3. Lyrics Y-Position:      {pos_y}% from top")
        print(f"  4. Karaoke Style:          {curr_style}")
        print("\n--- Palette & Styling ---")
        print(f"  5. Highlight Color:        {h_color}")
        print(f"  6. Base Text Color:        {b_color}")
        print(f"  7. Outline Color & Width:  {o_color} ({o_width}px)")
        print(
            f"  8. Drop Shadow:            {'Enabled' if shadow_on else 'Disabled'} (depth: {shadow_depth}px)"
        )
        print("\n--- Audio Visualizer ---")
        print(f"  9. Visualizer Type:        {curr_viz_type}")
        print(
            f" 10. Visualizer Placement:   {curr_viz_pos} (Height: {int(curr_viz_h_pct)}%, Color: {curr_viz_color})"
        )
        print("\n--- TikTok Elements & Handles ---")
        print(
            f" 11. TikTok Safety Dark Pad: {'Enabled' if pad_enabled else 'Disabled'} (opacity: {int(pad_opacity * 100)}%, height: {int(pad_h_pct)}%)"
        )
        h_summary = (
            f"{h_text} ({h_pos}, {h_font} {h_size}pt, {h_color})"
            if h_text
            else "[None]"
        )
        print(f" 12. Custom Creator Handle:  {h_summary}")
        print("\n--- Atmospheric & Particle Effects (TikTok Viral FX) ---")
        print(f" 13. Atmospheric Particles:  {part_label}")
        print(
            f"     Settings:               Density: {curr_part_density.title()} | Speed: {curr_part_speed.title()} | Audio: {curr_part_audio.title()} | Color: {curr_part_color}"
        )
        print("\n  0. Done (Save and Return)\n")

        choice = _safe_input("Select an option to modify [1-13, 0 to done]: ").strip()
        if choice in ("0", "b", "back", ""):
            break
        elif choice == "1":
            print("\nAvailable Fonts:")
            for f_i, f_name in enumerate(fonts, 1):
                marker = " (CURRENT)" if f_name == font_name else ""
                print(f"  {f_i}. {f_name}{marker}")
            f_ch = _safe_input(f"Select font [1-{len(fonts)}]: ").strip()
            try:
                f_idx = int(f_ch) - 1
                if 0 <= f_idx < len(fonts):
                    lyrics_cfg["font"] = fonts[f_idx]
            except ValueError:
                pass
        elif choice == "2":
            val = _safe_input(
                f"Enter font size in points [current {font_size}]: "
            ).strip()
            if val.isdigit() and int(val) > 10:
                lyrics_cfg["font_size"] = int(val)
        elif choice == "3":
            val = _safe_input(
                f"Enter vertical position percent 10-90 [current {pos_y}%]: "
            ).strip()
            try:
                p_val = float(val)
                if 10.0 <= p_val <= 90.0:
                    lyrics_cfg["position_y_percent"] = p_val
            except ValueError:
                pass
        elif choice == "4":
            print("\nSelect Dynamic Karaoke Animation Style:")
            for s_idx, (s_key, s_desc) in enumerate(karaoke_styles, 1):
                marker = " (CURRENT)" if s_key == curr_style else ""
                print(f"  {s_idx}. {s_desc}{marker}")
            s_ch = _safe_input(f"Select [1-{len(karaoke_styles)}]: ").strip()
            try:
                sel_s = int(s_ch) - 1
                if 0 <= sel_s < len(karaoke_styles):
                    lyrics_cfg["karaoke_style"] = karaoke_styles[sel_s][0]
            except ValueError:
                pass
        elif choice == "5":
            lyrics_cfg["highlight_color"] = color_picker_menu(
                h_color, "Highlight / Active Word Color"
            )
        elif choice == "6":
            lyrics_cfg["base_color"] = color_picker_menu(
                b_color, "Base Inactive Lyric Color"
            )
        elif choice == "7":
            print("\n1. Choose Outline Color Preset\n2. Set Outline Width (px)")
            sub_o = _safe_input("Select [1-2]: ").strip()
            if sub_o == "1":
                lyrics_cfg["outline_color"] = color_picker_menu(
                    o_color, "Outline / Stroke Color"
                )
            elif sub_o == "2":
                w_in = _safe_input(
                    f"Enter outline width [current {o_width}px]: "
                ).strip()
                if w_in.isdigit():
                    lyrics_cfg["outline_width"] = int(w_in)
        elif choice == "8":
            lyrics_cfg["shadow"] = not shadow_on
        elif choice == "9":
            print("\nSelect Audio Visualizer Engine:")
            for v_i, (v_k, v_label) in enumerate(visualizer_types, 1):
                marker = " (ACTIVE)" if v_k == curr_viz_type else ""
                print(f"  {v_i}. {v_label}{marker}")
            v_ch = _safe_input(f"Select [1-{len(visualizer_types)}]: ").strip()
            try:
                v_idx = int(v_ch) - 1
                if 0 <= v_idx < len(visualizer_types):
                    viz_cfg["type"] = visualizer_types[v_idx][0]
            except ValueError:
                pass
        elif choice == "10":
            print("\nVisualizer Placement & Dimensions:")
            print("1. Set Screen Placement / Alignment")
            print("2. Set Visualizer Height (% of screen)")
            print("3. Set Visualizer Color Palette")
            v_sub = _safe_input("Select [1-3, 0 to return]: ").strip()
            if v_sub == "1":
                print("\nVisualizer Placement Options:")
                for p_i, (p_k, p_label) in enumerate(visualizer_positions, 1):
                    marker = " (ACTIVE)" if p_k == curr_viz_pos else ""
                    print(f"  {p_i}. {p_label}{marker}")
                p_ch = _safe_input(f"Select [1-{len(visualizer_positions)}]: ").strip()
                try:
                    p_idx = int(p_ch) - 1
                    if 0 <= p_idx < len(visualizer_positions):
                        viz_cfg["position"] = visualizer_positions[p_idx][0]
                except ValueError:
                    pass
            elif v_sub == "2":
                h_in = _safe_input(
                    f"Enter visualizer height percent 5-30% [current {curr_viz_h_pct}%]: "
                ).strip()
                try:
                    h_val = float(h_in)
                    if 3.0 <= h_val <= 40.0:
                        viz_cfg["height_pct"] = h_val
                except ValueError:
                    pass
            elif v_sub == "3":
                viz_cfg["color"] = color_picker_menu(
                    curr_viz_color, "Visualizer Accent Color"
                )
        elif choice == "11":
            new_state = not pad_enabled
            pad_cfg["enabled"] = new_state
            if new_state:
                op_in = _safe_input(
                    f"Enter pad darkness opacity 0.1 to 1.0 [current {pad_opacity}]: "
                ).strip()
                try:
                    op_val = float(op_in)
                    if 0.05 <= op_val <= 1.0:
                        pad_cfg["opacity"] = op_val
                except ValueError:
                    pass
                h_in = _safe_input(
                    f"Enter pad height percent 15 to 40% [current {pad_h_pct}%]: "
                ).strip()
                try:
                    h_val = float(h_in)
                    if 10.0 <= h_val <= 50.0:
                        pad_cfg["height_pct"] = h_val
                except ValueError:
                    pass
        elif choice == "12":
            while True:
                clear_screen()
                print_header("Creator Handle Customization")
                cur_text = handle_cfg.get("text", "")
                cur_pos = handle_cfg.get("position", "bottom_right")
                cur_font = handle_cfg.get("font", "Montserrat")
                cur_size = handle_cfg.get("font_size", 28)
                cur_col = handle_cfg.get("color", "&H00FFFFFF&")
                cur_out = handle_cfg.get("outline_color", "&H00000000&")
                cur_out_w = handle_cfg.get("outline_width", 2)
                cur_opa = float(handle_cfg.get("opacity", 0.85))

                print(
                    f"  Handle Text:       {cur_text if cur_text else '[None / Disabled]'}"
                )
                print(f"  Placement:         {cur_pos}")
                print(f"  Font Family:       {cur_font}")
                print(f"  Font Size:         {cur_size} pt")
                print(f"  Font Color:        {cur_col}")
                print(f"  Outline / Border:  {cur_out} ({cur_out_w}px)")
                print(f"  Opacity:           {int(cur_opa * 100)}%")
                print("\n  1. Edit Handle Text (e.g. @TikTokWizard.Official)")
                print(
                    "  2. Change Screen Placement (bottom_right [default], mid_right, top_left, etc.)"
                )
                print("  3. Change Font Family (Montserrat, Chango, Arial, etc.)")
                print("  4. Change Font Size (pt)")
                print("  5. Change Font Color")
                print("  6. Change Outline Color & Width")
                print("  7. Change Opacity (10% - 100%)")
                print("  8. Clear / Disable Handle")
                print("\n  0. Return to Visuals Menu\n")

                sub_h = _safe_input("Select [1-8, 0 to return]: ").strip()
                if sub_h in ("0", "b", "back", ""):
                    break
                elif sub_h == "1":
                    t_in = _safe_input(
                        f"Enter Creator Handle [current '{cur_text}']: "
                    ).strip()
                    if t_in:
                        handle_cfg["text"] = t_in
                elif sub_h == "2":
                    positions = [
                        ("top_left", "Top Left"),
                        ("top_center", "Top Center"),
                        ("top_right", "Top Right"),
                        ("mid_left", "Middle Left"),
                        ("mid_center", "Middle Center"),
                        ("mid_right", "Middle Right"),
                        ("bottom_left", "Bottom Left"),
                        ("bottom_center", "Bottom Center"),
                        ("bottom_right", "Bottom Right [DEFAULT]"),
                    ]
                    print("\nChoose Handle Screen Placement:")
                    for p_i, (p_k, p_label) in enumerate(positions, 1):
                        p_marker = " (CURRENT)" if p_k == cur_pos else ""
                        print(f"  {p_i}. {p_label}{p_marker}")
                    p_ch = _safe_input(f"Select [1-{len(positions)}]: ").strip()
                    try:
                        p_idx = int(p_ch) - 1
                        if 0 <= p_idx < len(positions):
                            handle_cfg["position"] = positions[p_idx][0]
                    except ValueError:
                        pass
                elif sub_h == "3":
                    print("\nAvailable Fonts:")
                    for f_i, f_name in enumerate(fonts, 1):
                        marker = " (CURRENT)" if f_name == cur_font else ""
                        print(f"  {f_i}. {f_name}{marker}")
                    f_ch = _safe_input(f"Select font [1-{len(fonts)}]: ").strip()
                    try:
                        f_idx = int(f_ch) - 1
                        if 0 <= f_idx < len(fonts):
                            handle_cfg["font"] = fonts[f_idx]
                    except ValueError:
                        pass
                elif sub_h == "4":
                    s_in = _safe_input(
                        f"Enter font size in points [current {cur_size}]: "
                    ).strip()
                    if s_in.isdigit() and int(s_in) > 6:
                        handle_cfg["font_size"] = int(s_in)
                elif sub_h == "5":
                    handle_cfg["color"] = color_picker_menu(
                        cur_col, "Creator Handle Text Color"
                    )
                elif sub_h == "6":
                    print("\n1. Set Outline / Border Color\n2. Set Outline Width (px)")
                    o_sel = _safe_input("Select [1-2]: ").strip()
                    if o_sel == "1":
                        handle_cfg["outline_color"] = color_picker_menu(
                            cur_out, "Creator Handle Outline Color"
                        )
                    elif o_sel == "2":
                        ow_in = _safe_input(
                            f"Enter outline width [current {cur_out_w}px]: "
                        ).strip()
                        if ow_in.isdigit():
                            handle_cfg["outline_width"] = int(ow_in)
                elif sub_h == "7":
                    op_in = _safe_input(
                        f"Enter opacity (0.1 to 1.0) [current {cur_opa}]: "
                    ).strip()
                    try:
                        op_val = float(op_in)
                        if 0.05 <= op_val <= 1.0:
                            handle_cfg["opacity"] = op_val
                    except ValueError:
                        pass
                elif sub_h == "8":
                    handle_cfg["text"] = ""
                    print("Creator handle cleared.")
                    time.sleep(1)
        elif choice == "13":
            while True:
                clear_screen()
                print_header("Atmospheric & Particle Effects Engine (TikTok Viral FX)")
                curr_t = particle_cfg.get("type", "none")
                curr_d = particle_cfg.get("density", "medium")
                curr_s = particle_cfg.get("speed", "normal")
                curr_a = particle_cfg.get("audio_reactive", "pulse")
                curr_c = particle_cfg.get("color", "auto")
                print(f"  Active Style:      {part_type_names.get(curr_t, curr_t)}")
                print(f"  Particle Density:  {curr_d.title()}")
                print(f"  Drift Speed:       {curr_s.title()}")
                print(
                    f"  Audio Reactivity:  {'On (Beat Pulse/Bursts)' if curr_a != 'off' else 'Off (Ambient Drift)'}"
                )
                print(f"  Color Palette:     {curr_c}")
                print(
                    "\n  1. Select Atmosphere Preset Style (Bokeh, Embers, Snow, Beat Sparks, Stars)"
                )
                print("  2. Set Particle Density (Low, Medium, High)")
                print("  3. Set Drift Speed (Slow, Normal, Fast)")
                print("  4. Toggle Audio Reactivity (Beat Pulse)")
                print("  5. Choose Color Palette (Auto or Custom Accent)")
                print("\n  0. Back to Visuals Menu\n")
                sub_c = _safe_input("Select [1-5, 0 to return]: ").strip()
                if sub_c in ("0", "b", "back", ""):
                    break
                elif sub_c == "1":
                    particle_styles = [
                        (
                            "bokeh_dust",
                            "Bokeh Dust (Warm floating luminous golden orbs - Lo-Fi/Aesthetic)",
                        ),
                        (
                            "rising_embers",
                            "Rising Embers (Fiery upward cinders with micro-flicker - Phonk/Metal/Trap)",
                        ),
                        (
                            "cinematic_snow",
                            "Cinematic Snow (Parallax 3-layer snowfall with wind sway - Chill/Trailer)",
                        ),
                        (
                            "radial_burst",
                            "Radial Beat Burst (Audio-reactive spark shockwaves - EDM/Club)",
                        ),
                        (
                            "neon_stars",
                            "Neon Stars (4-point twinkling diamond cross stars - Synthwave/Cyberpunk)",
                        ),
                        (
                            "none",
                            "None / Disabled (Clean video background without particles)",
                        ),
                    ]
                    print("\nSelect Atmospheric Style:")
                    for p_i, (p_k, p_label) in enumerate(particle_styles, 1):
                        marker = " (ACTIVE)" if p_k == curr_t else ""
                        print(f"  {p_i}. {p_label}{marker}")
                    p_sel = _safe_input(f"Select [1-{len(particle_styles)}]: ").strip()
                    try:
                        p_idx = int(p_sel) - 1
                        if 0 <= p_idx < len(particle_styles):
                            particle_cfg["type"] = particle_styles[p_idx][0]
                    except ValueError:
                        pass
                elif sub_c == "2":
                    densities = [
                        ("low", "Low (30 particles - Subtle atmospheric accent)"),
                        (
                            "medium",
                            "Medium (65 particles - Balanced studio feel) [DEFAULT]",
                        ),
                        (
                            "high",
                            "High (130 particles - Dense viral visual atmosphere)",
                        ),
                    ]
                    print("\nSelect Particle Density:")
                    for d_i, (d_k, d_label) in enumerate(densities, 1):
                        marker = " (ACTIVE)" if d_k == curr_d else ""
                        print(f"  {d_i}. {d_label}{marker}")
                    d_sel = _safe_input(f"Select [1-{len(densities)}]: ").strip()
                    try:
                        d_idx = int(d_sel) - 1
                        if 0 <= d_idx < len(densities):
                            particle_cfg["density"] = densities[d_idx][0]
                    except ValueError:
                        pass
                elif sub_c == "3":
                    speeds = [
                        ("slow", "Slow (0.6x - Gentle, relaxed ambient drift)"),
                        ("normal", "Normal (1.0x - Standard tempo) [DEFAULT]"),
                        ("fast", "Fast (1.8x - High-velocity, energetic surges)"),
                    ]
                    print("\nSelect Particle Drift Speed:")
                    for s_i, (s_k, s_label) in enumerate(speeds, 1):
                        marker = " (ACTIVE)" if s_k == curr_s else ""
                        print(f"  {s_i}. {s_label}{marker}")
                    s_sel = _safe_input(f"Select [1-{len(speeds)}]: ").strip()
                    try:
                        s_idx = int(s_sel) - 1
                        if 0 <= s_idx < len(speeds):
                            particle_cfg["speed"] = speeds[s_idx][0]
                    except ValueError:
                        pass
                elif sub_c == "4":
                    if curr_a == "off":
                        particle_cfg["audio_reactive"] = "pulse"
                    else:
                        particle_cfg["audio_reactive"] = "off"
                elif sub_c == "5":
                    print("\nColor Palette Options:")
                    print(
                        "  1. Auto (Preset-matched signature color: Gold, Crimson, Ice White, Cyan, Pink)"
                    )
                    print("  2. Custom Hex / Color Picker")
                    col_c = _safe_input("Select [1-2]: ").strip()
                    if col_c == "1":
                        particle_cfg["color"] = "auto"
                    elif col_c == "2":
                        cur_hex = curr_c if curr_c.startswith("#") else "#ffd700"
                        particle_cfg["color"] = color_picker_menu(
                            cur_hex, "Particle Atmosphere Color"
                        )

    return fx


def select_background_menu(
    selected_fx: dict, defaults_dir: Path, project_folder: Path
) -> dict:
    fx = json.loads(json.dumps(selected_fx))
    bg_cfg = fx.setdefault("background", {})

    media_exts = {
        ".mp4",
        ".mov",
        ".mkv",
        ".webm",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
    }

    while True:
        clear_screen()
        print_header("Select Background Media Canvas")
        curr_type = bg_cfg.get("type", "video")
        curr_file = bg_cfg.get("file")
        curr_mode = bg_cfg.get("fit_mode", "blur_pad")

        found_candidates: List[Path] = []
        if defaults_dir.exists():
            for f in sorted(defaults_dir.rglob("*")):
                if f.is_file() and f.suffix.lower() in media_exts:
                    found_candidates.append(f)
        for f in sorted(project_folder.iterdir()):
            if (
                f.is_file()
                and f.suffix.lower() in media_exts
                and not f.name.endswith(".mp4")
            ):
                found_candidates.append(f)

        print(
            f"Current Background: {curr_file if curr_file else 'Automatic / Solid Canvas'}"
        )
        print(f"Current Fit Mode:   {curr_mode}")
        print()
        print("  1. Solid Black / Dark Canvas (Clean distraction-free)")
        print("  2. Change Fit Mode (blur_pad, crop_fill, pad, dark_pad, mirror_pad)")
        if found_candidates:
            print("\nAvailable Media Files in _Assets & Project Folder:")
            for m_i, m_p in enumerate(found_candidates, 3):
                marker = (
                    " (SELECTED)"
                    if curr_file and Path(curr_file).name == m_p.name
                    else ""
                )
                print(f"  {m_i}. {m_p.name}{marker} [{m_p.parent.name}]")
        print("\n  0. Return to Project Menu\n")

        choice = _safe_input(f"Select option [0-{len(found_candidates) + 2}]: ").strip()
        if choice in ("0", "b", "back", ""):
            break
        elif choice == "1":
            bg_cfg["file"] = None
            bg_cfg["type"] = "color"
            bg_cfg["colors"] = ["#0a0a0a"]
            print("Set background to solid dark canvas.")
            time.sleep(1)
        elif choice == "2":
            modes = ["blur_pad", "crop_fill", "pad", "dark_pad", "mirror_pad"]
            print("\nFit Modes:")
            for mode_i, mode_name in enumerate(modes, 1):
                print(f"  {mode_i}. {mode_name}")
            m_ch = _safe_input("Select fit mode: ").strip()
            try:
                m_idx = int(m_ch) - 1
                if 0 <= m_idx < len(modes):
                    bg_cfg["fit_mode"] = modes[m_idx]
            except ValueError:
                pass
        else:
            try:
                c_idx = int(choice) - 3
                if 0 <= c_idx < len(found_candidates):
                    target_file = found_candidates[c_idx]
                    bg_cfg["file"] = str(target_file)
                    ext = target_file.suffix.lower()
                    if ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                        bg_cfg["type"] = "image"
                    elif ext == ".gif":
                        bg_cfg["type"] = "gif"
                    else:
                        bg_cfg["type"] = "video"
            except ValueError:
                pass

    return fx


def select_resolution_menu(config: dict) -> dict:
    while True:
        clear_screen()
        print_header("Select Video Resolution & Aspect Ratio")
        curr_w = int(config.get("video_width", 1080))
        curr_h = int(config.get("video_height", 1920))

        for idx, (res_name, desc, w, h) in enumerate(STANDARD_RESOLUTIONS, 1):
            marker = " (CURRENT)" if curr_w == w and curr_h == h else ""
            print(f"  {idx}. {res_name}{marker}")
            print(f"     Target: {desc}\n")

        print("  0. Keep Current and Return\n")
        choice = _safe_input(
            f"Select [1-{len(STANDARD_RESOLUTIONS)}, 0 to return]: "
        ).strip()
        if choice in ("0", "b", "back", ""):
            break

        try:
            sel_idx = int(choice) - 1
            if 0 <= sel_idx < len(STANDARD_RESOLUTIONS):
                _, _, w, h = STANDARD_RESOLUTIONS[sel_idx]
                config["video_width"] = w
                config["video_height"] = h
                save_config(config)
                print(f"Resolution set to {w}x{h}.")
                time.sleep(1)
                break
        except ValueError:
            pass
        print("Invalid selection.")
        time.sleep(1)

    return config


def run_alignment_and_render_flow(
    project_folder: Path, config: dict, defaults_dir: Path, history_path: Path
) -> None:
    slug = project_folder.name
    audio_path = find_audio(project_folder)
    lyrics_path = find_lyrics(project_folder)

    if not audio_path or not lyrics_path:
        print("\nMissing audio or lyrics file in project folder.")
        _safe_input("\nPress Enter to return...")
        return

    ffmpeg_bin = config.get("ffmpeg_path", "ffmpeg")
    total_audio_duration = probe_audio_duration(ffmpeg_bin, audio_path)

    dot_data = get_project_data_dir(project_folder)
    ass_path = dot_data / "karaoke.ass"
    meta_path = dot_data / "karaoke_meta.json"
    synced_lyrics_path = dot_data / "lyrics_synced.txt"
    manifest_path = dot_data / "alignment_manifest.json"

    profiles = list_profiles()
    selected_profile_key = config.get("last_used_profile", "default")
    if selected_profile_key not in profiles:
        selected_profile_key = "default"
    selected_fx = dict(profiles[selected_profile_key])

    while True:
        clear_screen()
        has_alignment = manifest_path.exists()
        status_str = (
            "ALIGNED" if has_alignment else "NOT ALIGNED (Will run Universal Engine)"
        )

        print_header(f"Project: {slug}")
        lyrics_p = selected_fx.get("lyrics", {})
        pad_p = selected_fx.get("tiktok_pad", {})
        handle_p = selected_fx.get("creator_handle", {})
        viz_p = selected_fx.get("visualizer", {})
        part_p = selected_fx.get("particles", {})

        print(f"  Audio:       {audio_path.name} ({fmt_seconds(total_audio_duration)})")
        print(f"  Lyrics:      {lyrics_path.name}")
        print(f"  Status:      {status_str}")
        active_cp = resolve_active_profile(ffmpeg_bin=ffmpeg_bin, verify_encoders=False)
        print(
            f"  Compute:     {active_cp.display_name} [Stems: {active_cp.stem_mode.upper()}, Video: {active_cp.video_codec}]"
        )
        print(
            f"  Resolution:  {config.get('video_width', 1080)}x{config.get('video_height', 1920)}"
        )
        print(
            f"  Profile:     {selected_profile_key.upper()} ({selected_fx.get('name', selected_profile_key)})"
        )
        bg_desc = selected_fx.get("background", {}).get("file")
        print(
            f"  Background:  {Path(bg_desc).name if bg_desc else 'Solid / Auto Canvas'}"
        )
        print(
            f"  Font:        {lyrics_p.get('font', 'Chango')} ({lyrics_p.get('font_size', 72)}pt, Y:{lyrics_p.get('position_y_percent', 65)}%)"
        )
        print(
            f"  Colors:      Active: {lyrics_p.get('highlight_color', 'N/A')} | Base: {lyrics_p.get('base_color', 'N/A')}"
        )
        viz_type = viz_p.get("type", "none")
        print(
            f"  Visualizer:  {viz_type} ({viz_p.get('position', 'above_pad')}, H:{viz_p.get('height_pct', 12)}%, Col:{viz_p.get('color', '#00ffff')})"
        )
        part_desc = part_p.get("type", "none")
        print(
            f"  Atmosphere:  {part_desc.replace('_', ' ').title()} (Density: {part_p.get('density', 'medium')}, Audio: {part_p.get('audio_reactive', 'pulse')}, Col: {part_p.get('color', 'auto')})"
        )
        pad_stat = (
            f"Enabled ({int(float(pad_p.get('opacity', 0.90)) * 100)}% darkness, H:{int(float(pad_p.get('height_pct', 12.5)))}%)"
            if pad_p.get("enabled", True)
            else "Disabled"
        )
        print(f"  TikTok Pad:  {pad_stat}")
        h_text_p = handle_p.get("text", "")
        h_desc = (
            f"{h_text_p} ({handle_p.get('position', 'bottom_right')}, {handle_p.get('font', 'Montserrat')} {handle_p.get('font_size', 28)}pt, {handle_p.get('color', '&H00FFFFFF&')})"
            if h_text_p
            else "[None / Disabled]"
        )
        print(f"  Handle:      {h_desc}")
        print()
        print("  1. Render Video (Start FFmpeg)")
        print(
            "  2. Select Visual Profile Preset (20 Curated Styles: Cyberpunk, EDM, Phonk, Lo-Fi, Synthwave...)"
        )
        print("  3. Customize Visuals & Typography (Font, Size, Colors, Position)")
        print("  4. Select Background Media (Video, GIF, Image, Solid Color)")
        print(
            "  5. Select Video Resolution & Aspect Ratio (TikTok 9:16, 2K, 4K, 1:1, 16:9)"
        )
        print("  6. Force Re-run Alignment Engine (Recalculate Stems & Timings)")
        print("  7. Open Project Folder in File Explorer")
        print(
            "  8. Extract & Export Vocal Stems (Save Acapella WAV/MP3 to Project Folder)"
        )
        print("\n  0. Back to Project List\n")

        choice = _safe_input("Select an option [1-8, 0 to back]: ").strip()
        if choice in ("0", "b", "back"):
            break
        elif choice == "2":
            selected_profile_key = select_profile_menu(selected_profile_key, profiles)
            selected_fx = dict(profiles[selected_profile_key])
            config["last_used_profile"] = selected_profile_key
            save_config(config)
        elif choice == "3":
            selected_fx = customize_visuals_menu(selected_fx, defaults_dir)
        elif choice == "4":
            selected_fx = select_background_menu(
                selected_fx, defaults_dir, project_folder
            )
        elif choice == "5":
            config = select_resolution_menu(config)
        elif choice == "7":
            open_folder_and_select(project_folder)
        elif choice == "8":
            clear_screen()
            print_header(f"Export Vocal Stems & Acapella — {slug}")
            print_wrapped(
                "Extracting synchronized isolated vocal stems from audio track..."
            )
            print()
            from src.pipeline.vocal_exporter import export_vocal_stems

            try:
                exported = export_vocal_stems(
                    project_folder=project_folder,
                    audio_path=audio_path,
                    ffmpeg_bin=ffmpeg_bin,
                    export_mp3=True,
                    log_fn=print,
                )
                print()
                vocal_stem_dir = project_folder / "Vocal Stem"
                print_wrapped(
                    f"Successfully saved {len(exported)} isolated vocal stems to:"
                )
                print(f"  -> {vocal_stem_dir.resolve()}\n")
                open_choice = (
                    _safe_input("Open 'Vocal Stem' folder in File Explorer? [Y/n]: ")
                    .strip()
                    .lower()
                )
                if open_choice in ("", "y", "yes"):
                    open_folder_and_select(vocal_stem_dir)
            except Exception as e:
                print(f"\n[Error] Failed to export vocal stems: {e}")
                _safe_input("\nPress Enter to continue...")
        elif choice in ("1", "6"):
            force_align = choice == "6"
            manifest = None

            if has_alignment and not force_align:
                try:
                    manifest = load_json(manifest_path, None)
                except Exception:
                    manifest = None

            if not manifest or force_align:
                active_cp = resolve_active_profile(ffmpeg_bin=ffmpeg_bin, verify_encoders=False)
                print(
                    f"\nRunning Greenfield Universal Engine v5.0 Alignment ({active_cp.display_name}, stems={active_cp.stem_mode})..."
                )
                try:
                    from src.align_v2.pipeline import V5AlignmentEngine

                    engine = V5AlignmentEngine(
                        tau_song=float(config.get("tau_song", 0.23)),
                        word_target=float(config.get("word_target", 0.95)),
                        stem_mode=active_cp.stem_mode,
                    )
                    manifest = engine.run(
                        lyric_path=str(lyrics_path),
                        mp3_path=str(audio_path),
                        out_dir=str(dot_data),
                        log=print,
                    )
                except Exception as e_v5:
                    print(f"[align] Greenfield v5 error ({e_v5}), falling back to UniversalEngine v1...")
                    engine = UniversalEngine(
                        tau_song=float(config.get("tau_song", 0.23)),
                        word_target=float(config.get("word_target", 0.95)),
                        whisper_size=str(config.get("whisper_model", "small")),
                    )
                    manifest = engine.run(
                        lyric_path=str(lyrics_path),
                        mp3_path=str(audio_path),
                        out_dir=str(dot_data),
                        log=print,
                    )
                try:
                    from src.pipeline.vocal_exporter import export_vocal_stems

                    export_vocal_stems(
                        project_folder=project_folder,
                        audio_path=audio_path,
                        ffmpeg_bin=ffmpeg_bin,
                        export_mp3=True,
                        log_fn=lambda _: None,
                    )
                except Exception:
                    pass

            all_words = manifest.get("words", [])
            lines_info = manifest.get("lines", [])
            karaoke_entries: List[KaraokeEntry] = []
            for li_info in lines_info:
                idx = int(li_info["index"]) - 1
                line_words = [
                    w for w in all_words if int(w.get("line_index", -1)) == idx
                ]
                if not line_words:
                    continue
                k_words = [
                    KaraokeWord(
                        text=str(w["word"]),
                        duration_cs=max(
                            3,
                            int(
                                round(
                                    (float(w["end_sec"]) - float(w["start_sec"]))
                                    * 100.0
                                )
                            ),
                        ),
                        start=float(w["start_sec"]),
                        end=float(w["end_sec"]),
                    )
                    for w in line_words
                ]
                karaoke_entries.append(
                    KaraokeEntry(
                        line_index=idx,
                        words=k_words,
                        start=float(li_info["start_sec"]),
                        end=float(li_info["end_sec"]),
                        raw_text=str(li_info["text"]),
                    )
                )

            active_cp = resolve_active_profile(ffmpeg_bin=ffmpeg_bin, verify_encoders=False)
            vid_w = int(config.get("video_width", 1080))
            vid_h = int(config.get("video_height", 1920))
            render_config = dict(config)
            render_config["video_width"] = vid_w
            render_config["video_height"] = vid_h
            render_config["codec"] = active_cp.video_codec
            render_config["compute_profile"] = active_cp

            write_ass_karaoke(
                entries=karaoke_entries,
                ass_path=ass_path,
                config=render_config,
                effect_cfg=selected_fx,
                video_width=vid_w,
                video_height=vid_h,
                defaults_dir=defaults_dir,
            )
            export_easy_lyrics_file(ass_path, synced_lyrics_path)

            karaoke_meta = {
                "schema_version": "shotcut_tiktok.karaoke_meta.v5",
                "word_count": len(all_words),
                "line_count": len(lines_info),
                "cemented_words_ge_95_count": manifest.get(
                    "cemented_words_ge_95_count", len(all_words)
                ),
                "uncemented_words_count": manifest.get("uncemented_words_count", 0),
                "ass_checksum": sha256_file(ass_path),
                "lyrics_checksum": sha256_file(lyrics_path),
                "generated_at": now_iso(),
                "engine": manifest.get("engine", "greenfield_v5_dual_ctc_consensus"),
            }
            save_json_atomic(meta_path, karaoke_meta)

            if choice == "6":
                print(f"\n[OK] Alignment refreshed successfully for {slug}.")
                _safe_input("\nPress Enter to return to project menu...")
                continue

            pref_bg = selected_fx.get("background", {}).get("file")
            background_path = (
                Path(pref_bg)
                if pref_bg and Path(pref_bg).exists()
                else find_default_background(defaults_dir, preferred_file=pref_bg)
            )

            # Pre-Render Configuration Summary
            clear_screen()
            print_header("Confirm Video Render Configuration")
            lyrics_p = selected_fx.get("lyrics", {})
            pad_p = selected_fx.get("tiktok_pad", {})
            handle_p = selected_fx.get("creator_handle", {})
            viz_p = selected_fx.get("visualizer", {})

            print("Please review your active render settings:\n")
            print(f"  Project:            {slug}")
            print(f"  Resolution:         {vid_w}x{vid_h} @ {config.get('fps', 30)}fps")
            print(f"  Visual Profile:     {selected_profile_key.upper()}")
            print(
                f"  Karaoke Style:      {lyrics_p.get('karaoke_style', 'word_highlight')}"
            )
            print(
                f"  Font Family / Size: {lyrics_p.get('font', 'Chango')} @ {lyrics_p.get('font_size', 72)}pt"
            )
            print(
                f"  Colors:             Active: {lyrics_p.get('highlight_color', 'N/A')} | Base: {lyrics_p.get('base_color', 'N/A')}"
            )
            print(
                f"  Lyrics Y-Position:  {lyrics_p.get('position_y_percent', 65)}% from top"
            )
            print(
                f"  Background Media:   {Path(background_path).name if background_path else 'Solid / Auto Canvas'}"
            )
            viz_summary = (
                f"{viz_p.get('type', 'none')} (Position: {viz_p.get('position', 'above_pad')}, Height: {viz_p.get('height_pct', 12)}%, Color: {viz_p.get('color', '#00ffff')})"
                if viz_p.get("type", "none") != "none"
                else "Disabled"
            )
            print(f"  Visualizer:         {viz_summary}")
            part_p = selected_fx.get("particles", {})
            p_style = part_p.get("type", "none")
            p_dens = part_p.get("density", "medium")
            p_react = part_p.get("audio_reactive", "pulse")
            p_col = part_p.get("color", "auto")
            part_summary = (
                f"{p_style.replace('_', ' ').title()} (Density: {p_dens.title()}, Reactivity: {p_react.title()}, Color: {p_col})"
                if p_style != "none"
                else "Disabled"
            )
            print(f"  Atmosphere/FX:      {part_summary}")
            print(
                f"  TikTok Safety Pad:  {'Enabled (' + str(int(float(pad_p.get('opacity', 0.90)) * 100)) + '% darkness, bottom ' + str(int(float(pad_p.get('height_pct', 12.5)))) + '%)' if pad_p.get('enabled', True) else 'Disabled'}"
            )
            h_rend_desc = (
                f"{handle_p.get('text')} ({handle_p.get('position', 'bottom_right')}, {handle_p.get('font', 'Montserrat')} {handle_p.get('font_size', 28)}pt, {handle_p.get('color', '&H00FFFFFF&')})"
                if handle_p.get("text")
                else "[None / Disabled]"
            )
            print(f"  Creator Handle:     {h_rend_desc}")
            print(f"  Audio Duration:     {fmt_seconds(total_audio_duration)}")

            render_audio_path = audio_path
            acapella_cand = project_folder / "Vocal Stem" / "Acapella (Full Vocals).wav"
            if acapella_cand.exists():
                print(f"  Audio Track:        Master Mix (Original Song)")
                print()
                print("  Audio Track Option:")
                print(
                    "    1. Master Mix (Original Track with Instruments & Beats) [DEFAULT]"
                )
                print("    2. Isolated Acapella Vocals (Karaoke / Vocal Performance)")
                aud_choice = _safe_input(
                    "  Choose audio track [1-2, default=1]: "
                ).strip()
                if aud_choice == "2":
                    render_audio_path = acapella_cand
                    print("  -> Selected: Isolated Acapella Vocals for Video Audio!")
            print()

            confirm = _safe_input("Proceed with video render? [Y/n]: ").strip().lower()
            if confirm in ("n", "no", "cancel"):
                print("Render cancelled. Returning to menu.")
                time.sleep(1)
                continue

            existing_versions = [0]
            for f in project_folder.glob("*.mp4"):
                m_ver = re.search(r"\bv(\d+)\.mp4$", f.name, re.IGNORECASE)
                if m_ver:
                    existing_versions.append(int(m_ver.group(1)))
            ver = max(existing_versions) + 1
            output_path = project_folder / f"{slug} v{ver}.mp4"

            print(f"\n[1/2] Preparing audio-visual layers for {output_path.name}...")
            if p_style != "none":
                print(f"  -> Generating {p_style} atmospheric particle overlay...")
            if viz_p.get("type", "none") != "none":
                print(f"  -> Generating {viz_p.get('type')} visualizer overlay...")

            cmd = build_ffmpeg_command(
                config=render_config,
                effect_cfg=selected_fx,
                audio_path=render_audio_path,
                background_path=background_path,
                defaults_dir=defaults_dir,
                ass_path=ass_path,
                output_path=output_path,
                duration=total_audio_duration,
                start_time=0.0,
            )

            print(
                f"\n[2/2] Compositing & encoding video: {output_path.name} ({vid_w}x{vid_h}, codec={render_config.get('codec', 'libx264')}) with profile {selected_profile_key.upper()}..."
            )
            success, render_time, stderr = run_render_with_progress(
                cmd, total_audio_duration
            )
            if not success and str(render_config.get("codec", "libx264")) not in ("libx264", "h264"):
                print(
                    f"\n[WARN] Hardware encoder '{render_config.get('codec')}' failed; retrying safely with software libx264..."
                )
                render_config["codec"] = "libx264"
                cmd_sw = build_ffmpeg_command(
                    config=render_config,
                    effect_cfg=selected_fx,
                    audio_path=render_audio_path,
                    background_path=background_path,
                    defaults_dir=defaults_dir,
                    ass_path=ass_path,
                    output_path=output_path,
                    duration=total_audio_duration,
                    start_time=0.0,
                )
                success, render_time, stderr = run_render_with_progress(
                    cmd_sw, total_audio_duration
                )

            # Safe relocation of intermediate layer streams per File Retention Policy
            for tmp_layer in [
                output_path.parent / f".tmp_viz_{output_path.stem}.mov",
                output_path.parent / f".tmp_particles_{output_path.stem}.mov",
            ]:
                if tmp_layer.exists():
                    _safe_relocate_file(tmp_layer, "tmp_render_layer")

            if success:
                print(f"\n[OK] Rendered: {output_path}")
                manifest_path = write_video_render_manifest(
                    output_video_path=output_path,
                    config=render_config,
                    effect_cfg=selected_fx,
                    audio_path=render_audio_path,
                    lyrics_path=lyrics_path,
                    ass_path=ass_path,
                    background_path=background_path,
                    duration=total_audio_duration,
                    render_seconds=render_time,
                    iteration=ver,
                    profile_key=selected_profile_key,
                    project_folder=project_folder,
                    wizard_version="v5.4",
                )
                print(f"[OK] Settings Manifest: {manifest_path.name}")
                if config.get("auto_open_folder", True):
                    open_folder_and_select(output_path)
                record_iteration(
                    project_folder=project_folder,
                    iteration=ver,
                    effect_preset=selected_profile_key,
                    output_file=output_path,
                    render_seconds=render_time,
                    duration_seconds=total_audio_duration,
                    resolution=f"{vid_w}x{vid_h}",
                    fps=int(config.get("fps", 30)),
                    source_checksums={
                        audio_path.name: sha256_file(audio_path),
                        lyrics_path.name: sha256_file(lyrics_path),
                    },
                    notes="Greenfield v5.4 Dual-Model Consensus alignment",
                )
                append_history(
                    history_path,
                    {
                        "entry_id": str(uuid.uuid4())[:12],
                        "slug": slug,
                        "iteration": ver,
                        "effect_preset": selected_profile_key,
                        "output_file": str(output_path),
                        "render_seconds": round(render_time, 2),
                        "duration_seconds": round(total_audio_duration, 2),
                        "resolution": f"{vid_w}x{vid_h}",
                        "fps": int(config.get("fps", 30)),
                        "completed_at": now_iso(),
                    },
                )
            else:
                print("\n[ERROR] FFmpeg rendering failed.")
                if stderr:
                    print(stderr[-2000:])

            _safe_input("\nPress Enter to continue...")

def _load_narration_job_summary(job_dir: Path) -> dict:
    manifest = load_json(job_dir / "narration_job.json", {}) or {}
    progress = manifest.get("progress", {}) if isinstance(manifest, dict) else {}
    status = str(manifest.get("status", "unknown"))
    total = int(progress.get("total_chunks", 0) or 0)
    done = int(progress.get("completed_chunks", 0) or 0)
    out = manifest.get("output") if isinstance(manifest, dict) else None
    out_path = ""
    duration = ""
    if isinstance(out, dict):
        out_path = str(out.get("path", ""))
        duration = str(out.get("duration_hms", ""))
    backend = str(manifest.get("backend", manifest.get("params", {}).get("backend", "")) or "")
    return {
        "status": status,
        "done": done,
        "total": total,
        "output": out_path,
        "duration": duration,
        "script": str(manifest.get("script", "")) if isinstance(manifest, dict) else "",
        "voice": str(manifest.get("reference_voice", "")) if isinstance(manifest, dict) else "",
        "backend": backend,
    }


def _narration_backend_summary(caps: dict) -> tuple[bool, str, str]:
    """Return (any_available, preferred_mode, summary_text)."""
    f5 = caps.get("f5_tts", {}) or {}
    mms = caps.get("mms_fa", {}) or {}
    any_avail = bool(f5.get("available") or mms.get("available"))
    modes = set()
    for b in (f5, mms):
        if b.get("available") and b.get("mode"):
            modes.add(str(b.get("mode")))
    if "current" in modes and "env" in modes:
        mode = "mixed"
    elif "env" in modes:
        mode = "env"
    elif "current" in modes:
        mode = "current"
    else:
        mode = "none"
    parts = []
    for key, label in (("f5_tts", "F5-TTS clone"), ("mms_fa", "MMS Persian")):
        b = caps.get(key, {}) or {}
        st = "available" if b.get("available") else "not available"
        loc = b.get("mode", "none")
        parts.append(f"{label}: {st} [{loc}]")
    return any_avail, mode, " | ".join(parts)


def _choose_narration_backend(script_path: Path, caps: dict) -> str:
    try:
        sample = script_path.read_text(encoding="utf-8-sig", errors="ignore")[:8000]
    except Exception:
        sample = ""
    is_persian = detect_persian_script(sample)
    f5 = caps.get("f5_tts", {}) or {}
    mms = caps.get("mms_fa", {}) or {}
    default_choice = "2" if is_persian and mms.get("available") else ("1" if f5.get("available") else "2")

    print("\nSelect narration backend:")
    print("  1. F5-TTS voice cloning (best for English/Chinese; uses selected reference voice)")
    print("  2. MMS Persian (accurate Persian speech, fixed MMS voice; reference recorded for consent)")
    print("  3. Auto (prefer MMS for Persian scripts, else F5-TTS when available)")
    if is_persian:
        print("  Detected: Persian/Arabic script -> MMS Persian recommended for accuracy.")
    else:
        print("  Tip: For Persian lecture text, choose MMS Persian for accurate pronunciation/content.")
    if not f5.get("available"):
        print("  Note: F5-TTS backend not available right now.")
    if not mms.get("available"):
        print("  Note: MMS Persian backend deps not available right now.")
    choice = _safe_input(f"Choose backend [1-3, default {default_choice}]: ").strip() or default_choice
    if choice == "1":
        return "f5_tts"
    if choice == "2":
        return "mms_fa"
    if choice == "3":
        return "auto"
    return "auto"


def run_narration_studio(start_here_dir: Path) -> None:
    """Optional narration/voiceover workflow (separate from karaoke projects)."""
    paths = get_narration_paths(start_here_dir)
    ensure_narration_dirs(paths)

    while True:
        clear_screen()
        print_header("Narration Studio (Voiceover from Text)")
        print_wrapped(
            "Create long-form narration audio from a text script. For Persian lectures, use MMS Persian (accurate text). "
            "F5-TTS supports voice cloning but is best for English/Chinese; MMS Persian uses a fixed voice and records your reference sample for consent/traceability. "
            "This workflow is separate from karaoke/alignment and video rendering.",
            indent="  ",
        )
        print()
        caps = check_narration_backends(_project_root)
        any_available, backend_mode, backend_summary = _narration_backend_summary(caps)
        jobs = list_narration_jobs(paths.output_dir)
        resumable_jobs = 0
        for j in jobs:
            s = _load_narration_job_summary(j)
            if s["status"] in ("running", "interrupted", "failed") or (s["total"] and s["done"] < s["total"]):
                resumable_jobs += 1

        print(f"  Narration folder: {paths.root}")
        print(f"  Scripts:          {paths.scripts_dir} ({len(list_scripts(paths))} file(s))")
        print(f"  Voices:           {paths.voices_dir} ({len(list_voice_samples(paths))} file(s))")
        print(f"  Output:           {paths.output_dir} ({len(jobs)} job folder(s), {resumable_jobs} resumable)")
        mode_label = {"current": "app env", "env": "isolated env", "mixed": "app+isolated env", "none": "unavailable"}.get(backend_mode, backend_mode)
        print(f"  Backends:         [{mode_label}] {backend_summary}")
        f5b = caps.get("f5_tts", {}) or {}
        mmsb = caps.get("mms_fa", {}) or {}
        print(f"    - F5-TTS:       {'Available' if f5b.get('available') else 'Not available'} ({f5b.get('detail','')})")
        print(f"    - MMS Persian:  {'Available' if mmsb.get('available') else 'Not available'} ({mmsb.get('detail','')})")
        print()
        print("  1. Generate new narration (long-form, chunked)")
        print("  2. Resume existing narration job")
        print("  3. Open narration folder")
        print("  4. Refresh / show setup info")
        print("  0. Back\n")

        choice = _safe_input("Select [1-4, 0 to back]: ").strip().lower()
        if choice in ("0", "b", "back", "q", "exit", ""):
            return
        elif choice == "3":
            try:
                if os.name == "nt":
                    os.startfile(str(paths.root))
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", str(paths.root)])
                else:
                    subprocess.Popen(["xdg-open", str(paths.root)])
            except Exception:
                pass
            continue
        elif choice == "4":
            print("\nNarration notes (long lectures, 15 min to multi-hour):")
            print("  - Scripts are split into chunks, synthesized per chunk, and merged via streaming I/O (memory-bounded).")
            print("  - Each run creates a job folder in Output/ with chunk files and narration_job.json for progress/resume.")
            print("  - If interrupted, use 'Resume existing narration job' to continue from completed chunks.")
            print("  - Recommended: keep chunks enabled for long runs; use 400-900 chars/chunk and 0.2-0.4s pause.")
            print("\nBackend guidance:")
            print("  - MMS Persian (recommended for Persian .txt lectures): accurate Persian content, fixed MMS voice.")
            print("  - F5-TTS: voice cloning from your reference sample, but base model is best for English/Chinese (not reliably accurate for Persian).")
            print("  - Voice cloning a specific reference timbre onto accurate Persian speech requires an additional voice-conversion backend (not enabled by default).")
            if not f5b.get("available"):
                print("\nF5-TTS setup: install/configure f5-tts in isolated env DO NOT TOUCH\\envs\\narration (recommended).")
            if not mmsb.get("available"):
                print("MMS Persian setup: ensure transformers+torch and model access for facebook/mms-tts-fas in narration env.")
            _safe_input("\nPress Enter to continue...")
            continue
        elif choice == "2":
            all_jobs = list_narration_jobs(paths.output_dir)
            if not all_jobs:
                print("\nNo narration job folders found in Output/.")
                _safe_input("\nPress Enter to continue...")
                continue
            print("\nNarration jobs:")
            summaries = []
            for i, j in enumerate(all_jobs, 1):
                s = _load_narration_job_summary(j)
                summaries.append((j, s))
                prog = f"{s['done']}/{s['total']}" if s["total"] else "-/-"
                script_name = Path(s["script"]).name if s["script"] else "?"
                dur = f" dur {s['duration']}" if s.get("duration") else ""
                btag = f" [{s['backend']}]" if s.get("backend") else ""
                print(f"  {i}. {j.name}  [{s['status']}]{btag}  chunks {prog}{dur}  script: {script_name}")
            j_choice = _safe_input(f"Choose job to resume [1-{len(all_jobs)}, 0 to cancel]: ").strip()
            if j_choice in ("0", ""):
                continue
            try:
                job_dir, job_summary = summaries[int(j_choice) - 1]
            except Exception:
                print("Invalid job selection.")
                time.sleep(1)
                continue

            manifest = load_json(job_dir / "narration_job.json", {}) or {}
            params = manifest.get("params", {}) if isinstance(manifest, dict) else {}
            script_path = Path(str(manifest.get("script", "")))
            voice_raw = str(manifest.get("reference_voice", "") or "")
            voice_path = Path(voice_raw) if voice_raw else None

            # Legacy manifests predate backend field; prior behavior was F5-TTS.
            job_backend = str(params.get("backend") or manifest.get("backend") or "").strip().lower()
            if not job_backend:
                job_backend = "f5_tts"
            if job_backend not in ("f5_tts", "mms_fa"):
                print(f"\n[ERROR] Unsupported or unknown job backend: {job_backend}")
                _safe_input("\nPress Enter to continue...")
                continue

            if not script_path.exists():
                print("\n[ERROR] Cannot resume: original script file is missing.")
                print(f"  Script: {script_path}")
                _safe_input("\nPress Enter to continue...")
                continue

            # Voice requirements: F5 synthesis needs reference audio. MMS does not, but if a
            # non-empty reference voice path is recorded it should still be present.
            if job_backend == "f5_tts":
                if voice_path is None or not voice_path.exists():
                    print("\n[ERROR] Cannot resume F5-TTS job: reference voice file is missing.")
                    print(f"  Voice:  {voice_path if voice_path is not None else ''}")
                    _safe_input("\nPress Enter to continue...")
                    continue
            else:
                if voice_path is not None and voice_raw and not voice_path.exists():
                    print("\n[ERROR] Cannot resume: recorded reference voice file is missing.")
                    print(f"  Voice:  {voice_path}")
                    _safe_input("\nPress Enter to continue...")
                    continue
                if voice_path is None or not voice_raw:
                    # MMS jobs may have no stored reference voice; keep a placeholder path for worker args.
                    voice_path = script_path

            try:
                max_chars = int(params.get("max_chars_per_chunk", 700) or 700)
            except Exception:
                max_chars = 700
            try:
                pause_s = float(params.get("pause_seconds", 0.25))
            except Exception:
                pause_s = 0.25
            keep_chunks = bool(params.get("keep_chunks", True))
            stored_ref_text = str(params.get("ref_text", "") or "")

            caps = check_narration_backends(_project_root)
            backend_cap = caps.get(job_backend, {}) or {}
            if not backend_cap.get("available"):
                print(f"\n[ERROR] Cannot resume: required backend '{job_backend}' is not available.")
                print(f"  Detail: {backend_cap.get('detail', 'unavailable')}")
                _safe_input("\nPress Enter to continue...")
                continue
            exec_mode = str(backend_cap.get("mode", "none"))

            print(f"\nResuming job: {job_dir.name}")
            print(f"  Status: {job_summary['status']}  Progress: {job_summary['done']}/{job_summary['total'] or '?'} chunks")
            print(f"  Backend: {job_backend} ({exec_mode})")
            print(f"  Script: {script_path.name}")
            print(f"  Voice:  {Path(voice_raw).name if voice_raw else '(none recorded)'}")
            print(f"  Chunk size: {max_chars} chars | Pause: {pause_s:.2f}s | Keep chunks: {'Yes' if keep_chunks else 'No'}")
            print("\nRights & consent confirmation (required):")
            print("  - I confirm I own this voice sample or have explicit permission to clone/use it.")
            print("  - I will use the generated audio lawfully and in compliance with applicable rights and platform policies.")
            confirm_rights = _safe_input("Type YES to confirm and continue: ").strip()
            if confirm_rights != "YES":
                print("Confirmation not provided. Narration resume cancelled.")
                _safe_input("\nPress Enter to continue...")
                continue

            print("\nResuming narration generation...")
            try:
                gen_kwargs = dict(
                    script_path=script_path,
                    reference_voice_path=voice_path,
                    output_dir=paths.output_dir,
                    ref_text=stored_ref_text,
                    max_chars_per_chunk=max_chars,
                    pause_seconds=pause_s,
                    keep_chunks=keep_chunks,
                    resume=True,
                    job_name=job_dir.name,
                    progress=lambda m: print(f"  {m}"),
                    backend=job_backend,
                )
                if exec_mode == "env":
                    result = generate_narration_subprocess(app_root=_project_root, **gen_kwargs)
                elif exec_mode == "current":
                    result = generate_narration(**gen_kwargs)
                else:
                    raise NarrationError(f"Backend '{job_backend}' is not runnable in this environment.")
                out_path = Path(str(result.get("output", "")))
                print("\n[OK] Narration job completed.")
                print(f"  Backend:  {result.get('backend', job_backend)}")
                print(f"  Output:   {out_path}")
                print(f"  Job dir:  {result.get('job_dir', job_dir)}")
                print(f"  Duration: {result.get('duration_seconds', '?')}s")
                print(f"  Chunks:   {result.get('chunks', '?')} (synthesized now: {result.get('chunks_synthesized_now', '?')}, reused: {result.get('chunks_reused', '?')})")
                if result.get("backend") == "mms_fa":
                    print("  Note: MMS Persian used fixed voice for accurate Persian text (reference voice not cloned).")
                if out_path.exists() and os.name == "nt":
                    try:
                        subprocess.Popen(["explorer", "/select,", str(out_path)])
                    except Exception:
                        pass
            except NarrationError as e:
                print(f"\n[ERROR] {e}")
            except Exception as e:
                print(f"\n[ERROR] Unexpected narration failure: {e}")
            _safe_input("\nPress Enter to continue...")
            continue
        elif choice != "1":
            print("Invalid selection.")
            time.sleep(1)
            continue

        scripts = list_scripts(paths)
        voices = list_voice_samples(paths)
        if not scripts:
            print(f"\nNo scripts found in: {paths.scripts_dir}")
            print("Add a .txt script file to Scripts/ and try again.")
            _safe_input("\nPress Enter to continue...")
            continue
        if not voices:
            print(f"\nNo reference voice samples found in: {paths.voices_dir}")
            print("Add a clean 10-15s .wav/.mp3 voice sample (that you have rights to use) to Voices/ and try again.")
            _safe_input("\nPress Enter to continue...")
            continue

        # Auto-detect when files are pre-placed (hands-off when only one option exists)
        if len(scripts) == 1:
            script_path = scripts[0]
            try:
                size_kb = script_path.stat().st_size / 1024.0
                print(f"\nAuto-detected script: {script_path.name} ({size_kb:.1f} KB)")
            except Exception:
                print(f"\nAuto-detected script: {script_path.name}")
        else:
            print("\nSelect a script:")
            for i, sp in enumerate(scripts, 1):
                try:
                    size_kb = sp.stat().st_size / 1024.0
                    print(f"  {i}. {sp.name} ({size_kb:.1f} KB)")
                except Exception:
                    print(f"  {i}. {sp.name}")
            s_choice = _safe_input(f"Choose script [1-{len(scripts)}, 0 to cancel]: ").strip()
            if s_choice in ("0", ""):
                continue
            try:
                script_path = scripts[int(s_choice) - 1]
            except Exception:
                print("Invalid script selection.")
                time.sleep(1)
                continue

        if len(voices) == 1:
            voice_path = voices[0]
            try:
                size_kb = voice_path.stat().st_size / 1024.0
                print(f"Auto-detected reference voice: {voice_path.name} ({size_kb:.1f} KB)")
            except Exception:
                print(f"Auto-detected reference voice: {voice_path.name}")
        else:
            print("\nSelect a reference voice sample (use only voices you own or have explicit permission to use):")
            for i, vp in enumerate(voices, 1):
                try:
                    size_kb = vp.stat().st_size / 1024.0
                    print(f"  {i}. {vp.name} ({size_kb:.1f} KB)")
                except Exception:
                    print(f"  {i}. {vp.name}")
            v_choice = _safe_input(f"Choose voice [1-{len(voices)}, 0 to cancel]: ").strip()
            if v_choice in ("0", ""):
                continue
            try:
                voice_path = voices[int(v_choice) - 1]
            except Exception:
                print("Invalid voice selection.")
                time.sleep(1)
                continue

        selected_backend = _choose_narration_backend(script_path, caps)
        # refresh caps for execution checks
        caps = check_narration_backends(_project_root)
        any_available_now, _backend_mode_now, _backend_summary_now = _narration_backend_summary(caps)

        # Determine execution mode (and concrete backend for auto) from refreshed caps
        exec_mode = "none"
        exec_backend = selected_backend
        if selected_backend == "auto":
            try:
                script_sample = script_path.read_text(encoding="utf-8-sig", errors="ignore")[:20000]
            except Exception:
                script_sample = ""
            prefer = ["mms_fa", "f5_tts"] if detect_persian_script(script_sample) else ["f5_tts", "mms_fa"]
            for cand in prefer:
                info = caps.get(cand, {}) or {}
                if info.get("available"):
                    exec_backend = cand
                    exec_mode = str(info.get("mode", "none"))
                    break
            if exec_mode == "none":
                print("\n[ERROR] No narration backend is available for auto selection.")
                _safe_input("\nPress Enter to continue...")
                continue
        else:
            binfo = caps.get(selected_backend, {}) or {}
            if not binfo.get("available"):
                print(f"\n[ERROR] Selected backend '{selected_backend}' is not available.")
                print(f"Detail: {binfo.get('detail','unavailable')}")
                _safe_input("\nPress Enter to continue...")
                continue
            exec_mode = str(binfo.get("mode", "none"))
            if exec_mode not in ("current", "env"):
                print(f"\n[ERROR] Selected backend '{selected_backend}' is not runnable in this environment.")
                _safe_input("\nPress Enter to continue...")
                continue

        ref_text = _safe_input(
            "\nOptional: enter transcript of the reference voice clip (press Enter to skip): "
        ).strip()

        print("\nLong-lecture options (press Enter for defaults):")
        chunk_in = _safe_input("  Chunk size in characters [default 700, recommended 400-900]: ").strip()
        try:
            max_chars = int(chunk_in) if chunk_in else 700
            if max_chars < 120:
                max_chars = 120
            if max_chars > 4000:
                max_chars = 4000
        except Exception:
            max_chars = 700
        pause_in = _safe_input("  Pause between chunks in seconds [default 0.25]: ").strip()
        try:
            pause_s = float(pause_in) if pause_in else 0.25
            if pause_s < 0:
                pause_s = 0.0
            if pause_s > 5.0:
                pause_s = 5.0
        except Exception:
            pause_s = 0.25
        keep_in = _safe_input("  Keep chunk files for resume/debug? [Y/n, default Y]: ").strip().lower()
        keep_chunks = keep_in not in ("n", "no")

        print("\nRights & consent confirmation (required):")
        print("  - I confirm I own this voice sample or have explicit permission to clone/use it.")
        print("  - I will use the generated audio lawfully and in compliance with applicable rights and platform policies.")
        if selected_backend == "mms_fa":
            print("  - Note: MMS Persian will synthesize accurate Persian speech with a fixed MMS voice (not a clone of the reference voice).")
        confirm_rights = _safe_input("Type YES to confirm and continue: ").strip()
        if confirm_rights != "YES":
            print("Confirmation not provided. Narration generation cancelled.")
            _safe_input("\nPress Enter to continue...")
            continue

        print(f"\nGenerating narration (long-form job)...")
        print(f"  Script:  {script_path.name}")
        print(f"  Voice:   {voice_path.name}")
        print(f"  Backend: {selected_backend} (exec: {exec_mode}, target: {exec_backend})")
        print(f"  Chunk size: {max_chars} chars | Pause: {pause_s:.2f}s | Keep chunks: {'Yes' if keep_chunks else 'No'}")
        print("  Tip: you can interrupt and resume later from the job folder if needed.")
        try:
            gen_kwargs = dict(
                script_path=script_path,
                reference_voice_path=voice_path,
                output_dir=paths.output_dir,
                ref_text=ref_text,
                max_chars_per_chunk=max_chars,
                pause_seconds=pause_s,
                keep_chunks=keep_chunks,
                resume=True,
                progress=lambda m: print(f"  {m}"),
                backend=selected_backend,
            )
            if exec_mode == "env":
                result = generate_narration_subprocess(app_root=_project_root, **gen_kwargs)
            elif exec_mode == "current":
                result = generate_narration(**gen_kwargs)
            else:
                raise NarrationError(f"Backend '{exec_backend}' is not runnable in this environment.")
            out_path = Path(str(result.get("output", "")))
            print("\n[OK] Narration audio generated.")
            print(f"  Backend:  {result.get('backend', selected_backend)}")
            print(f"  Output:   {out_path}")
            print(f"  Job dir:  {result.get('job_dir', '')}")
            print(f"  Manifest: {result.get('manifest', '')}")
            print(f"  Duration: {result.get('duration_seconds', '?')}s")
            print(f"  Chunks:   {result.get('chunks', 1)} (synthesized now: {result.get('chunks_synthesized_now', '?')}, reused: {result.get('chunks_reused', 0)})")
            if result.get("backend") == "mms_fa":
                print("  Note: Accurate Persian content generated with fixed MMS voice. Reference voice sample was recorded but not cloned.")
            if out_path.exists() and os.name == "nt":
                try:
                    subprocess.Popen(["explorer", "/select,", str(out_path)])
                except Exception:
                    pass
        except NarrationError as e:
            print(f"\n[ERROR] {e}")
        except Exception as e:
            print(f"\n[ERROR] Unexpected narration failure: {e}")

        _safe_input("\nPress Enter to continue...")



def _apply_compute_profile(config: dict, profile: ComputeProfile) -> dict:
    """Persist machine-local ComputeProfile and set active session runtime env vars."""
    save_compute_profile(profile)
    os.environ["TW_COMPUTE_MODE"] = profile.compute_mode
    os.environ["TW_STEM_MODEL"] = profile.stem_model
    return config


def select_compute_profile_menu(
    config: dict,
    report: Optional[HardwareReport] = None,
    allow_keep: bool = True,
) -> ComputeProfile:
    """First startup window: hardware detection + CPU vs GPU Compute Profile selection."""
    ffmpeg_bin = config.get("ffmpeg_path", "ffmpeg")
    if report is None:
        report = detect_hardware(ffmpeg_bin=ffmpeg_bin, verify_encoders=True)

    saved = load_compute_profile()
    primary = report.primary_gpu()

    while True:
        clear_screen()
        print_header("TikTok Wizard v5.4 — Startup Hardware & Compute Profile")
        print(f"  Host System : {report.os_label} | RAM: {report.ram_gb or '?'} GB")
        print(f"  CPU         : {report.cpu_name} ({report.cpu_cores}C / {report.cpu_threads}T)")
        if primary:
            vram_txt = f"{primary.vram_gb:.1f} GB VRAM" if primary.vram_gb > 0 else "Shared/Unified VRAM"
            print(f"  Primary GPU : {primary.name} [{primary.vendor.upper()}, {vram_txt}]")
        else:
            print("  Primary GPU : None detected (CPU-only host)")
        print(
            f"  Encoders    : {', '.join(report.working_hw_encoders or report.ffmpeg_encoders or ['libx264'])}"
        )
        print(
            f"  ONNX/Torch  : Torch {report.torch_version} | ORT: {', '.join(report.ort_providers or ['CPUExecutionProvider'])}"
        )
        if saved:
            print(f"\n  Saved Profile: {saved.display_name} [Stems={saved.stem_mode}, Video={saved.video_codec}]")
        print()

        gpu_fast_prof = build_compute_profile(report, mode="gpu", stem_preference="auto")
        gpu_hq_prof = build_compute_profile(report, mode="gpu", stem_preference="hq")
        cpu_prof = build_compute_profile(report, mode="cpu", stem_preference="hq")

        is_win_dml = (
            primary is not None
            and primary.vendor in ("amd", "intel")
            and report.os_name == "windows"
        )

        print("Select Compute Mode before choosing a project:\n")
        if primary or report.has_capable_gpu():
            if is_win_dml:
                dml_note = "~25s Kim_Vocal_2 DirectML" if "DmlExecutionProvider" in report.ort_providers else "CPU fallback (install DirectML below for ~25s)"
                print(f"  1. GPU Fast Mode ({gpu_fast_prof.display_name})")
                print(f"     Stems: {dml_note} | Render: {gpu_fast_prof.video_codec}")
                print(f"  2. GPU Hybrid Studio Mode (BS-Roformer CPU Stems + {gpu_hq_prof.video_codec} Video)")
                print(f"     Stems: Highest-accuracy BS-Roformer CPU (~2-3.5m/min) | Render: {gpu_hq_prof.video_codec}")
                print(f"  3. CPU-Only Mode (Universal Safe — Studio BS-Roformer + libx264)")
            else:
                bs_note = "GPU CUDA/MPS/ROCm (~20s)" if gpu_hq_prof.bs_roformer_on_gpu else "CPU BS-Roformer"
                print(f"  1. GPU Accelerated Mode ({gpu_hq_prof.display_name})")
                print(f"     Stems: {bs_note} | Aligner: {gpu_hq_prof.align_device.upper()} | Render: {gpu_hq_prof.video_codec}")
                print(f"  2. GPU Fast-Stems Mode (Kim_Vocal_2 + {gpu_fast_prof.video_codec})")
                print(f"  3. CPU-Only Mode (Universal Safe — Studio BS-Roformer + libx264)")
        else:
            print("  1. CPU-Only Mode (Universal Safe — Studio BS-Roformer + libx264)")
            print("  2. CPU Fast Mode (Kim_Vocal_2 CPU Stems + libx264)")

        if gpu_fast_prof.package_gaps:
            print(f"\n  [!] GPU Package Notice: {gpu_fast_prof.package_gaps[0]}")
            print("  I. Install/Upgrade Matching GPU Acceleration Packages into venv")

        print("  R. Re-scan Hardware & Encoders")
        if allow_keep and saved is not None:
            print("  0. Continue with Saved Profile (or press Enter)")

        raw = _safe_input("\nSelect Compute Profile: ").strip().lower()
        if raw in ("", "0") and allow_keep and saved is not None:
            _apply_compute_profile(config, saved)
            return saved
        if raw == "r":
            report = detect_hardware(ffmpeg_bin=ffmpeg_bin, verify_encoders=True)
            continue
        if raw == "i" and primary:
            install_gpu_packages_for_vendor(primary.vendor, report.os_name, log_fn=print)
            report = detect_hardware(ffmpeg_bin=ffmpeg_bin, verify_encoders=True)
            _safe_input("\nPress Enter to refresh detected runtimes...")
            continue

        chosen: Optional[ComputeProfile] = None
        if primary or report.has_capable_gpu():
            if raw == "1":
                chosen = gpu_fast_prof if is_win_dml else gpu_hq_prof
            elif raw == "2":
                chosen = gpu_hq_prof if is_win_dml else build_compute_profile(report, mode="gpu", stem_preference="fast")
            elif raw == "3":
                chosen = cpu_prof
        else:
            if raw in ("", "1"):
                chosen = cpu_prof
            elif raw == "2":
                chosen = build_compute_profile(report, mode="cpu", stem_preference="fast")

        if chosen is not None:
            _apply_compute_profile(config, chosen)
            return chosen
        print("Invalid selection.")
        time.sleep(1)


def main() -> int:
    config = load_config()
    projects_dir = (_project_root / config["projects_dir"]).resolve()
    defaults_dir = (_project_root / config["defaults_dir"]).resolve()
    build_dir = (_project_root / config["build_dir"]).resolve()
    history_path = (_project_root / config["history_file"]).resolve()

    ensure_dir(build_dir)
    init_default_profiles()

    # First window: detect hardware and choose Compute Profile (CPU vs GPU Mode)
    print("Detecting host CPU/GPU hardware and encoders...")
    hw_report = detect_hardware(ffmpeg_bin=config.get("ffmpeg_path", "ffmpeg"), verify_encoders=True)
    active_cp = select_compute_profile_menu(config, report=hw_report, allow_keep=True)

    while True:
        clear_screen()
        print_header("TikTok Wizard v5.4 (Greenfield Engine v5 & Dual-Model Consensus)")
        print_wrapped(
            "Turns your song + Holy-Grail lyrics.txt into viral TikTok/Shorts karaoke videos."
        )
        print(f"  Active Compute: {active_cp.display_name} [Stems: {active_cp.stem_mode.upper()}, Video: {active_cp.video_codec}]")
        print()

        projects = find_projects(projects_dir)
        start_here_dir = (
            projects_dir.parent
            if projects_dir.name.lower() == "_projects"
            else (_project_root / "START HERE")
        )

        if not projects:
            print_wrapped(f"No project folders found in: {projects_dir}", indent="  ")
            print("\n  C. Change Compute Profile (CPU / GPU Mode)")
            print("  N. Narration Studio (voiceover from text)")
            print("  0. Exit\n")
            choice = _safe_input("Select [C, N, 0 to exit]: ").strip().lower()
            if choice == "c":
                active_cp = select_compute_profile_menu(config, report=hw_report, allow_keep=True)
            elif choice in ("n", "narration", "voiceover"):
                run_narration_studio(start_here_dir)
            elif choice in ("0", "exit", "q"):
                return 0
            continue

        print("Select a Project:\n")
        for i, p in enumerate(projects, 1):
            audio = find_audio(p)
            lyrics = find_lyrics(p)
            wc = 0
            if lyrics:
                try:
                    wc = len(lyrics.read_text(encoding="utf-8-sig").split())
                except Exception:
                    pass
            dot_data = get_project_data_dir(p)
            has_align = (dot_data / "alignment_manifest.json").exists()
            status_tag = "[Aligned]" if has_align else "[Ready]"
            print(f"  {i}. {p.name} {status_tag}")
            print(f"     Audio:  {audio.name if audio else '[MISSING]'}")
            print(f"     Lyrics: {lyrics.name if lyrics else '[MISSING]'} ({wc} words)")
        print("\n  C. Change Compute Profile (CPU / GPU Mode)")
        print("  N. Narration Studio (voiceover from text)")
        print("  0. Exit\n")

        choice = (
            _safe_input(f"Select [1-{len(projects)}, C, N, 0 to exit]: ").strip().lower()
        )
        if choice in ("0", "exit", "q"):
            return 0
        if choice == "c":
            active_cp = select_compute_profile_menu(config, report=hw_report, allow_keep=True)
            continue
        if choice in ("n", "narration", "voiceover"):
            run_narration_studio(start_here_dir)
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
        run_alignment_and_render_flow(
            project_folder, config, defaults_dir, history_path
        )


if __name__ == "__main__":
    sys.exit(main())
