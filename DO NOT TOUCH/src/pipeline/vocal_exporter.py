#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TikTok Wizard — Vocal Stem & Acapella Isolation Exporter

Exports high-fidelity isolated vocal stems directly into the user's project
directory under `START HERE/_Projects/<project>/Vocal Stem/`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional


def find_stems_dir(project_folder: Path) -> Optional[Path]:
    """Locate the internal stems directory for a project."""
    project_folder = Path(project_folder)
    # Check .data/stems in project folder
    cand1 = project_folder / ".data" / "stems"
    if cand1.exists() and (cand1 / "acapella_full.wav").exists():
        return cand1

    # Check DO NOT TOUCH/data/<project>/stems
    dnt_root = Path(__file__).resolve().parent.parent.parent
    cand2 = dnt_root / "data" / project_folder.name / "stems"
    if cand2.exists() and (cand2 / "acapella_full.wav").exists():
        return cand2

    return None


def export_vocal_stems(
    project_folder: Path,
    audio_path: Optional[Path] = None,
    ffmpeg_bin: str = "ffmpeg",
    export_mp3: bool = True,
    log_fn=print,
) -> Dict[str, Path]:
    """
    Extract or retrieve isolated vocal stems and copy them into `<project>/Vocal Stem/`.

    Exports:
    - Acapella (Full Vocals).wav
    - Acapella (Lead Vocals - Dry).wav
    - Acapella (Backing & Harmonies).wav
    - Acapella (Full Vocals).mp3 (optional 320k high-quality transcode)
    """
    project_folder = Path(project_folder)
    dest_dir = project_folder / "Vocal Stem"
    dest_dir.mkdir(parents=True, exist_ok=True)

    stems_dir = find_stems_dir(project_folder)
    stems_match_audio = True
    if stems_dir and audio_path and Path(audio_path).exists():
        info_file = stems_dir / "backend_info.json"
        if info_file.exists():
            try:
                import json
                bdata = json.loads(info_file.read_text(encoding="utf-8"))
                src_name = bdata.get("source_audio_name")
                if src_name and src_name != Path(audio_path).name:
                    stems_match_audio = False
            except Exception:
                pass

    # If stems do not exist or were generated for a different audio file, run stem builder
    if not stems_dir or not (stems_dir / "acapella_full.wav").exists() or not stems_match_audio:
        if not audio_path or not Path(audio_path).exists():
            # Try to find audio in project folder
            for ext in (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"):
                cands = sorted(project_folder.glob(f"*{ext}"), key=lambda p: p.name.lower())
                if cands:
                    audio_path = cands[0]
                    break

        if not audio_path or not Path(audio_path).exists():
            raise FileNotFoundError(f"No audio file found in project folder: {project_folder}")

        log_fn("  [Vocal Stem] Generating synchronized 16 kHz acoustic stems via UniversalEngine...")
        from src.align.pipeline import UniversalEngine
        engine = UniversalEngine()
        if not stems_dir:
            dnt_root = Path(__file__).resolve().parent.parent.parent
            dot_data = dnt_root / "data" / project_folder.name
            dot_data.mkdir(parents=True, exist_ok=True)
            stems_dir = dot_data / "stems"
        engine._build_stems(str(audio_path), stems_dir=str(stems_dir))

    stems_dir = Path(stems_dir)
    results: Dict[str, Path] = {}

    stem_mappings = [
        ("acapella_full.wav", "Acapella (Full Vocals).wav", "full_wav"),
        ("acapella_lead_dry.wav", "Acapella (Lead Vocals - Dry).wav", "lead_wav"),
        ("acapella_backing_echo.wav", "Acapella (Backing & Harmonies).wav", "backing_wav"),
        ("original_mix_vocal_band.wav", "Original Mix (Vocal Bandpass).wav", "vocal_band_wav"),
    ]

    for src_name, dst_name, key in stem_mappings:
        src_file = stems_dir / src_name
        if src_file.exists():
            dst_file = dest_dir / dst_name
            shutil.copy2(src_file, dst_file)
            results[key] = dst_file
            log_fn(f"  ✓ Exported: {dst_name} ({dst_file.stat().st_size / 1024 / 1024:.2f} MB)")

    # Generate high-bitrate MP3 version of full acapella for instant mobile/device playback
    if export_mp3 and "full_wav" in results:
        full_wav = results["full_wav"]
        full_mp3 = dest_dir / "Acapella (Full Vocals).mp3"
        try:
            cmd = [
                ffmpeg_bin,
                "-y",
                "-i", str(full_wav),
                "-codec:a", "libmp3lame",
                "-b:a", "320k",
                str(full_mp3),
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            if full_mp3.exists():
                results["full_mp3"] = full_mp3
                log_fn(f"  ✓ Transcoded: {full_mp3.name} ({full_mp3.stat().st_size / 1024 / 1024:.2f} MB @ 320kbps)")
        except Exception:
            pass

    # Write a quick info README inside Vocal Stem folder
    readme_path = dest_dir / "README.txt"
    readme_content = f"""================================================================================
TIKTOK WIZARD — ISOLATED VOCAL STEMS & ACAPELLA
================================================================================
Project: {project_folder.name}

FILES IN THIS FOLDER:
1. Acapella (Full Vocals).wav
   Complete isolated vocal track with all lead and backing harmonies.
   Perfect for remixing, mashups, and karaoke vocal isolation.

2. Acapella (Lead Vocals - Dry).wav
   Center-channel dry vocal performance stripped of room reverb, echo,
   and stereo backing vocals.

3. Acapella (Backing & Harmonies).wav
   Stereo backing vocals, ad-libs, and room acoustic reflections.

4. Acapella (Full Vocals).mp3
   Studio-quality 320kbps MP3 transcode for mobile listening and instant sharing.

5. Original Mix (Vocal Bandpass).wav
   Mid-channel harmonic voice-band frequencies filtered directly from the
   original master mix.
================================================================================
Generated by TikTok Wizard Universal Alignment Engine.
"""
    readme_path.write_text(readme_content, encoding="utf-8")

    return results
