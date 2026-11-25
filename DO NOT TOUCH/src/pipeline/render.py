#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShotCut TikTok Automator — Cross-Platform Render Pipeline (v4.1)

Builds the FFmpeg command for burning TikTok karaoke subtitles (ASS)
over looping/custom backgrounds (videos, pictures, gifs), with optional audio
spectrum/waveform visualizers, custom time-frame clipping, and cross-platform
Linux (fontconfig / fc-list), macOS, and Windows font discovery.
"""

from __future__ import annotations

import subprocess
import json
import sys
import numpy as np
from pathlib import Path
from typing import List, Optional

from .manifest_writer import write_video_render_manifest


def _escape_ffmpeg_path(path: Path) -> str:
    """Escape file paths for FFmpeg filtergraph syntax across Windows and POSIX."""
    p_str = Path(path).resolve().as_posix()
    p_str = p_str.replace("\\", "/")
    p_str = p_str.replace(":", r"\:")
    p_str = p_str.replace("'", r"\'")
    return p_str


def _find_font_dir(
    font_name: str,
    font_file: Optional[str] = None,
    defaults_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Return directory containing font files for libass fontsdir across Windows, Linux, and macOS."""
    if font_file:
        pf = Path(font_file)
        if pf.exists():
            return pf.parent if pf.is_file() else pf

    clean_target = font_name.lower().replace(" ", "").replace("-", "").replace("_", "")

    # 1. Check project _Assets/Fonts directory first
    if defaults_dir and Path(defaults_dir).exists():
        defaults_path = Path(defaults_dir)
        try:
            for f in defaults_path.rglob("*"):
                if not f.is_file():
                    continue
                if f.suffix.lower() in (".ttf", ".otf", ".ttc", ".woff2"):
                    c_cand = (
                        f.stem.lower()
                        .replace(" ", "")
                        .replace("-", "")
                        .replace("_", "")
                    )
                    if clean_target in c_cand or c_cand in clean_target:
                        return f.parent
        except Exception:
            pass

    # 2. Check local script/project assets
    local_fonts = (
        Path(__file__).resolve().parent.parent.parent
        / "START HERE"
        / "_Projects"
        / "_Assets"
        / "Fonts"
    )
    if local_fonts.exists():
        try:
            for f in local_fonts.rglob("*"):
                if not f.is_file():
                    continue
                if f.suffix.lower() in (".ttf", ".otf", ".ttc", ".woff2"):
                    c_cand = (
                        f.stem.lower()
                        .replace(" ", "")
                        .replace("-", "")
                        .replace("_", "")
                    )
                    if clean_target in c_cand or c_cand in clean_target:
                        return f.parent
        except Exception:
            pass

    # 3. Check OS-specific standard fonts directories
    system_font_dirs = []
    if sys.platform == "win32":
        import os

        system_font_dirs.extend(
            [
                Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts",
                Path(os.environ.get("LOCALAPPDATA", ""))
                / "Microsoft"
                / "Windows"
                / "Fonts",
            ]
        )
    elif sys.platform == "darwin":
        system_font_dirs.extend(
            [
                Path("/Library/Fonts"),
                Path("/System/Library/Fonts"),
                Path.home() / "Library" / "Fonts",
            ]
        )
    else:  # Linux / POSIX
        system_font_dirs.extend(
            [
                Path("/usr/share/fonts"),
                Path("/usr/local/share/fonts"),
                Path.home() / ".fonts",
                Path.home() / ".local" / "share" / "fonts",
            ]
        )

    for font_dir in system_font_dirs:
        if not font_dir.exists():
            continue
        try:
            for f in font_dir.rglob("*"):
                if not f.is_file():
                    continue
                if f.suffix.lower() in (".ttf", ".otf", ".ttc"):
                    c_cand = (
                        f.stem.lower()
                        .replace(" ", "")
                        .replace("-", "")
                        .replace("_", "")
                    )
                    if clean_target in c_cand or c_cand in clean_target:
                        return f.parent
        except Exception:
            pass

    return None


def find_default_background(
    defaults_dir: Path, preferred_file: Optional[str] = None
) -> Optional[Path]:
    """
    Search _Assets/Backgrounds directory for videos, gifs, or images.
    If preferred_file is specified (from active profile), look for that specific filename first.
    """
    if not defaults_dir.exists():
        return None

    # If profile specifies a preferred background file, check if it exists
    if preferred_file:
        pref_cand = defaults_dir / preferred_file
        if pref_cand.exists():
            return pref_cand
        pref_bg_cand = defaults_dir / "Backgrounds" / "Videos" / preferred_file
        if pref_bg_cand.exists():
            return pref_bg_cand
        for f in defaults_dir.rglob(preferred_file):
            if f.is_file():
                return f

    video_exts = {".mp4", ".mov", ".mkv", ".webm"}
    gif_exts = {".gif"}
    img_exts = {".jpg", ".jpeg", ".png", ".webp"}

    # 1. Look for preferred named video files in _Assets
    for ext in video_exts:
        for name in ("default", "background", "loop", "bg"):
            cand = defaults_dir / f"{name}{ext}"
            if cand.is_file():
                return cand
            cand_sub = defaults_dir / "Backgrounds" / f"{name}{ext}"
            if cand_sub.is_file():
                return cand_sub
            cand_vid = defaults_dir / "Backgrounds" / "Videos" / f"{name}{ext}"
            if cand_vid.is_file():
                return cand_vid

    # 2. Check for any video files
    for f in sorted(defaults_dir.rglob("*")):
        if f.is_file() and f.suffix.lower() in video_exts:
            return f

    # 3. Check for any GIF files
    for f in sorted(defaults_dir.rglob("*")):
        if f.is_file() and f.suffix.lower() in gif_exts:
            return f

    # 4. Check for any picture files
    for f in sorted(defaults_dir.rglob("*")):
        if f.is_file() and f.suffix.lower() in img_exts:
            return f

    return None


def _build_video_encoder_args(vcodec: str, crf: int) -> List[str]:
    """Build hardware/software H.264 quality & preset flags compatible with each FFmpeg encoder."""
    vc = (vcodec or "libx264").strip().lower()
    q = int(crf)
    if "_qsv" in vc:
        return ["-global_quality", str(q), "-preset", "fast"]
    if "_nvenc" in vc:
        return ["-rc", "vbr", "-cq", str(q), "-b:v", "0", "-preset", "fast"]
    if "_amf" in vc:
        return ["-rc", "cqp", "-qp_i", str(q), "-qp_p", str(q), "-quality", "balanced"]
    if "_videotoolbox" in vc:
        vt_q = max(30, min(85, 100 - q * 2))
        return ["-q:v", str(vt_q), "-allow_sw", "1"]
    if "_vaapi" in vc:
        return ["-qp", str(q)]
    return ["-crf", str(q), "-preset", "fast"]


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
    ffmpeg_bin = config.get("ffmpeg_path", "ffmpeg")
    w = int(config.get("video_width", 1080))
    h = int(config.get("video_height", 1920))
    fps = int(config.get("fps", 30))

    # Base audio inputs
    inputs: List[str] = [
        "-ss",
        f"{start_time:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(audio_path.resolve()),
    ]

    filter_chains: List[str] = []

    # 1. Background Layer Setup
    bg_cfg = effect_cfg.get("background", {})
    fit_mode = bg_cfg.get("fit_mode", "blur_pad")

    if background_path and background_path.exists():
        ext = background_path.suffix.lower()
        is_video = ext in {".mp4", ".mov", ".mkv", ".webm"}
        is_image = ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        is_gif = ext == ".gif"

        if is_image:
            inputs.extend(
                [
                    "-loop",
                    "1",
                    "-t",
                    f"{duration:.3f}",
                    "-i",
                    str(background_path.resolve()),
                ]
            )
        elif is_video:
            inputs.extend(
                [
                    "-stream_loop",
                    "-1",
                    "-t",
                    f"{duration:.3f}",
                    "-i",
                    str(background_path.resolve()),
                ]
            )
        elif is_gif:
            inputs.extend(
                [
                    "-ignore_loop",
                    "0",
                    "-t",
                    f"{duration:.3f}",
                    "-i",
                    str(background_path.resolve()),
                ]
            )
        else:
            inputs.extend(
                ["-t", f"{duration:.3f}", "-i", str(background_path.resolve())]
            )

        bg_input = "[1:v]"

        if fit_mode == "blur_pad":
            # Viral TikTok Aesthetic: Scaled blurred video in background, sharp centered 1:1/horizontal original in foreground
            filter_chains.append(
                f"{bg_input}split=2[bg_raw][fg_raw];"
                f"[bg_raw]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},gblur=sigma=35[bg_blurred];"
                f"[fg_raw]scale={w}:{h}:force_original_aspect_ratio=decrease[fg_sharp];"
                f"[bg_blurred][fg_sharp]overlay=(W-w)/2:(H-h)/2[v_bg]"
            )
        elif fit_mode == "crop_fill":
            filter_chains.append(
                f"{bg_input}scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}[v_bg]"
            )
        elif fit_mode == "dark_pad":
            filter_chains.append(
                f"{bg_input}scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black@0.9[v_bg]"
            )
        elif fit_mode == "mirror_pad":
            filter_chains.append(
                f"{bg_input}split=2[orig][mirr];"
                f"[mirr]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur=20:5[mbg];"
                f"[orig]scale={w}:{h}:force_original_aspect_ratio=decrease[mfg];"
                f"[mbg][mfg]overlay=(W-w)/2:(H-h)/2[v_bg]"
            )
        else:  # Standard letterbox pad
            filter_chains.append(
                f"{bg_input}scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black[v_bg]"
            )
    else:
        # Default solid gradient dark background
        filter_chains.append(
            f"color=c=#0f0c1b:s={w}x{h}:d={duration:.3f}:r={fps}[v_bg]"
        )

    current_vid = "v_bg"

    # 1.5 Atmospheric & Particle Effects (TikTok Viral FX Layer)
    part_cfg = effect_cfg.get("particles", {})
    part_type = part_cfg.get("type", "none")
    if part_type != "none":
        from src.pipeline.particles import generate_particle_overlay

        tmp_particles_mov = (
            output_path.parent / f".tmp_particles_{output_path.stem}.mov"
        )
        try:
            overlay_file = generate_particle_overlay(
                particle_cfg=part_cfg,
                audio_path=audio_path,
                output_mov_path=tmp_particles_mov,
                width=w,
                height=h,
                fps=fps,
                duration=duration,
                start_time=start_time,
                ffmpeg_bin=ffmpeg_bin,
                defaults_dir=defaults_dir,
            )
            if overlay_file and Path(overlay_file).exists():
                num_i_p = inputs.count("-i")
                inputs.extend(
                    ["-stream_loop", "-1", "-i", str(Path(overlay_file).resolve())]
                )
                filter_chains.append(
                    f"[{current_vid}][{num_i_p}:v]overlay=0:0:shortest=0[v_particles]"
                )
                current_vid = "v_particles"
        except Exception as e:
            print(f"  [Warning] Particle generation bypassed: {e}")

    # 2. Visualizer + TikTok Safety Pad layout
    # Pad is always directly underneath the visualizer; handle sits under visualizer and above pad.
    pad_cfg = effect_cfg.get("tiktok_pad", {})
    viz_cfg = effect_cfg.get("visualizer", {})
    viz_type = viz_cfg.get("type", "none")
    viz_color = viz_cfg.get("color", "#00ffff")
    viz_h_pct = float(viz_cfg.get("height_pct", 12))
    viz_h = max(20, int(h * (viz_h_pct / 100.0)))
    viz_pos = viz_cfg.get("position", "above_pad")
    viz_y_offset_pct = float(viz_cfg.get("y_offset_pct", -3.0))
    viz_y_offset_px = int(round(h * (viz_y_offset_pct / 100.0)))

    pad_h_pct = float(pad_cfg.get("height_pct", 12.5))
    pad_h_cfg = int(h * (pad_h_pct / 100.0))
    base_pad_y = h - pad_h_cfg

    viz_active = viz_type not in ("none", "disabled", "off", "")
    if viz_active:
        if viz_pos == "above_pad":
            viz_y = max(0, base_pad_y + viz_y_offset_px - viz_h)
        elif viz_pos == "below_lyrics":
            lyrics_y_pct = float(effect_cfg.get("lyrics", {}).get("position_y_percent", 65))
            viz_y = int(h * (lyrics_y_pct / 100.0)) + 60
        elif viz_pos == "center":
            viz_y = (h - viz_h) // 2
        elif viz_pos == "top":
            viz_y = 60
        elif viz_pos == "bottom":
            viz_y = h - viz_h
        else:
            viz_y = max(0, base_pad_y + viz_y_offset_px - viz_h)
        viz_y = max(0, min(viz_y, h - viz_h))
        viz_bottom = viz_y + viz_h
        pad_y = min(max(0, viz_bottom), h)
        pad_h = max(0, h - pad_y)
        pad_top_y = pad_y
    else:
        viz_y = max(0, base_pad_y - viz_h)
        viz_bottom = viz_y
        pad_y = base_pad_y
        pad_h = pad_h_cfg
        pad_top_y = pad_y

    if pad_cfg.get("enabled", True) and pad_h > 0:
        pad_opacity = float(pad_cfg.get("opacity", 0.90))
        filter_chains.append(
            f"[{current_vid}]drawbox=x=0:y={pad_y}:w={w}:h={pad_h}:color=black@{pad_opacity:.2f}:t=fill[v_padded]"
        )
        current_vid = "v_padded"

    # 3. Audio Spectrum / Waveform Visualizer Layer

    # Modern 2026 Studio-Grade Procedural Visualizers
    modern_styles = {
        "spectrum_bars",
        "radial_bass",
        "stereo_mirror",
        "particle_dots",
        "spectrum_wave",
        "radial_ring",
        "volt_radial",
        "segmented_bars",
        "peak_dots",
        "dual_wave",
        "radial_rays",
        "polygon_mesh",
        "crosshair_bars",
        "center_flower",
        "starburst",
        "sound_pillows",
        "curved_ribbon",
    }

    if viz_type in modern_styles:
        from src.visualizers.audio import analyze, AnalysisCfg
        from src.visualizers.config import VisualizerCfg
        from src.visualizers.layers.visualizer import Visualizer
        import hashlib
        import shutil

        style_map = {
            "spectrum_bars": "bars",
            "radial_bass": "circle",
            "stereo_mirror": "mirror",
            "particle_dots": "dots",
            "spectrum_wave": "wave",
            "radial_ring": "ring",
            "volt_radial": "center_flower",
            "segmented_bars": "segmented_bars",
            "peak_dots": "peak_dots",
            "dual_wave": "dual_wave",
            "radial_rays": "radial_rays",
            "polygon_mesh": "polygon_mesh",
            "crosshair_bars": "crosshair_bars",
            "center_flower": "center_flower",
            "starburst": "starburst",
            "sound_pillows": "sound_pillows",
            "curved_ribbon": "curved_ribbon",
        }
        style = style_map.get(viz_type, "bars")
        col2 = viz_cfg.get("color2", "#ff007f")
        raw_amp = float(viz_cfg.get("amplitude", 160.0))
        amp = raw_amp * 100.0 if raw_amp < 10.0 else raw_amp
        baseline = float(viz_y + viz_h) / float(h)
        raw_radius = float(
            viz_cfg.get("radius", 180.0 if style in ("circle", "ring", "dots") else 0.0)
        )
        v_radius = raw_radius * 1080.0 if 0.0 < raw_radius < 1.0 else raw_radius

        if viz_pos == "center":
            v_baseline = 0.5
        elif viz_pos == "above_pad":
            v_baseline = pad_top_y / float(h)
        else:
            v_baseline = baseline

        tmp_mov = output_path.parent / f".tmp_viz_{output_path.stem}.mov"
        # Persistent visualizer cache (per audio segment + settings) to avoid full re-render
        viz_cache_hit = False
        cached_viz = None
        try:
            ap = audio_path.resolve()
            st = ap.stat()
            viz_key_obj = {
                "v": 1,
                "audio": str(ap),
                "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
                "size": int(st.st_size),
                "start": round(float(start_time), 3),
                "dur": round(float(duration), 3),
                "fps": int(fps),
                "w": int(w),
                "h": int(h),
                "viz_type": viz_type,
                "style": style,
                "color": viz_color,
                "color2": col2,
                "amp": round(float(amp), 3),
                "radius": round(float(v_radius), 3),
                "baseline": round(float(v_baseline), 6),
                "viz_y": int(viz_y),
                "viz_h": int(viz_h),
                "pos": viz_pos,
                "thickness": 3.5,
                "gap": 2.0,
                "glow": 15.0,
                "glow_strength": 1.6,
            }
            viz_hash = hashlib.sha1(json.dumps(viz_key_obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]
            viz_cache_dir = Path(__file__).resolve().parent.parent.parent / "data" / "viz_cache"
            viz_cache_dir.mkdir(parents=True, exist_ok=True)
            cached_viz = viz_cache_dir / f"viz_{style}_{viz_hash}.mov"
            if cached_viz.exists() and cached_viz.stat().st_size > 1000:
                shutil.copy2(cached_viz, tmp_mov)
                viz_cache_hit = True
                print(f"  [Visualizer] Instant reuse: {cached_viz.name} (cache hit)")
        except Exception:
            viz_cache_hit = False

        if not viz_cache_hit:
            cmd_a = [
                ffmpeg_bin,
                "-y",
                *(["-ss", f"{start_time:.3f}"] if start_time > 0 else []),
                "-t",
                f"{duration:.3f}",
                "-i",
                str(audio_path.resolve()),
                "-ac",
                "1",
                "-ar",
                "44100",
                "-f",
                "f32le",
                "-",
            ]
            p_a = subprocess.run(cmd_a, capture_output=True, check=True)
            audio_mono = np.frombuffer(p_a.stdout, dtype=np.float32)

            an_cfg = AnalysisCfg(fmin=30.0, fmax=14000.0, n_bands=64, tilt=0.7)
            analysis = analyze(audio_mono, sample_rate=44100, fps=fps, cfg=an_cfg)

            v_cfg = VisualizerCfg(
                style=style,
                color=viz_color,
                color2=col2,
                radius=v_radius,
                amplitude=amp,
                thickness=3.5,
                gap=2.0,
                glow=15.0,
                glow_strength=1.6,
                center=(0.5, v_baseline),
                baseline=v_baseline,
            )
            viz = Visualizer(v_cfg, size=(w, h), analysis=analysis)

            cmd_enc = [
                ffmpeg_bin,
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgba",
                "-s",
                f"{w}x{h}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-c:v",
                "qtrle",
                str(tmp_mov),
            ]
            p_enc = subprocess.Popen(
                cmd_enc, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
            n_frames_viz = len(analysis)
            report_step = max(30, n_frames_viz // 20)
            for f_i in range(n_frames_viz):
                if f_i % report_step == 0 or f_i == n_frames_viz - 1:
                    pct = int(round((f_i + 1) / n_frames_viz * 100))
                    print(
                        f"  [Visualizer] Rendering {viz_type} frames: {f_i + 1}/{n_frames_viz} ({pct}%)...",
                        end="\r",
                        flush=True,
                    )
                rendered = viz.render(analysis, f_i)
                full_frame = np.zeros((h, w, 4), dtype=np.uint8)
                if rendered is not None:
                    img_patch, (px, py) = rendered
                    pw, ph = img_patch.size
                    full_frame[py : py + ph, px : px + pw] = np.array(img_patch)
                p_enc.stdin.write(full_frame.tobytes())
            p_enc.stdin.close()
            p_enc.wait()
            print(
                f"  [Visualizer] Rendering {viz_type} frames: {n_frames_viz}/{n_frames_viz} (100%) - Done!       "
            )
            try:
                if cached_viz and tmp_mov.exists() and tmp_mov.stat().st_size > 1000:
                    shutil.copy2(tmp_mov, cached_viz)
            except Exception:
                pass

        num_i = inputs.count("-i")
        inputs.extend(["-i", str(tmp_mov.resolve())])
        filter_chains.append(f"[{current_vid}][{num_i}:v]overlay=0:0[v_viz]")
        current_vid = "v_viz"
    elif viz_type == "audio_waveform":
        filter_chains.append(
            f"[0:a]showwaves=s={w}x{viz_h}:mode=line:colors={viz_color}[wave]"
        )
        filter_chains.append(
            f"[{current_vid}][wave]overlay=0:{viz_y}:format=auto[v_viz]"
        )
        current_vid = "v_viz"
    elif viz_type in ("audio_bars", "neon_glow_bars"):
        filter_chains.append(
            f"[0:a]showfreqs=s={w}x{viz_h}:mode=bar:ascale=log:fscale=log:colors={viz_color}[bars]"
        )
        filter_chains.append(
            f"[{current_vid}][bars]overlay=0:{viz_y}:format=auto[v_viz]"
        )
        current_vid = "v_viz"
    elif viz_type in ("stereo_wave", "p2p_waveform"):
        filter_chains.append(
            f"[0:a]showwaves=s={w}x{viz_h}:mode=p2p:colors={viz_color}:scale=lin[waves]"
        )
        filter_chains.append(
            f"[{current_vid}][waves]overlay=0:{viz_y}:format=auto[v_viz]"
        )
        current_vid = "v_viz"
    elif viz_type in ("minimal_line", "cline_waveform"):
        filter_chains.append(
            f"[0:a]showwaves=s={w}x{viz_h}:mode=cline:colors={viz_color}[cline]"
        )
        filter_chains.append(
            f"[{current_vid}][cline]overlay=0:{viz_y}:format=auto[v_viz]"
        )
        current_vid = "v_viz"

    # 4. Brand Stamp & Creator Handles
    _brand_active = True
    handle_cfg = effect_cfg.get("creator_handle", {})
    overlay_text_filters: List[str] = []

    # Mandatory runtime brand overlay (obfuscated payload/placement)
    if _brand_active:
        _wm_sig = bytes(
            [
                26,
                46,
                51,
                49,
                46,
                53,
                49,
                45,
                51,
                32,
                59,
                40,
                62,
                116,
                53,
                60,
                60,
                51,
                57,
                51,
                59,
                54,
                122,
                122,
                38,
                122,
                122,
                46,
                51,
                49,
                46,
                53,
                49,
                45,
                51,
                32,
                59,
                40,
                62,
                116,
                57,
                53,
                55,
            ]
        )
        wm_text = bytes([b ^ 0x5A for b in _wm_sig]).decode("utf-8")
        wm_font_size = max(20, int(h * 0.016))
        _wm_pos = bytes(
            [
                b ^ 0x5A
                for b in bytes(
                    [
                        34,
                        103,
                        114,
                        45,
                        119,
                        46,
                        63,
                        34,
                        46,
                        5,
                        45,
                        115,
                        117,
                        104,
                        96,
                        35,
                        103,
                        105,
                        104,
                    ]
                )
            ]
        ).decode("utf-8")
        _wm_fc = bytes(
            [b ^ 0x5A for b in bytes([35, 63, 54, 54, 53, 45, 26, 106, 116, 98])]
        ).decode("utf-8")
        _wm_sh = bytes(
            [b ^ 0x5A for b in bytes([56, 54, 59, 57, 49, 26, 106, 116, 109, 104])]
        ).decode("utf-8")
        _wm_shxy = bytes(
            [
                b ^ 0x5A
                for b in bytes(
                    [
                        96,
                        41,
                        50,
                        59,
                        62,
                        53,
                        45,
                        34,
                        103,
                        104,
                        96,
                        41,
                        50,
                        59,
                        62,
                        53,
                        45,
                        35,
                        103,
                        104,
                    ]
                )
            ]
        ).decode("utf-8")
        overlay_text_filters.append(
            f"drawtext=text='{wm_text}':{_wm_pos}:fontsize={wm_font_size}:"
            f"fontcolor={_wm_fc}:shadowcolor={_wm_sh}{_wm_shxy}"
        )

    # Custom creator handles (Top LCR, Middle LCR, Bottom LCR)
    handle_text = str(handle_cfg.get("text", "")).strip()
    if handle_text:
        h_pos = handle_cfg.get("position", "bottom_right")
        h_font_size = int(handle_cfg.get("font_size", max(24, int(h * 0.022))))
        h_alpha = float(handle_cfg.get("opacity", 0.85))
        h_font = str(handle_cfg.get("font", "Montserrat"))
        h_color_raw = handle_cfg.get("color", "white")
        h_outline_color_raw = handle_cfg.get("outline_color", "black")
        h_outline_w = int(handle_cfg.get("outline_width", 2))

        # Convert hex / ASS colors if necessary
        def _to_ffmpeg_color(col: str) -> str:
            if not col:
                return "white"
            col = str(col).strip()
            if col.startswith("&H") and col.endswith("&"):
                val = col[2:-1]
                if len(val) >= 6:
                    b = val[-6:-4]
                    g = val[-4:-2]
                    r = val[-2:]
                    return f"0x{r}{g}{b}"
            if col.startswith("#"):
                return f"0x{col[1:]}"
            return col

        col_str = _to_ffmpeg_color(h_color_raw)
        out_col_str = _to_ffmpeg_color(h_outline_color_raw)

        # Find font directory if present
        h_font_dir = _find_font_dir(h_font, defaults_dir=defaults_dir)
        h_font_file_opt = ""
        if h_font_dir and h_font_dir.exists():
            for f_cand in h_font_dir.glob(f"*{h_font}*"):
                if f_cand.suffix.lower() in (".ttf", ".otf", ".woff2"):
                    esc_f = _escape_ffmpeg_path(f_cand)
                    h_font_file_opt = f":fontfile='{esc_f}'"
                    break

        # Handle: underneath the visualizer and above the black pad (bottom-right by default)
        x_expr = "w-text_w-40"
        if viz_active:
            handle_y = viz_bottom + max(6, int(h * 0.004))
        else:
            handle_y = pad_top_y - max(24, int(h * 0.02))
        y_expr = str(max(0, min(int(handle_y), h - 8)))
        coord_expr = f"x={x_expr}:y={y_expr}"

        border_opt = (
            f":borderw={h_outline_w}:bordercolor={out_col_str}@{h_alpha:.2f}"
            if h_outline_w > 0
            else ":shadowcolor=black@0.8:shadowx=2:shadowy=2"
        )

        overlay_text_filters.append(
            f"drawtext=text='{handle_text}'{h_font_file_opt}:{coord_expr}:fontsize={h_font_size}:"
            f"fontcolor={col_str}@{h_alpha:.2f}{border_opt}"
        )

    if overlay_text_filters:
        combined_text = ",".join(overlay_text_filters)
        filter_chains.append(f"[{current_vid}]{combined_text}[v_overlays]")
        current_vid = "v_overlays"

    # 5. ASS Subtitle Burn-In (Top-most layer)
    escaped_ass = _escape_ffmpeg_path(ass_path)
    lyrics_font = effect_cfg.get("lyrics", {}).get(
        "font", config.get("karaoke", {}).get("font", "Chango")
    )
    resolved_font_dir = _find_font_dir(
        lyrics_font, config.get("karaoke", {}).get("font_file"), defaults_dir
    )

    fontsdir_opt = ""
    if resolved_font_dir and resolved_font_dir.exists():
        escaped_font_dir = _escape_ffmpeg_path(resolved_font_dir)
        fontsdir_opt = f":fontsdir='{escaped_font_dir}'"

    if float(start_time) > 0.0:
        pre = f"[{current_vid}]setpts=PTS-STARTPTS+{float(start_time):.3f}/TB[v_assin]"
        post = f"[v_ass]setpts=PTS-STARTPTS[final]"
        ass_filter = f"[v_assin]ass='{escaped_ass}'{fontsdir_opt}[v_ass]"
        filter_chains.append(pre)
        filter_chains.append(ass_filter)
        filter_chains.append(post)
    else:
        ass_filter = f"[{current_vid}]ass='{escaped_ass}'{fontsdir_opt}[final]"
        filter_chains.append(ass_filter)

    filter_complex_str = ";".join(filter_chains)

    vcodec = str(config.get("codec", "h264"))
    quality_args = _build_video_encoder_args(vcodec, int(config.get("crf", 22)))

    cmd = [
        ffmpeg_bin,
        "-y",
        *inputs,
        "-filter_complex",
        filter_complex_str,
        "-map",
        "[final]",
        "-map",
        "0:a",
        "-c:v",
        vcodec,
        *quality_args,
        "-r",
        str(fps),
        "-c:a",
        str(config.get("audio_codec", "aac")),
        "-b:a",
        "192k",
        "-t",
        f"{duration:.3f}",
        str(output_path.resolve()),
    ]

    return cmd
