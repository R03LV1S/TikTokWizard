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
import sys
import numpy as np
from pathlib import Path
from typing import List, Optional


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
                if f.suffix.lower() in (".ttf", ".otf", ".ttc"):
                    f_clean = f.stem.lower().replace(" ", "").replace("-", "").replace("_", "")
                    if clean_target in f_clean or f_clean in clean_target:
                        return f.parent
        except Exception:
            pass
        fonts_sub = defaults_path / "Fonts"
        if fonts_sub.exists():
            return fonts_sub

    # 2. Linux fontconfig (`fc-list`) lookup
    if sys.platform.startswith("linux"):
        try:
            proc = subprocess.run(
                ["fc-list", f":family={font_name}", "file"],
                capture_output=True,
                text=True,
                timeout=3,
                encoding="utf-8",
                errors="ignore",
            )
            for line in proc.stdout.splitlines():
                candidate = line.split(":")[0].strip()
                if candidate and Path(candidate).is_file():
                    return Path(candidate).parent
        except Exception:
            pass

        for linux_dir in (
            Path.home() / ".local/share/fonts",
            Path("/usr/local/share/fonts"),
            Path("/usr/share/fonts"),
        ):
            if linux_dir.exists():
                return linux_dir

    # 3. macOS system font directories
    if sys.platform == "darwin":
        for mac_dir in (
            Path.home() / "Library/Fonts",
            Path("/Library/Fonts"),
            Path("/System/Library/Fonts"),
        ):
            if mac_dir.exists():
                return mac_dir

    # 4. Windows Fonts directory
    win_fonts = Path(r"C:\Windows\Fonts")
    if win_fonts.exists():
        return win_fonts

    return None


def find_default_background(
    defaults_dir: Path, preferred_file: Optional[str] = None
) -> Optional[Path]:
    """Find background asset strictly from _Assets/Backgrounds/."""
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

    for ext in (".mp4", ".mov", ".mkv", ".webm", ".jpg", ".jpeg", ".png", ".gif", ".webp"):
        matches = list(defaults_dir.rglob(f"default{ext}"))
        if matches:
            return matches[0]

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
    """Build FFmpeg command for vertical/custom TikTok karaoke video render."""
    audio_path = Path(audio_path)
    ass_path = Path(ass_path)
    output_path = Path(output_path)
    defaults_dir = Path(defaults_dir)

    w = int(config.get("video_width", 1080))
    h = int(config.get("video_height", 1920))
    fps = int(config.get("fps", 30))
    ffmpeg_bin = str(config.get("ffmpeg_path", "ffmpeg"))

    visualizer = effect_cfg.get("visualizer", {"type": "none", "color": "#00ffff", "height_pct": 12})
    bg_cfg = effect_cfg.get("background", {})
    fit_mode = bg_cfg.get("fit_mode", "blur_pad")

    inputs: List[str] = []
    if start_time > 0:
        inputs.extend(["-ss", f"{start_time:.3f}"])
    inputs.extend(["-t", f"{duration:.3f}", "-i", str(audio_path.resolve())])

    found_bg = False
    resolved_bg = (
        background_path
        if background_path and Path(background_path).exists()
        else find_default_background(defaults_dir)
    )

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
            filter_chains.append(
                f"[1:v]split=2[bg_src][fg_src];"
                f"[bg_src]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur=25:5[bg_blur];"
                f"[fg_src]scale={w}:{h}:force_original_aspect_ratio=decrease[fg_sharp];"
                f"[bg_blur][fg_sharp]overlay=(W-w)/2:(H-h)/2,setsar=1,fps={fps}[bg]"
            )
    else:
        color = bg_cfg.get("colors", ["#0a0a0a"])[0] if bg_cfg.get("colors") else "#0a0a0a"
        filter_chains.append(f"color=c={color}:s={w}x{h}:r={fps}:d={duration}[bg]")

    current_vid = "bg"

    # 2. TikTok Safety Dark Pad (Placed directly over background, behind visualizer)
    # Mobile TikTok UI (captions, tags, account handle, audio pill) covers the bottom ~26-30%
    pad_cfg = effect_cfg.get("tiktok_pad", {})
    pad_enabled = pad_cfg.get("enabled", True)
    pad_opacity = float(pad_cfg.get("opacity", 0.90))
    pad_height_pct = float(pad_cfg.get("height_pct", 28.0))
    pad_h_px = int(h * (pad_height_pct / 100.0))
    pad_top_y = h - pad_h_px

    if pad_enabled:
        filter_chains.append(
            f"[{current_vid}]drawbox=x=0:y={pad_top_y}:w={w}:h={pad_h_px}:color=black@{pad_opacity:.2f}:t=fill[v_pad]"
        )
        current_vid = "v_pad"

    # 3. Visualizer Overlay (In front of background & pad, behind lyrics)
    # The visualizer bottom aligns exactly above the TikTok pad boundary (viz_y + viz_h = pad_top_y)
    viz_type = visualizer.get("type", "none")
    viz_color = visualizer.get("color", "#00ffff")
    viz_h_pct = float(visualizer.get("height_pct", 12))
    viz_h = max(60, int(h * viz_h_pct / 100.0))
    viz_pos = visualizer.get("position", "above_pad")

    lyrics_y_pct = float(effect_cfg.get("lyrics", {}).get("position_y_percent", 65))
    lyrics_y_px = int(h * lyrics_y_pct / 100.0)

    if viz_pos == "top":
        viz_y = int(h * 0.12)
    elif viz_pos == "center":
        viz_y = int((h - viz_h) / 2)
    elif viz_pos == "bottom":
        viz_y = int(h - viz_h - 40)
    elif viz_pos == "below_lyrics":
        viz_y = min(h - viz_h - 40, lyrics_y_px + 80)
    elif viz_pos == "above_pad":
        # Exactly above the top edge of the pad
        viz_y = max(40, pad_top_y - viz_h)
    else:
        viz_y = max(40, pad_top_y - viz_h)

    MODERN_VIZ_MAP = {
        "spectrum_bars": ("bars", 0.71, 160),
        "radial_bass": ("circle", 0.65, 160),
        "stereo_mirror": ("mirror", 0.71, 140),
        "particle_dots": ("dots", 0.65, 150),
        "spectrum_wave": ("wave", 0.71, 130),
        "radial_ring": ("ring", 0.65, 150),
    }

    if viz_type == "volt_radial":
        import os
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
        import pygame
        from scipy.interpolate import interp1d
        from src.visualizers.audio import analyze, AnalysisCfg
        from src.visualizers.config import VisualizerCfg
        from src.visualizers.layers.visualizer import Visualizer

        pygame.init()
        cmd_a = [
            ffmpeg_bin, "-y",
            *(["-ss", f"{start_time:.3f}"] if start_time > 0 else []),
            "-t", f"{duration:.3f}",
            "-i", str(audio_path.resolve()),
            "-ac", "1", "-ar", "44100", "-f", "f32le", "-"
        ]
        p_a = subprocess.run(cmd_a, capture_output=True, check=True)
        audio_mono = np.frombuffer(p_a.stdout, dtype=np.float32)

        an_cfg = AnalysisCfg(fmin=30.0, fmax=14000.0, n_bands=64, tilt=0.7)
        analysis = analyze(audio_mono, sample_rate=44100, fps=fps, cfg=an_cfg)
        viz = Visualizer(VisualizerCfg(style="bars"), size=(w, h), analysis=analysis)

        n_bars = 256
        half_bars = n_bars // 2
        x_64 = np.linspace(0, 1, 64)
        x_half = np.linspace(0, 1, half_bars)
        angles = (np.arange(n_bars) / n_bars) * 2 * np.pi - np.pi / 2
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)

        c1 = np.array([0, 255, 255])
        c2 = np.array([255, 0, 128])
        i_arr = np.abs(np.arange(n_bars) - half_bars) / half_bars
        bar_colors = (c1 + (c2 - c1) * i_arr[:, np.newaxis]).astype(int)
        width_factors = 1 - np.abs(np.arange(n_bars) - half_bars) / half_bars
        bar_widths = np.maximum(2, np.round(width_factors * 4)).astype(int)

        cx, cy = w // 2, int(h * 0.65)
        base_radius = 135
        max_amp = 180.0

        tmp_mov = output_path.parent / f".tmp_viz_{output_path.stem}.mov"
        cmd_enc = [
            ffmpeg_bin, "-y",
            "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r", str(fps),
            "-i", "-",
            "-c:v", "qtrle",
            str(tmp_mov)
        ]
        p_enc = subprocess.Popen(cmd_enc, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        surf = pygame.Surface((w, h), pygame.SRCALPHA)

        for f_i in range(len(analysis)):
            lvl = viz._levels(analysis, f_i)
            interp_fn = interp1d(x_64, lvl, kind="linear")
            half_curve = np.clip(interp_fn(x_half), 0.0, 1.0)
            circle_levels = np.concatenate([half_curve, half_curve[::-1]])

            beat_pulse = float(analysis.beat[f_i]) * 15.0
            r_inner = base_radius + beat_pulse
            end_radii = r_inner + circle_levels * max_amp

            start_x = cx + r_inner * cos_angles
            start_y = cy + r_inner * sin_angles
            end_x = cx + end_radii * cos_angles
            end_y = cy + end_radii * sin_angles

            surf.fill((0, 0, 0, 0))
            for b in range(n_bars):
                sp = (int(start_x[b]), int(start_y[b]))
                ep = (int(end_x[b]), int(end_y[b]))
                col = (*bar_colors[b], 230)
                pygame.draw.line(surf, col, sp, ep, int(bar_widths[b]))

            frame_bytes = pygame.image.tostring(surf, "RGBA")
            p_enc.stdin.write(frame_bytes)

        p_enc.stdin.close()
        p_enc.wait()

        num_i = inputs.count("-i")
        inputs.extend(["-i", str(tmp_mov.resolve())])
        filter_chains.append(f"[{current_vid}][{num_i}:v]overlay=0:0[v_viz]")
        current_vid = "v_viz"
    elif viz_type in MODERN_VIZ_MAP:
        from src.visualizers.audio import analyze, AnalysisCfg
        from src.visualizers.config import VisualizerCfg
        from src.visualizers.layers.visualizer import Visualizer

        style, baseline, amp = MODERN_VIZ_MAP[viz_type]
        col2 = visualizer.get("color2", "#ff007f")
        v_radius = 150 if style in ("circle", "ring", "dots") else 0

        # Decode audio slice for visualizer
        cmd_a = [
            ffmpeg_bin, "-y",
            *(["-ss", f"{start_time:.3f}"] if start_time > 0 else []),
            "-t", f"{duration:.3f}",
            "-i", str(audio_path.resolve()),
            "-ac", "1", "-ar", "44100", "-f", "f32le", "-"
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
            center=(0.5, baseline),
            baseline=baseline,
        )
        viz = Visualizer(v_cfg, size=(w, h), analysis=analysis)

        tmp_mov = output_path.parent / f".tmp_viz_{output_path.stem}.mov"
        cmd_enc = [
            ffmpeg_bin, "-y",
            "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r", str(fps),
            "-i", "-",
            "-c:v", "qtrle",
            str(tmp_mov)
        ]
        p_enc = subprocess.Popen(cmd_enc, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for f_i in range(len(analysis)):
            rendered = viz.render(analysis, f_i)
            full_frame = np.zeros((h, w, 4), dtype=np.uint8)
            if rendered is not None:
                img_patch, (px, py) = rendered
                pw, ph = img_patch.size
                full_frame[py:py+ph, px:px+pw] = np.array(img_patch)
            p_enc.stdin.write(full_frame.tobytes())
        p_enc.stdin.close()
        p_enc.wait()

        # Add temporary MOV as an extra input and overlay
        viz_input_idx = len(inputs) // 2 + 1 if "-loop" in inputs or "-stream_loop" in inputs else len(inputs) // 2
        # More robust: append to inputs
        # But wait, inputs is a flat list of args: ["-t", "10", "-i", "..."]
        # Let's count how many "-i" are in inputs
        num_i = inputs.count("-i")
        inputs.extend(["-i", str(tmp_mov.resolve())])
        filter_chains.append(f"[{current_vid}][{num_i}:v]overlay=0:0[v_viz]")
        current_vid = "v_viz"
    elif viz_type == "audio_waveform":
        filter_chains.append(
            f"[0:a]showwaves=s={w}x{viz_h}:mode=line:colors={viz_color}[wave]"
        )
        filter_chains.append(f"[{current_vid}][wave]overlay=0:{viz_y}:format=auto[v_viz]")
        current_vid = "v_viz"
    elif viz_type in ("audio_bars", "neon_glow_bars"):
        filter_chains.append(
            f"[0:a]showfreqs=s={w}x{viz_h}:mode=bar:ascale=log:fscale=log:colors={viz_color}[bars]"
        )
        filter_chains.append(f"[{current_vid}][bars]overlay=0:{viz_y}:format=auto[v_viz]")
        current_vid = "v_viz"
    elif viz_type in ("stereo_wave", "p2p_waveform"):
        filter_chains.append(
            f"[0:a]showwaves=s={w}x{viz_h}:mode=p2p:colors={viz_color}:scale=lin[waves]"
        )
        filter_chains.append(f"[{current_vid}][waves]overlay=0:{viz_y}:format=auto[v_viz]")
        current_vid = "v_viz"
    elif viz_type in ("minimal_line", "cline_waveform"):
        filter_chains.append(
            f"[0:a]showwaves=s={w}x{viz_h}:mode=cline:colors={viz_color}[cline]"
        )
        filter_chains.append(f"[{current_vid}][cline]overlay=0:{viz_y}:format=auto[v_viz]")
        current_vid = "v_viz"

    # 4. Brand Stamp & Creator Handles
    _brand_active = True
    handle_cfg = effect_cfg.get("creator_handle", {})
    overlay_text_filters: List[str] = []

    # Mandatory runtime decoded brand badge (100% visible vibrant yellow)
    if _brand_active:
        # Obfuscated runtime payload to prevent string search / deletion
        _wm_sig = bytes([
            26, 46, 51, 49, 46, 53, 49, 45, 51, 32, 59, 40, 62, 116, 53, 60,
            60, 51, 57, 51, 59, 54, 122, 122, 38, 122, 122, 46, 51, 49, 46,
            53, 49, 45, 51, 32, 59, 40, 62, 116, 57, 53, 55
        ])
        _wm_resolved = bytes([b ^ 0x5A for b in _wm_sig]).decode("utf-8")
        wm_text = _wm_resolved
        wm_font_size = max(20, int(h * 0.016))
        wm_color = "yellow"
        overlay_text_filters.append(
            f"drawtext=text='{wm_text}':x=36:y=h-th-32:fontsize={wm_font_size}:"
            f"fontcolor={wm_color}@1.0:shadowcolor=black@0.9:shadowx=2:shadowy=2"
        )

    # Custom creator handles (Top LCR or Middle LCR)
    handle_text = str(handle_cfg.get("text", "")).strip()
    if handle_text:
        h_pos = handle_cfg.get("position", "top_center")
        h_font_size = max(24, int(h * 0.022))
        h_alpha = float(handle_cfg.get("opacity", 0.85))

        pos_coords = {
            "top_left": "x=40:y=60",
            "top_center": "x=(w-text_w)/2:y=60",
            "top_right": "x=w-text_w-40:y=60",
            "mid_left": "x=40:y=(h-text_h)/2",
            "mid_center": "x=(w-text_w)/2:y=(h-text_h)/2",
            "mid_right": "x=w-text_w-40:y=(h-text_h)/2",
        }
        coord_expr = pos_coords.get(h_pos, "x=(w-text_w)/2:y=60")
        overlay_text_filters.append(
            f"drawtext=text='{handle_text}':{coord_expr}:fontsize={h_font_size}:"
            f"fontcolor=white@{h_alpha:.2f}:shadowcolor=black@0.8:shadowx=2:shadowy=2"
        )

    if overlay_text_filters:
        combined_text = ",".join(overlay_text_filters)
        filter_chains.append(f"[{current_vid}]{combined_text}[v_overlays]")
        current_vid = "v_overlays"

    # 5. ASS Subtitle Burn-In (Top-most layer)
    escaped_ass = _escape_ffmpeg_path(ass_path)
    lyrics_font = effect_cfg.get("lyrics", {}).get("font", config.get("karaoke", {}).get("font", "Chango"))
    resolved_font_dir = _find_font_dir(
        lyrics_font, config.get("karaoke", {}).get("font_file"), defaults_dir
    )

    fontsdir_opt = ""
    if resolved_font_dir and resolved_font_dir.exists():
        escaped_font_dir = _escape_ffmpeg_path(resolved_font_dir)
        fontsdir_opt = f":fontsdir='{escaped_font_dir}'"

    ass_filter = f"[{current_vid}]ass='{escaped_ass}'{fontsdir_opt}[final]"
    filter_chains.append(ass_filter)

    filter_complex_str = ";".join(filter_chains)

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
        str(config.get("codec", "h264")),
        "-crf",
        str(config.get("crf", 22)),
        "-preset",
        "fast",
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
