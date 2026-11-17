#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TikTok Wizard — Viral Procedural Particle & Atmospheric Overlay Engine

Replicates studio-grade visual atmospheres (as seen on viral TikTok and Reels):
1. "bokeh_dust": Warm, out-of-focus luminous floating orbs with sinusoidal drift & breathing alpha.
2. "rising_embers": Upward-surging fiery cinders with micro-turbulent flicker & flame gradients.
3. "cinematic_snow": Multi-layer parallax snowfall with wind sway.
4. "radial_burst": Audio-reactive spark shockwaves exploding outward on beat/kick transients.
5. "neon_stars": 4-point diamond cross twinkle stars & cosmic stardust.
6. "none": Clean background (disabled).

Renders directly to transparent QuickTime MOV (qtrle) via FFmpeg rawvideo pipe at 100+ FPS.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Color Parsing Utilities
# ---------------------------------------------------------------------------
def _parse_color_rgb(color_str: str, default: Tuple[int, int, int] = (255, 215, 120)) -> Tuple[int, int, int]:
    """Parse color string in hex (#RRGGBB, #RGB), ASS format (&H00BBGGRR&), or name."""
    if not color_str or not isinstance(color_str, str):
        return default
    c = color_str.strip()
    if c.startswith("&H") and c.endswith("&"):
        # ASS hex: &HAABBGGRR& or &HBBGGRR&
        val = c[2:-1]
        if len(val) >= 6:
            try:
                b = int(val[-6:-4], 16)
                g = int(val[-4:-2], 16)
                r = int(val[-2:], 16)
                return (r, g, b)
            except ValueError:
                return default
    if c.startswith("#"):
        c = c.lstrip("#")
        if len(c) == 6:
            try:
                return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))
            except ValueError:
                return default
        elif len(c) == 3:
            try:
                return (int(c[0]*2, 16), int(c[1]*2, 16), int(c[2]*2, 16))
            except ValueError:
                return default

    named_colors = {
        "gold": (255, 215, 0),
        "amber": (255, 170, 40),
        "orange": (255, 100, 20),
        "crimson": (255, 30, 60),
        "red": (255, 40, 40),
        "cyan": (0, 240, 255),
        "teal": (0, 220, 180),
        "pink": (255, 0, 128),
        "magenta": (255, 0, 200),
        "violet": (180, 0, 255),
        "purple": (150, 40, 255),
        "green": (0, 255, 102),
        "lime": (57, 255, 20),
        "white": (255, 255, 255),
        "ice": (220, 240, 255),
    }
    return named_colors.get(c.lower(), default)


# ---------------------------------------------------------------------------
# Audio Transient & Beat Envelope Extraction
# ---------------------------------------------------------------------------
def extract_beat_envelope(
    audio_path: Path,
    duration: float,
    start_time: float,
    fps: int,
    ffmpeg_bin: str = "ffmpeg",
) -> np.ndarray:
    """Extract frame-by-frame normalized beat intensity array [0.0, 1.0] from audio."""
    total_frames = max(1, int(round(duration * fps)))
    cmd = [
        ffmpeg_bin, "-y",
        *(["-ss", f"{start_time:.3f}"] if start_time > 0 else []),
        "-t", f"{duration:.3f}",
        "-i", str(audio_path.resolve()),
        "-ac", "1", "-ar", "44100", "-f", "f32le", "-"
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True)
        raw_audio = np.frombuffer(proc.stdout, dtype=np.float32)
    except Exception:
        return np.zeros(total_frames, dtype=np.float32)

    if len(raw_audio) == 0:
        return np.zeros(total_frames, dtype=np.float32)

    # Frame RMS energy
    hop = 44100 // fps
    energies = np.zeros(total_frames, dtype=np.float32)
    for f in range(total_frames):
        s = f * hop
        e = min(len(raw_audio), s + hop)
        if e > s:
            chunk = raw_audio[s:e]
            energies[f] = np.sqrt(np.mean(chunk * chunk))

    max_e = float(np.max(energies))
    if max_e > 1e-5:
        norm_e = energies / max_e
    else:
        norm_e = energies

    # Onset / Transient acceleration: difference from moving average
    diff = np.diff(norm_e, prepend=0.0)
    pos_diff = np.maximum(0.0, diff)
    max_d = float(np.max(pos_diff))
    if max_d > 1e-5:
        beat_curve = (pos_diff / max_d) ** 1.3
    else:
        beat_curve = pos_diff

    return np.clip(beat_curve, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Procedural Particle Renderers
# ---------------------------------------------------------------------------
def _render_bokeh_dust(
    draw: ImageDraw.ImageDraw,
    particles: np.ndarray,
    frame: int,
    w: int,
    h: int,
    color_rgb: Tuple[int, int, int],
    beat_val: float,
    speed_mult: float,
) -> None:
    """Render floating out-of-focus luminous orbs."""
    # particles[:, 0] = x
    # particles[:, 1] = y
    # particles[:, 2] = vx
    # particles[:, 3] = vy
    # particles[:, 4] = radius
    # particles[:, 5] = base_alpha
    # particles[:, 6] = phase
    r_base, g_base, b_base = color_rgb

    # Update positions
    wobble = np.sin(particles[:, 6] + frame * 0.045) * 1.4 * speed_mult
    particles[:, 0] += (particles[:, 2] + wobble) * speed_mult
    particles[:, 1] += particles[:, 3] * speed_mult

    # Wrap boundaries
    particles[:, 0] %= w
    particles[:, 1] %= h

    # Audio reactivity factor
    alpha_boost = 1.0 + beat_val * 0.45
    radius_boost = 1.0 + beat_val * 0.25

    for p in particles:
        px, py = p[0], p[1]
        rad = p[4] * radius_boost
        alpha = min(1.0, p[5] * alpha_boost)

        # Draw soft concentric glow rings
        # 1. Outer faint glow
        a_outer = int(alpha * 70)
        if a_outer > 0:
            draw.ellipse([px - rad, py - rad, px + rad, py + rad], fill=(r_base, g_base, b_base, a_outer))
        # 2. Mid halo
        r_mid = rad * 0.65
        a_mid = int(alpha * 150)
        if a_mid > 0:
            draw.ellipse([px - r_mid, py - r_mid, px + r_mid, py + r_mid], fill=(r_base, g_base, b_base, a_mid))
        # 3. Core
        r_core = rad * 0.35
        a_core = int(alpha * 255)
        if a_core > 0:
            draw.ellipse([px - r_core, py - r_core, px + r_core, py + r_core], fill=(255, 255, 255, a_core))


def _render_rising_embers(
    draw: ImageDraw.ImageDraw,
    particles: np.ndarray,
    frame: int,
    w: int,
    h: int,
    color_rgb: Tuple[int, int, int],
    beat_val: float,
    speed_mult: float,
) -> None:
    """Render upward-surging glowing fiery embers with micro-turbulent flicker."""
    r_base, g_base, b_base = color_rgb

    # Velocity surge on beat
    beat_surge = 1.0 + beat_val * 0.85

    particles[:, 0] += (particles[:, 2] + np.sin(particles[:, 6] + frame * 0.12) * 2.2) * speed_mult
    particles[:, 1] += particles[:, 3] * speed_mult * beat_surge

    # Reset embers that reach the top back to bottom
    top_mask = particles[:, 1] < -20
    particles[top_mask, 1] = np.random.uniform(h, h + 80, np.sum(top_mask))
    particles[top_mask, 0] = np.random.uniform(0, w, np.sum(top_mask))

    particles[:, 0] %= w

    for p in particles:
        px, py = p[0], p[1]
        rad = p[4]
        # Frame-to-frame flicker
        flicker = 0.75 + 0.25 * math.sin(p[6] * 7.0 + frame * 0.35)
        alpha = min(1.0, p[5] * flicker * (1.0 + beat_val * 0.5))

        # Color gradient: hot yellow center to amber/crimson exterior
        draw.ellipse([px - rad*1.8, py - rad*1.8, px + rad*1.8, py + rad*1.8], fill=(r_base, g_base, b_base, int(alpha * 90)))
        draw.ellipse([px - rad, py - rad, px + rad, py + rad], fill=(255, min(255, g_base + 60), 40, int(alpha * 220)))
        draw.ellipse([px - rad*0.4, py - rad*0.4, px + rad*0.4, py + rad*0.4], fill=(255, 255, 200, int(alpha * 255)))


def _render_cinematic_snow(
    draw: ImageDraw.ImageDraw,
    particles: np.ndarray,
    frame: int,
    w: int,
    h: int,
    color_rgb: Tuple[int, int, int],
    beat_val: float,
    speed_mult: float,
) -> None:
    """Render parallax multi-depth snowfall with horizontal wind sway."""
    r_base, g_base, b_base = color_rgb

    # Horizontal wind wave
    wind = math.sin(frame * 0.035) * 1.8 * speed_mult
    particles[:, 0] += (particles[:, 2] + wind) * speed_mult
    particles[:, 1] += particles[:, 3] * speed_mult  # Snow falls downward

    # Wrap snow that falls past bottom back to top
    bottom_mask = particles[:, 1] > h + 20
    particles[bottom_mask, 1] = np.random.uniform(-40, -5, np.sum(bottom_mask))
    particles[bottom_mask, 0] = np.random.uniform(0, w, np.sum(bottom_mask))

    particles[:, 0] %= w

    for p in particles:
        px, py = p[0], p[1]
        rad = p[4]
        alpha = p[5]
        # Faint soft flake
        draw.ellipse([px - rad, py - rad, px + rad, py + rad], fill=(r_base, g_base, b_base, int(alpha * 230)))


def _render_radial_burst(
    draw: ImageDraw.ImageDraw,
    sparks: List[dict],
    frame: int,
    w: int,
    h: int,
    color_rgb: Tuple[int, int, int],
    beat_val: float,
    speed_mult: float,
) -> None:
    """Render High-energy explosive radial beat bursts shooting outward from center."""
    cx, cy = w // 2, int(h * 0.65)
    r_base, g_base, b_base = color_rgb

    # Spawn spark burst if heavy beat transient detected
    if beat_val > 0.45:
        burst_count = int(24 + beat_val * 35)
        for _ in range(burst_count):
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(7.0, 22.0) * speed_mult
            sparks.append({
                "x": float(cx),
                "y": float(cy),
                "vx": float(math.cos(angle) * speed),
                "vy": float(math.sin(angle) * speed),
                "rad": float(np.random.uniform(2.5, 6.0)),
                "alpha": 1.0,
                "decay": float(np.random.uniform(0.035, 0.075)),
                "color": (r_base, g_base, b_base) if np.random.random() > 0.3 else (255, 255, 255),
            })

    # Update and draw active sparks
    surviving_sparks = []
    for sp in sparks:
        sp["x"] += sp["vx"]
        sp["y"] += sp["vy"]
        sp["vx"] *= 0.94  # Drag deceleration
        sp["vy"] *= 0.94
        sp["alpha"] -= sp["decay"]

        if sp["alpha"] > 0.05 and 0 <= sp["x"] <= w and 0 <= sp["y"] <= h:
            surviving_sparks.append(sp)
            sx, sy = sp["x"], sp["y"]
            srad = sp["rad"]
            cr, cg, cb = sp["color"]
            a = int(sp["alpha"] * 255)

            # Draw spark with motion tail
            draw.ellipse([sx - srad, sy - srad, sx + srad, sy + srad], fill=(cr, cg, cb, a))
            tail_x = sx - sp["vx"] * 1.5
            tail_y = sy - sp["vy"] * 1.5
            draw.line([(tail_x, tail_y), (sx, sy)], fill=(cr, cg, cb, int(a * 0.6)), width=max(1, int(srad * 0.8)))

    sparks.clear()
    sparks.extend(surviving_sparks[:250])  # Cap memory buffer


def _render_neon_stars(
    draw: ImageDraw.ImageDraw,
    particles: np.ndarray,
    frame: int,
    w: int,
    h: int,
    color_rgb: Tuple[int, int, int],
    beat_val: float,
    speed_mult: float,
) -> None:
    """Render cosmic 4-point diamond cross twinkle stars & stardust."""
    r_base, g_base, b_base = color_rgb

    particles[:, 0] += particles[:, 2] * speed_mult
    particles[:, 1] += particles[:, 3] * speed_mult
    particles[:, 0] %= w
    particles[:, 1] %= h

    for p in particles:
        px, py = p[0], p[1]
        rad = p[4]
        # Rotational twinkle phase
        twinkle = abs(math.sin(p[6] + frame * 0.09))
        alpha = min(1.0, p[5] * (0.3 + 0.7 * twinkle) * (1.0 + beat_val * 0.4))
        a_int = int(alpha * 255)
        if a_int <= 5:
            continue

        # 4-point star cross
        draw.line([(px - rad * 1.6, py), (px + rad * 1.6, py)], fill=(r_base, g_base, b_base, a_int), width=1)
        draw.line([(px, py - rad * 1.6), (px, py + rad * 1.6)], fill=(r_base, g_base, b_base, a_int), width=1)
        # Center glow dot
        draw.ellipse([px - rad*0.5, py - rad*0.5, px + rad*0.5, py + rad*0.5], fill=(255, 255, 255, a_int))


# ---------------------------------------------------------------------------
# High-Level Particle Video Generator (QuickTime Alpha MOV Stream)
# ---------------------------------------------------------------------------
def generate_particle_layer(
    audio_path: Path,
    duration: float,
    start_time: float,
    fps: int,
    width: int,
    height: int,
    particle_cfg: dict,
    output_mov_path: Path,
    ffmpeg_bin: str = "ffmpeg",
) -> Optional[Path]:
    """
    Generate a full transparent QuickTime MOV containing the requested particle atmosphere.
    Returns output_mov_path if rendered, or None if disabled ('none').
    """
    p_type = str(particle_cfg.get("type", "none")).lower().strip()
    if p_type in ("none", "disabled", "off", ""):
        return None

    density_str = str(particle_cfg.get("density", "medium")).lower()
    density_map = {"low": 30, "medium": 65, "high": 130}
    num_particles = density_map.get(density_str, 65)

    speed_str = str(particle_cfg.get("speed", "normal")).lower()
    speed_map = {"slow": 0.6, "normal": 1.0, "fast": 1.8}
    speed_mult = speed_map.get(speed_str, 1.0)

    color_cfg = particle_cfg.get("color", "auto")
    default_color = (255, 215, 120)
    if p_type == "rising_embers":
        default_color = (255, 90, 20)
    elif p_type == "cinematic_snow":
        default_color = (240, 248, 255)
    elif p_type == "radial_burst":
        default_color = (0, 240, 255)
    elif p_type == "neon_stars":
        default_color = (255, 0, 200)

    color_rgb = _parse_color_rgb(color_cfg, default_color)

    # Audio reactivity setup
    audio_reactive = particle_cfg.get("audio_reactive", "pulse")
    if audio_reactive in ("off", "none", False):
        beats = np.zeros(max(1, int(round(duration * fps))), dtype=np.float32)
    else:
        beats = extract_beat_envelope(audio_path, duration, start_time, fps, ffmpeg_bin)

    total_frames = max(1, int(round(duration * fps)))

    # Initialize particle state array [N, 7]
    particles = np.zeros((num_particles, 7), dtype=np.float32)
    particles[:, 0] = np.random.uniform(0, width, num_particles)
    particles[:, 1] = np.random.uniform(0, height, num_particles)

    if p_type == "bokeh_dust":
        particles[:, 2] = np.random.uniform(-0.4, 0.4, num_particles)
        particles[:, 3] = np.random.uniform(-1.2, -0.3, num_particles)
        particles[:, 4] = np.random.uniform(12.0, 38.0, num_particles)
        particles[:, 5] = np.random.uniform(0.18, 0.55, num_particles)
        particles[:, 6] = np.random.uniform(0, 2 * np.pi, num_particles)
    elif p_type == "rising_embers":
        particles[:, 2] = np.random.uniform(-0.8, 0.8, num_particles)
        particles[:, 3] = np.random.uniform(-4.5, -2.0, num_particles)
        particles[:, 4] = np.random.uniform(3.0, 7.5, num_particles)
        particles[:, 5] = np.random.uniform(0.4, 0.9, num_particles)
        particles[:, 6] = np.random.uniform(0, 2 * np.pi, num_particles)
    elif p_type == "cinematic_snow":
        particles[:, 2] = np.random.uniform(-0.6, 0.6, num_particles)
        particles[:, 3] = np.random.uniform(1.8, 4.5, num_particles)  # Downward
        particles[:, 4] = np.random.uniform(2.5, 7.5, num_particles)
        particles[:, 5] = np.random.uniform(0.3, 0.85, num_particles)
        particles[:, 6] = np.random.uniform(0, 2 * np.pi, num_particles)
    elif p_type == "neon_stars":
        particles[:, 2] = np.random.uniform(-0.25, 0.25, num_particles)
        particles[:, 3] = np.random.uniform(-0.5, 0.5, num_particles)
        particles[:, 4] = np.random.uniform(4.0, 11.0, num_particles)
        particles[:, 5] = np.random.uniform(0.35, 0.85, num_particles)
        particles[:, 6] = np.random.uniform(0, 2 * np.pi, num_particles)

    sparks_pool: List[dict] = []

    output_mov_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_enc = [
        ffmpeg_bin, "-y",
        "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-",
        "-c:v", "qtrle",
        str(output_mov_path.resolve())
    ]

    p_enc = subprocess.Popen(cmd_enc, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    try:
        report_step = max(30, total_frames // 20)
        for f in range(total_frames):
            if f % report_step == 0 or f == total_frames - 1:
                pct = int(round((f + 1) / total_frames * 100))
                print(f"  [Atmosphere/FX] Rendering {p_type} frames: {f+1}/{total_frames} ({pct}%)...", end="\r", flush=True)

            b_val = float(beats[f]) if f < len(beats) else 0.0

            img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)

            if p_type == "bokeh_dust":
                _render_bokeh_dust(draw, particles, f, width, height, color_rgb, b_val, speed_mult)
            elif p_type == "rising_embers":
                _render_rising_embers(draw, particles, f, width, height, color_rgb, b_val, speed_mult)
            elif p_type == "cinematic_snow":
                _render_cinematic_snow(draw, particles, f, width, height, color_rgb, b_val, speed_mult)
            elif p_type == "radial_burst":
                _render_radial_burst(draw, sparks_pool, f, width, height, color_rgb, b_val, speed_mult)
            elif p_type == "neon_stars":
                _render_neon_stars(draw, particles, f, width, height, color_rgb, b_val, speed_mult)

            frame_bytes = img.tobytes("raw", "RGBA")
            p_enc.stdin.write(frame_bytes)

        p_enc.stdin.close()
        p_enc.wait()
        print(f"  [Atmosphere/FX] Rendering {p_type} frames: {total_frames}/{total_frames} (100%) - Done!       ")
    except Exception as e:
        if p_enc.stdin:
            p_enc.stdin.close()
        p_enc.kill()
        return None

    if output_mov_path.exists() and output_mov_path.stat().st_size > 1000:
        return output_mov_path
    return None
