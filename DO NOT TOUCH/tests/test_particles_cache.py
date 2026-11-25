"""Unit test suite for atmosphere & FX persistent caching."""

import sys
import os
from pathlib import Path

# Ensure DO NOT TOUCH root is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline.particles import (
    generate_particle_layer,
    resolve_overlay_path,
    get_fx_cache_dir,
    get_stock_overlays_dir
)


def test_stock_and_cache_directories():
    stock_dir = get_stock_overlays_dir()
    cache_dir = get_fx_cache_dir()
    assert stock_dir.exists()
    assert cache_dir.exists()


def test_overlay_instant_cache_hit(tmp_path):
    # Test stock cinematic_snow resolution (falls back to temp fixture on fresh clones where binary .mov assets are gitignored)
    cfg = {"type": "cinematic_snow"}
    defaults_dir = None
    stock_mov = get_stock_overlays_dir() / "cinematic_snow.mov"
    if not stock_mov.exists():
        overlays_dir = tmp_path / "Overlays"
        overlays_dir.mkdir(parents=True, exist_ok=True)
        (overlays_dir / "cinematic_snow.mov").write_bytes(b"\x00" * 2048)
        defaults_dir = str(tmp_path)

    path, is_cached = resolve_overlay_path(cfg, 1080, 1920, 30, defaults_dir=defaults_dir)
    assert is_cached is True
    assert path is not None
    assert path.exists()

    # Calling generate_particle_layer should return instantly
    res = generate_particle_layer(
        particle_cfg=cfg,
        duration=15.0,
        fps=30,
        width=1080,
        height=1920,
        defaults_dir=defaults_dir,
    )
    assert res is not None
    assert res.exists()
    assert res.resolve() == path.resolve()


def test_disabled_particles_return_none():
    assert generate_particle_layer(particle_cfg={"type": "none"}) is None
    assert generate_particle_layer(particle_cfg={"type": "disabled"}) is None
    assert generate_particle_layer(particle_cfg=None) is None
