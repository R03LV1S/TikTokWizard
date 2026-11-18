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


def test_overlay_instant_cache_hit():
    # Test stock cinematic_snow resolution
    cfg = {"type": "cinematic_snow"}
    path, is_cached = resolve_overlay_path(cfg, 1080, 1920, 30)
    assert is_cached is True
    assert path is not None
    assert path.exists()

    # Calling generate_particle_layer should return instantly
    res = generate_particle_layer(
        particle_cfg=cfg,
        duration=15.0,
        fps=30,
        width=1080,
        height=1920
    )
    assert res is not None
    assert res.exists()
    assert res.resolve() == path.resolve()


def test_disabled_particles_return_none():
    assert generate_particle_layer(particle_cfg={"type": "none"}) is None
    assert generate_particle_layer(particle_cfg={"type": "disabled"}) is None
    assert generate_particle_layer(particle_cfg=None) is None
