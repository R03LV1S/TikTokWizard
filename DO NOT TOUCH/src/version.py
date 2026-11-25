#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TikTok Wizard — Dynamic Version & Engine Introspection Engine (SSOT)

Provides universal, self-calibrating version resolution and engine telemetry:
1. Dynamically detects git release tags/commits without hardcoded string drift.
2. Self-introspects active alignment engine classes (V5, V8, etc.) to extract
   engine version, name, and architecture dynamically.
3. Automatically adapts banners, manifests, and companion metadata to the
   live runtime environment with zero manual user intervention.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

APP_NAME: str = "TikTok Wizard"
CANONICAL_APP_VERSION: str = "5.8.0"
CANONICAL_APP_VERSION_SHORT: str = "5.8"


def get_git_tag_or_hash(repo_dir: Optional[Path] = None) -> Optional[str]:
    """
    Query the active git tag or commit hash dynamically.
    Returns tag string like '5.7' or 'v5.7' or None if git is unavailable.
    """
    try:
        root = repo_dir or Path(__file__).resolve().parent.parent.parent
        res = subprocess.run(
            ["git", "describe", "--tags", "--always"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=2,
        )
        if res.returncode == 0 and res.stdout.strip():
            raw = res.stdout.strip()
            # Clean leading 'v'
            clean = raw.lstrip("v")
            # If tag contains git commit suffix like '5.7-2-g1a2b3c', extract base tag
            if "-" in clean:
                return clean.split("-")[0]
            return clean
    except Exception:
        pass
    return None


def get_app_version() -> str:
    """
    Resolve the active TikTok Wizard application version dynamically.
    Prioritizes live git repository release tag, falling back to CANONICAL_APP_VERSION_SHORT.
    """
    git_v = get_git_tag_or_hash()
    if git_v and git_v.replace(".", "").isdigit():
        return git_v
    return CANONICAL_APP_VERSION_SHORT


def get_engine_info(engine_instance: Optional[object] = None) -> Tuple[str, str, str]:
    """
    Introspect the active alignment engine dynamically.
    Supports V5, V6, V8, or future engines via self-introspection attributes.

    Returns:
        (engine_name, engine_version, engine_display_name)
    """
    if engine_instance is not None:
        # Check standard introspection attributes
        eng_id = getattr(engine_instance, "ENGINE_ID", getattr(engine_instance, "engine_id", "unknown_engine"))
        eng_name = getattr(engine_instance, "NAME", getattr(engine_instance, "name", "Alignment Engine"))
        eng_ver = str(getattr(engine_instance, "VERSION", getattr(engine_instance, "version", get_app_version())))
        eng_disp = getattr(
            engine_instance,
            "DISPLAY_NAME",
            getattr(engine_instance, "display_name", f"{eng_name} v{eng_ver}"),
        )
        return str(eng_name), str(eng_ver), str(eng_disp)

    # Fallback to schema definition
    try:
        from src.align_v2.schema import ENGINE_VERSION, ENGINE_V5
        eng_name = "Greenfield Universal Engine"
        eng_ver = str(ENGINE_VERSION)
        eng_disp = f"Greenfield Universal Engine v{eng_ver} (Dual-Model CTC Consensus)"
        return eng_name, eng_ver, eng_disp
    except Exception:
        app_v = get_app_version()
        return "Greenfield Universal Engine", app_v, f"Greenfield Universal Engine v{app_v}"


def get_app_header(engine_instance: Optional[object] = None) -> str:
    """
    Generate unified CLI top banner header dynamically.
    Example: 'TikTok Wizard v5.7 (Greenfield Universal Engine v5.7)'
    """
    app_v = get_app_version()
    _, eng_v, eng_disp = get_engine_info(engine_instance)
    return f"{APP_NAME} v{app_v} ({eng_disp})"
