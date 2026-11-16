# __init__.py
"""TikTok Wizard — Greenfield Universal Alignment Engine (v4.1)."""

from __future__ import annotations

import os

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

__version__: str = "4.1.0"
__engine__: str = "greenfield_v4_1_real_mms_fa"
__all__ = ["__version__", "__engine__"]
