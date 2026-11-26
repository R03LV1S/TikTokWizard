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

import importlib
import importlib.util
import inspect
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

APP_NAME: str = "TikTok Wizard"
CANONICAL_APP_VERSION: str = "5.10.0"
CANONICAL_APP_VERSION_SHORT: str = "5.10"

_EXTRA_REGISTERED_ENGINES: List[Type[Any]] = []


def get_git_tag_or_hash(repo_dir: Optional[Path] = None) -> Optional[str]:
    """
    Query the active git tag or commit hash dynamically.
    Returns version string (without leading 'v') or None if git is unavailable.
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
            clean = raw.lstrip("v")
            if "-" in clean:
                return clean.split("-")[0]
            return clean
    except Exception:
        pass
    return None


def _parse_ver_tuple(v: str) -> Tuple[int, ...]:
    try:
        parts = [int(p) for p in str(v).strip().lstrip("v").split(".") if p.isdigit()]
        return tuple(parts) if parts else (0,)
    except Exception:
        return (0,)


def get_app_version() -> str:
    """
    Resolve the active TikTok Wizard application version dynamically.
    Uses the newer of the live git repository release tag and CANONICAL_APP_VERSION_SHORT.
    """
    git_v = get_git_tag_or_hash()
    if git_v and git_v.replace(".", "").isdigit():
        if _parse_ver_tuple(git_v) >= _parse_ver_tuple(CANONICAL_APP_VERSION_SHORT):
            return git_v
    return CANONICAL_APP_VERSION_SHORT


def register_alignment_engine(engine_cls: Type[Any]) -> None:
    """Register an external alignment engine class for dynamic auto-discovery."""
    if engine_cls not in _EXTRA_REGISTERED_ENGINES:
        _EXTRA_REGISTERED_ENGINES.append(engine_cls)


def _is_engine_class(obj: Any) -> bool:
    return (
        inspect.isclass(obj)
        and hasattr(obj, "ENGINE_ID")
        and hasattr(obj, "VERSION")
        and callable(getattr(obj, "run", None))
    )


def _load_pipeline_classes(pkg_dir: Path, src_root: Path) -> List[Type[Any]]:
    """Load alignment engine classes from `<pkg_dir>/pipeline.py` safely."""
    pipeline_py = pkg_dir / "pipeline.py"
    if not pipeline_py.is_file():
        return []
    module = None
    default_src = Path(__file__).resolve().parent
    if src_root.resolve() == default_src.resolve():
        mod_name = f"src.{pkg_dir.name}.pipeline"
        try:
            module = importlib.import_module(mod_name)
        except Exception:
            module = None
    else:
        mod_name = f"_dyn_{pkg_dir.name}_{abs(hash(str(pipeline_py)))}"
        try:
            spec = importlib.util.spec_from_file_location(mod_name, str(pipeline_py))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
        except Exception:
            module = None

    if module is None:
        return []

    found: List[Type[Any]] = []
    for _, obj in inspect.getmembers(module, _is_engine_class):
        if obj not in found:
            found.append(obj)
    return found


def discover_alignment_engines(
    search_root: Optional[Path] = None,
    extra_classes: Optional[Sequence[Type[Any]]] = None,
) -> List[Type[Any]]:
    """
    Auto-discover all available alignment engine classes (`src/align*` packages)
    sorted descending by semantic VERSION so newly dropped engines (e.g. align_v8)
    become active automatically with zero static code edits.
    """
    src_root = Path(search_root) if search_root is not None else Path(__file__).resolve().parent
    discovered: List[Type[Any]] = []

    if extra_classes:
        for cls in extra_classes:
            if _is_engine_class(cls) and cls not in discovered:
                discovered.append(cls)

    for cls in _EXTRA_REGISTERED_ENGINES:
        if search_root is None and _is_engine_class(cls) and cls not in discovered:
            discovered.append(cls)

    if src_root.is_dir():
        for pkg in sorted(src_root.iterdir(), key=lambda p: p.name.lower()):
            if pkg.is_dir() and pkg.name.startswith("align"):
                for cls in _load_pipeline_classes(pkg, src_root):
                    if cls not in discovered:
                        discovered.append(cls)

    # Sort descending by semantic VERSION, preferring non-legacy over v1 on ties
    discovered.sort(
        key=lambda c: (
            _parse_ver_tuple(str(getattr(c, "VERSION", "0"))),
            0 if getattr(c, "ENGINE_ID", "") == "universal_engine_v1" else 1,
            getattr(c, "__name__", ""),
        ),
        reverse=True,
    )
    return discovered


def resolve_engine_class(
    engine_spec: Optional[str] = None,
    search_root: Optional[Path] = None,
    extra_classes: Optional[Sequence[Type[Any]]] = None,
) -> Type[Any]:
    """
    Resolve an alignment engine class dynamically from `engine_spec` ('auto', 'v8', 'v5', 'v2', 'v1', ENGINE_ID).
    """
    engines = discover_alignment_engines(search_root=search_root, extra_classes=extra_classes)
    if not engines:
        raise RuntimeError("No alignment engines discovered.")

    spec = (engine_spec or "auto").strip().lower()
    if spec in ("auto", "latest", "default"):
        return engines[0]

    # Legacy alias v2 -> latest 5.x (align_v2 / greenfield_v5) if present, else highest
    if spec in ("v1", "legacy", "universal_engine_v1"):
        for cls in engines:
            ver = str(getattr(cls, "VERSION", ""))
            eid = str(getattr(cls, "ENGINE_ID", "")).lower()
            if eid == "universal_engine_v1" or ver.startswith("1."):
                return cls

    if spec in ("v2", "align_v2"):
        for cls in engines:
            if ".align_v2." in getattr(cls, "__module__", "") or "v5" in str(getattr(cls, "ENGINE_ID", "")).lower():
                return cls

    spec_num = spec.lstrip("v")
    for cls in engines:
        eid = str(getattr(cls, "ENGINE_ID", "")).lower()
        ver = str(getattr(cls, "VERSION", "")).lower()
        cname = getattr(cls, "__name__", "").lower()
        if spec == eid or spec == cname or ver == spec_num or ver.startswith(f"{spec_num}."):
            return cls

    # Fallback: return highest discovered modern engine
    return engines[0]


def _instantiate_with_signature(cls: Type[Any], kwargs: Dict[str, Any]) -> Any:
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_var_kw:
        return cls(**kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in params and k != "self"}
    return cls(**filtered)


def create_alignment_engine(
    engine_spec: Optional[str] = None,
    search_root: Optional[Path] = None,
    extra_classes: Optional[Sequence[Type[Any]]] = None,
    **kwargs: Any,
) -> Any:
    """
    Instantiate the requested or highest-version discovered alignment engine dynamically,
    inspecting constructor signature for forward compatibility and falling back gracefully.
    """
    engines = discover_alignment_engines(search_root=search_root, extra_classes=extra_classes)
    primary = resolve_engine_class(
        engine_spec=engine_spec, search_root=search_root, extra_classes=extra_classes
    )

    spec_low = (engine_spec or "auto").strip().lower()
    ordered = [primary]
    if spec_low in ("auto", "latest", "default"):
        for cls in engines:
            if cls not in ordered:
                ordered.append(cls)

    last_exc: Optional[Exception] = None
    for cls in ordered:
        try:
            return _instantiate_with_signature(cls, kwargs)
        except Exception as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Failed to instantiate any alignment engine.")


def get_engine_info(engine_instance: Optional[object] = None) -> Tuple[str, str, str]:
    """
    Introspect the active alignment engine dynamically (either a live instance or
    the highest-version auto-discovered engine class).

    Returns:
        (engine_name, engine_version, engine_display_name)
    """
    target: Any = engine_instance
    if target is None:
        try:
            target = resolve_engine_class("auto")
        except Exception:
            target = None

    if target is not None:
        eng_name = getattr(target, "NAME", getattr(target, "name", "Alignment Engine"))
        eng_ver = str(getattr(target, "VERSION", getattr(target, "version", get_app_version())))
        eng_disp = getattr(
            target,
            "DISPLAY_NAME",
            getattr(target, "display_name", f"{eng_name} v{eng_ver}"),
        )
        return str(eng_name), str(eng_ver), str(eng_disp)

    # Schema fallback if discovery had no importable engine
    try:
        from src.align_v2.schema import ENGINE_VERSION
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
    """
    app_v = get_app_version()
    _, _, eng_disp = get_engine_info(engine_instance)
    return f"{APP_NAME} v{app_v} ({eng_disp})"
