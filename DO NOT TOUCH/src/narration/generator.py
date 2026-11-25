#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Narration/voiceover generation (optional, separate from karaoke pipeline).

Long-form capable design:
- chunked synthesis from long scripts (15 min to multi-hour lectures)
- per-job working directory with stable chunk files and manifest
- resumable runs (skip completed chunks)
- streaming, memory-bounded merge (no full-audio concat in RAM)
- graceful failure with setup guidance when optional deps are missing

Backends:
- f5_tts: voice-cloning TTS (best for supported languages, e.g. English/Chinese).
- mms_fa: Meta MMS Persian (fas) TTS for accurate Persian narration (fixed voice).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import textwrap
import time
import uuid
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
SCRIPT_EXTS = (".txt",)
JOB_MANIFEST_NAME = "narration_job.json"
DEFAULT_MAX_CHARS_PER_CHUNK = 700
DEFAULT_PAUSE_SECONDS = 0.25
DEFAULT_BACKEND = "auto"
SUPPORTED_BACKENDS = ("auto", "f5_tts", "mms_fa")


class NarrationError(RuntimeError):
    """Raised for user-facing narration/setup errors."""


@dataclass
class NarrationPaths:
    root: Path
    scripts_dir: Path
    voices_dir: Path
    output_dir: Path


def get_narration_paths(start_here_dir: Path) -> NarrationPaths:
    root = Path(start_here_dir) / "Narration"
    return NarrationPaths(
        root=root,
        scripts_dir=root / "Scripts",
        voices_dir=root / "Voices",
        output_dir=root / "Output",
    )


def ensure_narration_dirs(paths: NarrationPaths) -> None:
    for p in (paths.root, paths.scripts_dir, paths.voices_dir, paths.output_dir):
        p.mkdir(parents=True, exist_ok=True)


def list_scripts(paths: NarrationPaths) -> List[Path]:
    if not paths.scripts_dir.exists():
        return []
    return sorted(
        [
            p
            for p in paths.scripts_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SCRIPT_EXTS
        ],
        key=lambda p: p.name.lower(),
    )


def list_voice_samples(paths: NarrationPaths) -> List[Path]:
    if not paths.voices_dir.exists():
        return []
    return sorted(
        [
            p
            for p in paths.voices_dir.iterdir()
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS
        ],
        key=lambda p: p.name.lower(),
    )


def list_narration_jobs(output_dir: Path) -> List[Path]:
    """Return job directories under Output that contain a narration manifest."""
    out = Path(output_dir)
    if not out.exists():
        return []
    jobs: List[Path] = []
    for p in out.iterdir():
        if p.is_dir() and (p / JOB_MANIFEST_NAME).exists():
            jobs.append(p)
    return sorted(jobs, key=lambda p: p.name.lower())


def _has_spec(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def check_f5_tts_available() -> Tuple[bool, str]:
    """Return (available, detail). Does not import heavy modules unless present."""
    if not _has_spec("f5_tts"):
        return False, "Python package 'f5_tts' is not installed in this environment."
    return True, "f5_tts package detected."


def check_mms_fa_available() -> Tuple[bool, str]:
    """Check if MMS Persian (facebook/mms-tts-fas) deps are available locally."""
    missing = [m for m in ("transformers", "torch") if not _has_spec(m)]
    if missing:
        return False, f"Missing deps for MMS Persian TTS: {', '.join(missing)}"
    return True, "MMS Persian TTS deps detected (transformers+torch)."


def detect_persian_script(text: str) -> bool:
    if not text:
        return False
    persian = sum(
        1
        for ch in text
        if "\u0600" <= ch <= "\u06ff"
        or "\u0750" <= ch <= "\u077f"
        or "\ufb50" <= ch <= "\ufdff"
        or "\ufe70" <= ch <= "\ufeff"
    )
    alpha = sum(1 for ch in text if ch.isalpha())
    return persian >= 20 and (alpha == 0 or (persian / max(1, alpha)) >= 0.4)


def resolve_backend(requested: str, script_text: Optional[str] = None) -> str:
    req = (requested or DEFAULT_BACKEND).strip().lower()
    if req not in SUPPORTED_BACKENDS:
        return "f5_tts"
    if req == "auto":
        if script_text and detect_persian_script(script_text):
            return "mms_fa"
        return "f5_tts"
    return req


def _safe_read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8-sig")


def _read_script_text(script_path: Path) -> str:
    text = _safe_read_text(script_path).strip()
    if not text:
        raise NarrationError("Selected script is empty.")
    return text


def _chunk_text(text: str, max_chars: int = DEFAULT_MAX_CHARS_PER_CHUNK) -> List[str]:
    # Prefer paragraph/sentence-aware chunks for more stable TTS and resumability.
    max_chars = max(120, int(max_chars or DEFAULT_MAX_CHARS_PER_CHUNK))
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not paras:
        paras = [text.strip()]
    chunks: List[str] = []
    for para in paras:
        if len(para) <= max_chars:
            chunks.append(para)
            continue
        sentences = re.split(r"(?<=[.!?،؛؟])\s+", para)
        buf = ""
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if len(s) > max_chars:
                if buf:
                    chunks.append(buf.strip())
                    buf = ""
                chunks.extend(
                    textwrap.wrap(
                        s,
                        width=max_chars,
                        break_long_words=False,
                        break_on_hyphens=False,
                    )
                )
                continue
            cand = (buf + " " + s).strip() if buf else s
            if len(cand) <= max_chars:
                buf = cand
            else:
                if buf:
                    chunks.append(buf.strip())
                buf = s
        if buf:
            chunks.append(buf.strip())
    return [c for c in chunks if c]


def _slugify(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("._")
    return s or "narration"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fmt_hms(seconds: float) -> str:
    try:
        s = int(max(0, round(float(seconds))))
    except Exception:
        s = 0
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{sec:02d}"
    return f"{m:d}:{sec:02d}"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _load_f5_tts(model_name: Optional[str] = None):
    """Best-effort loader for common F5-TTS Python APIs."""
    last_err: Optional[Exception] = None
    try:
        from f5_tts.api import F5TTS  # type: ignore

        try:
            return F5TTS(model=model_name) if model_name else F5TTS()
        except TypeError:
            return F5TTS()
    except Exception as e:
        last_err = e

    try:
        import f5_tts.api  # type: ignore  # noqa: F401
    except Exception as e:
        last_err = e

    raise NarrationError(
        "F5-TTS is installed but its Python API could not be initialized. "
        "Please verify your f5-tts installation/version and model setup. "
        f"Details: {last_err}"
    )


def _synthesize_with_f5(
    tts: object,
    text: str,
    ref_audio: Path,
    out_path: Path,
    ref_text: str = "",
) -> None:
    """Best-effort synthesis across common F5TTS.infer signatures."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    infer = getattr(tts, "infer", None)
    if infer is None:
        raise NarrationError(
            "F5-TTS API object has no 'infer' method; unsupported f5-tts version/API."
        )

    attempts = [
        dict(
            ref_file=str(ref_audio),
            ref_text=ref_text,
            gen_text=text,
            file_wave=str(out_path),
        ),
        dict(
            ref_file=str(ref_audio),
            ref_text=ref_text,
            gen_text=text,
            output_path=str(out_path),
        ),
        dict(
            ref_audio=str(ref_audio),
            ref_text=ref_text,
            gen_text=text,
            file_wave=str(out_path),
        ),
        dict(
            ref_file=str(ref_audio),
            ref_text=ref_text,
            text=text,
            file_wave=str(out_path),
        ),
    ]
    last_err: Optional[Exception] = None
    for kwargs in attempts:
        try:
            infer(**kwargs)
            if out_path.exists() and out_path.stat().st_size > 0:
                return
        except TypeError as e:
            last_err = e
            continue
        except Exception as e:
            raise NarrationError(f"F5-TTS synthesis failed: {e}") from e

    try:
        result = infer(ref_file=str(ref_audio), ref_text=ref_text, gen_text=text)
        if isinstance(result, (str, Path)):
            rp = Path(result)
            if rp.exists():
                if rp.resolve() != out_path.resolve():
                    out_path.write_bytes(rp.read_bytes())
                if out_path.exists() and out_path.stat().st_size > 0:
                    return
        if isinstance(result, tuple) and result:
            wav = result[0]
            sr = result[1] if len(result) > 1 else 24000
            try:
                import soundfile as sf  # type: ignore

                sf.write(str(out_path), wav, int(sr))
                if out_path.exists() and out_path.stat().st_size > 0:
                    return
            except Exception:
                pass
    except Exception as e:
        last_err = e

    raise NarrationError(
        "Could not synthesize with the installed F5-TTS API (signature/output not recognized). "
        f"Please check your f5-tts version. Details: {last_err}"
    )


def _load_mms_fa():
    """Load Meta MMS Persian TTS model/tokenizer (cached locally after first download)."""
    try:
        from transformers import VitsModel, AutoTokenizer  # type: ignore
        import torch  # type: ignore  # noqa: F401
    except Exception as e:
        raise NarrationError(
            "MMS Persian TTS requires 'transformers' and 'torch' in the narration environment. "
            f"Details: {e}"
        ) from e
    try:
        model_id = "facebook/mms-tts-fas"
        model = VitsModel.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return model, tokenizer
    except Exception as e:
        raise NarrationError(
            "Failed to load MMS Persian TTS model (facebook/mms-tts-fas). "
            "Ensure the model is cached/downloaded and dependencies are available. "
            f"Details: {e}"
        ) from e


def _synthesize_with_mms_fa(mms: tuple, text: str, out_path: Path) -> None:
    """Synthesize Persian text with MMS-TTS and write 16-bit PCM WAV."""
    try:
        import numpy as np  # type: ignore
        import torch  # type: ignore
    except Exception as e:
        raise NarrationError(f"MMS synthesis deps missing (numpy/torch): {e}") from e

    model, tokenizer = mms
    out_path.parent.mkdir(parents=True, exist_ok=True)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform
    wav = output.squeeze().detach().cpu().numpy()
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    wav = np.clip(wav, -1.0, 1.0)
    wav_i16 = (wav * 32767.0).astype(np.int16)
    sr = int(getattr(getattr(model, "config", None), "sampling_rate", 16000) or 16000)

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with wave.open(str(tmp), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(wav_i16.tobytes())
    os.replace(tmp, out_path)


def _validate_chunk_audio(
    path: Path,
    expected_sr: Optional[int],
    expected_channels: Optional[int],
    expected_sampwidth: Optional[int] = None,
) -> Tuple[int, int, int, int]:
    """Return (sample_rate, channels, frames, sampwidth) for a valid chunk WAV."""
    if not path.exists() or path.stat().st_size == 0:
        raise NarrationError(f"Chunk audio missing or empty: {path}")
    try:
        with wave.open(str(path), "rb") as wf:
            sr = int(wf.getframerate())
            ch = int(wf.getnchannels())
            n = int(wf.getnframes())
            sampwidth = int(wf.getsampwidth())
            comptype = wf.getcomptype()
        if n <= 0:
            raise NarrationError(f"Chunk audio has no frames: {path}")
        if comptype != "NONE":
            raise NarrationError(
                f"Unsupported compressed WAV chunk format ({comptype}): {path}"
            )
        if sampwidth not in (1, 2, 3, 4):
            raise NarrationError(
                f"Unsupported WAV sample width ({sampwidth} bytes): {path}"
            )
        if expected_sr is not None and sr != expected_sr:
            raise NarrationError(
                f"Sample rate mismatch while merging narration: expected {expected_sr} Hz, got {sr} Hz in {path.name}. "
                "Please regenerate with a consistent backend/model sample rate."
            )
        if expected_channels is not None and ch != expected_channels:
            raise NarrationError(
                f"Channel count mismatch while merging narration: expected {expected_channels}, got {ch} in {path.name}."
            )
        if expected_sampwidth is not None and sampwidth != expected_sampwidth:
            raise NarrationError(
                f"Sample width mismatch while merging narration: expected {expected_sampwidth} bytes, got {sampwidth} bytes in {path.name}. "
                "Please regenerate chunks with a consistent audio format."
            )
        return sr, ch, n, sampwidth
    except NarrationError:
        raise
    except wave.Error as e:
        raise NarrationError(f"Invalid WAV chunk produced ({path.name}): {e}") from e
    except Exception as e:
        raise NarrationError(f"Failed to read chunk audio ({path.name}): {e}") from e


def _merge_chunks_streaming(
    chunk_paths: List[Path],
    final_path: Path,
    pause_seconds: float,
    progress: Optional[Callable[[str], None]] = None,
) -> Tuple[int, int, int]:
    """Stream-merge WAV chunks to disk with bounded memory. Returns (sr, channels, total_frames)."""
    if not chunk_paths:
        raise NarrationError("No chunk audio files to merge.")

    first_sr, first_ch, _n, first_sw = _validate_chunk_audio(
        chunk_paths[0], None, None, None
    )
    for cp in chunk_paths[1:]:
        _validate_chunk_audio(cp, first_sr, first_ch, first_sw)

    pause_frames = max(0, int(round(float(pause_seconds) * first_sr)))
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = final_path.with_suffix(final_path.suffix + ".merging")
    if tmp_out.exists():
        try:
            tmp_out.unlink()
        except Exception:
            pass

    total_frames = 0
    block_frames = 65536
    with wave.open(str(tmp_out), "wb") as out_wf:
        out_wf.setnchannels(first_ch)
        out_wf.setsampwidth(first_sw)
        out_wf.setframerate(first_sr)

        for i, cp in enumerate(chunk_paths, 1):
            if progress:
                progress(f"Merging chunk {i}/{len(chunk_paths)}...")
            with wave.open(str(cp), "rb") as in_wf:
                in_sw = in_wf.getsampwidth()
                in_ch = in_wf.getnchannels()
                in_sr = in_wf.getframerate()
                if in_sw != first_sw or in_ch != first_ch or in_sr != first_sr:
                    raise NarrationError(
                        f"Chunk audio format mismatch during merge: {cp.name} "
                        f"(sr={in_sr}, ch={in_ch}, sw={in_sw}) != expected "
                        f"(sr={first_sr}, ch={first_ch}, sw={first_sw})."
                    )
                while True:
                    frames = in_wf.readframes(block_frames)
                    if not frames:
                        break
                    out_wf.writeframes(frames)
                    frame_count = len(frames) // max(1, in_sw * in_ch)
                    total_frames += frame_count
            if pause_frames > 0 and i < len(chunk_paths):
                silence = (b"\x00" * first_sw) * pause_frames * first_ch
                out_wf.writeframes(silence)
                total_frames += pause_frames

    os.replace(tmp_out, final_path)
    return first_sr, first_ch, total_frames


def _estimate_total_chars(chunks: List[str]) -> int:
    return sum(len(c) for c in chunks)


def _emit_progress(progress: Optional[Callable[[str], None]], msg: str) -> None:
    if progress:
        progress(msg)


def get_narration_env_python(app_root: Path) -> Optional[Path]:
    """Return isolated narration feature env python if present (portable per-copy layout)."""
    root = Path(app_root)
    candidates = [
        root / "DO NOT TOUCH" / "envs" / "narration" / "Scripts" / "python.exe",
        root / "DO NOT TOUCH" / "envs" / "narration" / "bin" / "python",
        root / "envs" / "narration" / "Scripts" / "python.exe",
        root / "envs" / "narration" / "bin" / "python",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _probe_backends_in_python(py_exe: Path) -> Dict[str, Any]:
    code = (
        "import importlib.util, json\n"
        "def has(n):\n"
        "  try: return importlib.util.find_spec(n) is not None\n"
        "  except Exception: return False\n"
        "f5 = has('f5_tts')\n"
        "mms = has('transformers') and has('torch')\n"
        "print(json.dumps({'f5_tts': f5, 'mms_fa': mms}))\n"
    )
    try:
        import subprocess

        r = subprocess.run(
            [str(py_exe), "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        if r.returncode == 0:
            line = (r.stdout or "").strip().splitlines()[-1]
            data = json.loads(line)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {"f5_tts": False, "mms_fa": False}


def check_narration_backend(app_root: Optional[Path] = None) -> Tuple[bool, str, str]:
    """Backward-compatible backend check.

    Prefers f5_tts (voice cloning) for legacy callers, but also considers mms_fa
    and isolated narration env. Returns (available_any, detail, mode) where mode
    is 'current', 'env', or 'none'.
    """
    caps = check_narration_backends(app_root)
    for key in ("f5_tts", "mms_fa"):
        info = caps.get(key, {})
        if info.get("available") and info.get("mode") == "current":
            return True, str(info.get("detail", "available")), "current"
    for key in ("f5_tts", "mms_fa"):
        info = caps.get(key, {})
        if info.get("available") and info.get("mode") == "env":
            return True, str(info.get("detail", "available in isolated env")), "env"
    details = []
    for key in ("f5_tts", "mms_fa"):
        info = caps.get(key, {})
        if info.get("detail"):
            details.append(f"{key}: {info.get('detail')}")
    detail = "; ".join(details) if details else "No narration backend available."
    return False, detail, "none"


def check_narration_backends(
    app_root: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """Return capability map for supported narration backends."""
    caps: Dict[str, Dict[str, Any]] = {}
    f5_ok, f5_detail = check_f5_tts_available()
    caps["f5_tts"] = {
        "available": bool(f5_ok),
        "mode": "current" if f5_ok else "none",
        "detail": f5_detail,
        "label": "F5-TTS voice cloning",
        "supports_voice_clone": True,
        "best_for": "English/Chinese and other F5-supported languages",
    }
    mms_ok, mms_detail = check_mms_fa_available()
    caps["mms_fa"] = {
        "available": bool(mms_ok),
        "mode": "current" if mms_ok else "none",
        "detail": mms_detail,
        "label": "MMS Persian (accurate)",
        "supports_voice_clone": False,
        "best_for": "Persian (fa) narration with accurate text rendering",
    }

    if app_root is not None:
        env_py = get_narration_env_python(app_root)
        if env_py is not None:
            env_caps = _probe_backends_in_python(env_py)
            for key in ("f5_tts", "mms_fa"):
                if caps[key].get("available"):
                    continue
                if env_caps.get(key):
                    caps[key]["available"] = True
                    caps[key]["mode"] = "env"
                    caps[key]["detail"] = (
                        f"Detected in isolated narration env ({env_py})."
                    )
                    caps[key]["env_python"] = str(env_py)
                else:
                    if caps[key]["mode"] == "none":
                        caps[key]["detail"] = (
                            caps[key]["detail"]
                            + f" Isolated env present ({env_py}) but backend missing."
                        )
            caps["_env_python"] = {"path": str(env_py), "exists": True}
        else:
            caps["_env_python"] = {"path": None, "exists": False}
    return caps


def _select_execution_backend(
    requested_backend: str,
    script_text: str,
    caps: Dict[str, Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    req = (requested_backend or DEFAULT_BACKEND).strip().lower()
    if req == "auto":
        if detect_persian_script(script_text):
            if caps.get("mms_fa", {}).get("available"):
                return "mms_fa", caps["mms_fa"]
            if caps.get("f5_tts", {}).get("available"):
                return "f5_tts", caps["f5_tts"]
        else:
            if caps.get("f5_tts", {}).get("available"):
                return "f5_tts", caps["f5_tts"]
            if caps.get("mms_fa", {}).get("available"):
                return "mms_fa", caps["mms_fa"]
        raise NarrationError(
            "No narration backend is available. Install/configure f5-tts and/or MMS Persian dependencies in the narration env."
        )

    resolved = resolve_backend(req, script_text)
    info = caps.get(resolved, {})
    if not info.get("available"):
        detail = info.get("detail", "unavailable")
        raise NarrationError(
            f"Requested narration backend '{resolved}' is not available in this environment. ({detail})"
        )
    return resolved, info


def generate_narration_subprocess(
    app_root: Path,
    script_path: Path,
    reference_voice_path: Path,
    output_dir: Path,
    ref_text: str = "",
    model_name: Optional[str] = None,
    max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
    pause_seconds: float = DEFAULT_PAUSE_SECONDS,
    progress: Optional[Callable[[str], None]] = None,
    keep_chunks: bool = True,
    resume: bool = True,
    job_name: Optional[str] = None,
    timeout: Optional[int] = None,
    backend: str = DEFAULT_BACKEND,
) -> Dict[str, Any]:
    """Run narration generation in isolated narration env worker process (if available)."""
    env_py = get_narration_env_python(app_root)
    if env_py is None:
        raise NarrationError(
            "Isolated narration environment not found (DO NOT TOUCH/envs/narration)."
        )

    worker = Path(__file__).resolve()
    cmd = [
        str(env_py),
        str(worker),
        "--worker",
        "--script",
        str(script_path),
        "--voice",
        str(reference_voice_path),
        "--output-dir",
        str(output_dir),
        "--ref-text",
        ref_text or "",
        "--max-chars",
        str(int(max_chars_per_chunk)),
        "--pause",
        str(float(pause_seconds)),
        "--backend",
        str(backend or DEFAULT_BACKEND),
    ]
    if model_name:
        cmd += ["--model-name", str(model_name)]
    if job_name:
        cmd += ["--job-name", str(job_name)]
    cmd += ["--keep-chunks" if keep_chunks else "--no-keep-chunks"]
    cmd += ["--resume" if resume else "--no-resume"]

    _emit_progress(progress, f"Launching narration worker in isolated env: {env_py}")
    try:
        import subprocess

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except Exception as e:
        raise NarrationError(f"Failed to launch narration worker: {e}") from e

    final_result: Optional[Dict[str, Any]] = None
    final_error: Optional[str] = None
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except Exception:
                _emit_progress(progress, line)
                continue
            etype = evt.get("type")
            if etype == "progress":
                _emit_progress(progress, str(evt.get("message", "")))
            elif etype == "result":
                final_result = (
                    evt.get("result") if isinstance(evt.get("result"), dict) else evt
                )
            elif etype == "error":
                final_error = str(
                    evt.get("message") or evt.get("error") or "Narration worker failed."
                )
            else:
                _emit_progress(progress, line)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
        raise NarrationError("Narration worker timed out.")
    except Exception as e:
        try:
            proc.kill()
        except Exception:
            pass
        raise NarrationError(f"Narration worker communication failed: {e}") from e

    if final_error:
        raise NarrationError(final_error)
    if proc.returncode != 0 and final_result is None:
        raise NarrationError(f"Narration worker exited with code {proc.returncode}.")
    if not final_result:
        raise NarrationError("Narration worker finished without returning a result.")
    return final_result


def generate_narration(
    script_path: Path,
    reference_voice_path: Path,
    output_dir: Path,
    ref_text: str = "",
    model_name: Optional[str] = None,
    max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
    pause_seconds: float = DEFAULT_PAUSE_SECONDS,
    progress: Optional[Callable[[str], None]] = None,
    keep_chunks: bool = True,
    resume: bool = True,
    job_name: Optional[str] = None,
    backend: str = DEFAULT_BACKEND,
) -> Dict[str, Any]:
    """Generate narration audio from text.

    Backends:
      - f5_tts: voice-cloning synthesis using reference voice sample
      - mms_fa: accurate Persian TTS (fixed voice; ignores reference voice for synthesis)
      - auto: chooses mms_fa for Persian-script text when available, else f5_tts
    """
    script_path = Path(script_path)
    reference_voice_path = Path(reference_voice_path)
    output_dir = Path(output_dir)

    if not script_path.exists():
        raise NarrationError(f"Script not found: {script_path}")
    if reference_voice_path and not reference_voice_path.exists():
        raise NarrationError(
            f"Reference voice sample not found: {reference_voice_path}"
        )
    if (
        reference_voice_path
        and reference_voice_path.exists()
        and reference_voice_path.suffix.lower() not in AUDIO_EXTS
    ):
        raise NarrationError(
            "Reference voice must be an audio file (.wav/.mp3/.flac/.m4a/.ogg)."
        )

    max_chars_per_chunk = max(
        120, int(max_chars_per_chunk or DEFAULT_MAX_CHARS_PER_CHUNK)
    )
    try:
        pause_seconds = float(pause_seconds)
    except Exception:
        pause_seconds = DEFAULT_PAUSE_SECONDS
    pause_seconds = max(0.0, min(5.0, pause_seconds))

    text = _read_script_text(script_path)
    caps = check_narration_backends(None)
    selected_backend, _backend_info = _select_execution_backend(backend, text, caps)

    if selected_backend == "f5_tts":
        ok, detail = check_f5_tts_available()
        if not ok:
            raise NarrationError(
                f"F5-TTS backend selected but not available in this environment. ({detail})"
            )
    elif selected_backend == "mms_fa":
        ok, detail = check_mms_fa_available()
        if not ok:
            raise NarrationError(
                f"MMS Persian backend selected but dependencies are not available. ({detail})"
            )

    chunks = _chunk_text(text, max_chars=max_chars_per_chunk)
    if not chunks:
        raise NarrationError("No text chunks could be produced from the script.")

    output_dir.mkdir(parents=True, exist_ok=True)
    base = _slugify(script_path.stem)
    stamp = _timestamp()

    job_dir_name = _slugify(job_name) if job_name else f"{base}_{stamp}_job"
    job_dir = output_dir / job_dir_name
    if job_dir.exists() and not resume:
        job_dir = output_dir / f"{base}_{stamp}_{uuid.uuid4().hex[:4]}_job"
    chunks_dir = job_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = job_dir / JOB_MANIFEST_NAME
    existing_manifest = _load_json(manifest_path) or {}

    script_stat = script_path.stat()
    voice_stat = reference_voice_path.stat() if reference_voice_path.exists() else None
    script_fingerprint = {
        "path": str(script_path.resolve()),
        "size": int(script_stat.st_size),
        "mtime_ns": int(
            getattr(script_stat, "st_mtime_ns", int(script_stat.st_mtime * 1e9))
        ),
        "text_sha256": _sha256_text(text),
    }
    voice_fingerprint = {
        "path": str(reference_voice_path.resolve())
        if reference_voice_path.exists()
        else "",
        "size": int(voice_stat.st_size) if voice_stat else 0,
        "mtime_ns": int(
            getattr(voice_stat, "st_mtime_ns", int(voice_stat.st_mtime * 1e9))
        )
        if voice_stat
        else 0,
    }

    effective_ref_text = ref_text or ""
    if resume and existing_manifest:
        old_script = existing_manifest.get("script_fingerprint") or {}
        old_voice = existing_manifest.get("voice_fingerprint") or {}
        old_params = existing_manifest.get("params") or {}
        if not str(effective_ref_text).strip():
            effective_ref_text = str(old_params.get("ref_text", "") or "")
        mismatch = []
        if (
            old_script.get("text_sha256")
            and old_script.get("text_sha256") != script_fingerprint["text_sha256"]
        ):
            mismatch.append("script text changed")
        if old_voice.get("path") and old_voice.get("path") != voice_fingerprint["path"]:
            mismatch.append("reference voice changed")
        if old_voice.get("size") and int(old_voice.get("size", -1)) != int(
            voice_fingerprint["size"]
        ):
            mismatch.append("reference voice file changed")
        if (
            old_params.get("max_chars_per_chunk")
            and int(old_params.get("max_chars_per_chunk")) != max_chars_per_chunk
        ):
            mismatch.append("chunk size changed")

        old_backend = old_params.get("backend") or existing_manifest.get("backend")
        if not old_backend:
            old_backend = "f5_tts"  # legacy manifests predate backend field
        if str(old_backend) != selected_backend:
            mismatch.append("backend changed")
        if mismatch:
            job_dir = output_dir / f"{base}_{stamp}_{uuid.uuid4().hex[:4]}_job"
            chunks_dir = job_dir / "chunks"
            chunks_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = job_dir / JOB_MANIFEST_NAME
            existing_manifest = {}
            effective_ref_text = ref_text or ""

    chunk_paths = [chunks_dir / f"chunk_{i:05d}.wav" for i in range(1, len(chunks) + 1)]

    manifest: Dict[str, Any] = {
        "version": 1,
        "created_at": existing_manifest.get("created_at") or _now_iso(),
        "updated_at": _now_iso(),
        "status": "running",
        "job_dir": str(job_dir),
        "script": str(script_path),
        "reference_voice": str(reference_voice_path)
        if reference_voice_path.exists()
        else "",
        "backend": selected_backend,
        "backend_requested": backend,
        "script_fingerprint": script_fingerprint,
        "voice_fingerprint": voice_fingerprint,
        "params": {
            "ref_text_provided": bool(str(effective_ref_text).strip()),
            "ref_text": str(effective_ref_text or ""),
            "model_name": model_name,
            "max_chars_per_chunk": max_chars_per_chunk,
            "pause_seconds": pause_seconds,
            "keep_chunks": bool(keep_chunks),
            "resume": bool(resume),
            "backend": selected_backend,
            "backend_requested": backend,
        },
        "progress": {
            "total_chunks": len(chunks),
            "completed_chunks": 0,
            "skipped_chunks": 0,
            "synthesized_chunks": 0,
            "total_chars": _estimate_total_chars(chunks),
        },
        "chunks": [
            {
                "index": i,
                "path": str(chunk_paths[i - 1]),
                "chars": len(chunks[i - 1]),
                "status": "pending",
            }
            for i in range(1, len(chunks) + 1)
        ],
        "output": None,
        "error": None,
        "notes": [],
    }

    def emit(msg: str) -> None:
        _emit_progress(progress, msg)

    def save_manifest() -> None:
        manifest["updated_at"] = _now_iso()
        _atomic_write_json(manifest_path, manifest)

    pre_completed = 0
    for i, cp in enumerate(chunk_paths, 1):
        if resume and cp.exists() and cp.stat().st_size > 0:
            manifest["chunks"][i - 1]["status"] = "completed"
            pre_completed += 1
    manifest["progress"]["completed_chunks"] = pre_completed
    manifest["progress"]["skipped_chunks"] = pre_completed
    save_manifest()

    emit(
        f"Prepared {len(chunks)} chunk(s) for narration "
        f"(backend={selected_backend}, ~{manifest['progress']['total_chars']} chars, chunk size {max_chars_per_chunk}, pause {pause_seconds:.2f}s)."
    )
    if selected_backend == "mms_fa":
        emit(
            "MMS Persian mode: accurate Persian text synthesis using fixed MMS voice "
            "(reference voice is recorded for consent/traceability but not cloned in this backend)."
        )
    if pre_completed:
        emit(
            f"Resuming job: {pre_completed}/{len(chunks)} chunk(s) already completed in {job_dir.name}."
        )

    tts_engine = None
    t_start = time.time()
    synthesized_now = 0
    try:
        for i, chunk in enumerate(chunks, 1):
            cp = chunk_paths[i - 1]
            chunk_meta = manifest["chunks"][i - 1]

            if resume and cp.exists() and cp.stat().st_size > 0:
                chunk_meta["status"] = "completed"
                continue

            if tts_engine is None:
                if selected_backend == "f5_tts":
                    emit("Loading F5-TTS backend...")
                    tts_engine = _load_f5_tts(model_name=model_name)
                elif selected_backend == "mms_fa":
                    emit("Loading MMS Persian (facebook/mms-tts-fas) backend...")
                    tts_engine = _load_mms_fa()
                else:
                    raise NarrationError(
                        f"Unsupported narration backend: {selected_backend}"
                    )

            done_before = sum(
                1 for c in manifest["chunks"] if c.get("status") == "completed"
            )
            pct = (done_before / max(1, len(chunks))) * 100.0
            emit(
                f"Synthesizing chunk {i}/{len(chunks)} ({pct:.1f}% complete, {len(chunk)} chars)..."
            )

            chunk_meta["status"] = "in_progress"
            save_manifest()

            t_chunk = time.time()
            if selected_backend == "f5_tts":
                _synthesize_with_f5(
                    tts_engine, chunk, reference_voice_path, cp, ref_text=effective_ref_text
                )
            elif selected_backend == "mms_fa":
                _synthesize_with_mms_fa(tts_engine, chunk, cp)
            else:
                raise NarrationError(
                    f"Unsupported narration backend: {selected_backend}"
                )

            if not cp.exists() or cp.stat().st_size == 0:
                raise NarrationError(
                    f"Synthesis completed but chunk file is missing/empty: {cp.name}"
                )

            chunk_meta["status"] = "completed"
            chunk_meta["seconds"] = round(time.time() - t_chunk, 3)
            synthesized_now += 1
            manifest["progress"]["synthesized_chunks"] = synthesized_now
            manifest["progress"]["completed_chunks"] = sum(
                1 for c in manifest["chunks"] if c.get("status") == "completed"
            )
            save_manifest()

            done_now = manifest["progress"]["completed_chunks"]
            if synthesized_now >= 1 and done_now < len(chunks):
                elapsed = max(0.001, time.time() - t_start)
                rate = synthesized_now / elapsed
                remaining = len(chunks) - done_now
                eta_s = remaining / max(rate, 1e-6)
                emit(
                    f"Progress: {done_now}/{len(chunks)} chunks done. ETA ~{_fmt_hms(eta_s)} (rough)."
                )

    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["error"] = (
            "Interrupted by user. Re-run with resume enabled to continue from completed chunks."
        )
        save_manifest()
        raise NarrationError(
            f"Narration interrupted. Progress saved in job folder: {job_dir}. "
            "Run again with resume to continue from completed chunks."
        )
    except Exception as e:
        manifest["status"] = "failed"
        manifest["error"] = str(e)
        save_manifest()
        raise

    final_path = job_dir / f"{base}_narration.wav"
    emit(
        f"All chunks synthesized ({len(chunks)}/{len(chunks)}). Merging to final audio (streaming)..."
    )
    sr, ch, total_frames = _merge_chunks_streaming(
        chunk_paths, final_path, pause_seconds=pause_seconds, progress=emit
    )

    duration_s = (float(total_frames) / float(sr)) if sr else 0.0
    manifest["status"] = "completed"
    manifest["output"] = {
        "path": str(final_path),
        "sample_rate": sr,
        "channels": ch,
        "frames": total_frames,
        "duration_seconds": round(duration_s, 3),
        "duration_hms": _fmt_hms(duration_s),
    }
    if selected_backend == "mms_fa" and reference_voice_path.exists():
        manifest["notes"].append(
            "MMS Persian backend generated accurate Persian speech with a fixed MMS voice. "
            "Reference voice sample was recorded for consent/traceability but not cloned in this backend."
        )
    save_manifest()

    if not keep_chunks:
        emit("Cleaning up chunk files (keep_chunks=False)...")
        for cp in chunk_paths:
            try:
                if cp.exists():
                    cp.unlink()
            except Exception:
                pass
        try:
            chunks_dir.rmdir()
        except OSError:
            pass

    emit(
        f"Done. Output duration: {manifest['output']['duration_hms']} ({duration_s:.1f}s)."
    )

    return {
        "script": str(script_path),
        "reference_voice": str(reference_voice_path)
        if reference_voice_path.exists()
        else "",
        "output": str(final_path),
        "job_dir": str(job_dir),
        "manifest": str(manifest_path),
        "chunks": len(chunks),
        "chunks_synthesized_now": synthesized_now,
        "chunks_reused": pre_completed,
        "duration_seconds": round(duration_s, 3),
        "sample_rate": sr,
        "channels": ch,
        "kept_chunks": bool(keep_chunks),
        "backend": selected_backend,
        "backend_requested": backend,
        "created_at": _now_iso(),
    }


def _worker_main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Narration worker (isolated env)")
    ap.add_argument("--worker", action="store_true", help="Run in worker JSONL mode")
    ap.add_argument("--script", required=True)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--ref-text", default="")
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS_PER_CHUNK)
    ap.add_argument("--pause", type=float, default=DEFAULT_PAUSE_SECONDS)
    ap.add_argument("--job-name", default=None)
    ap.add_argument(
        "--backend", default=DEFAULT_BACKEND, choices=list(SUPPORTED_BACKENDS)
    )
    ap.add_argument(
        "--keep-chunks", dest="keep_chunks", action="store_true", default=True
    )
    ap.add_argument("--no-keep-chunks", dest="keep_chunks", action="store_false")
    ap.add_argument("--resume", dest="resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args(argv)

    def jprint(obj: Dict[str, Any]) -> None:
        print(json.dumps(obj, ensure_ascii=False), flush=True)

    def prog(msg: str) -> None:
        jprint({"type": "progress", "message": msg})

    try:
        result = generate_narration(
            script_path=Path(args.script),
            reference_voice_path=Path(args.voice),
            output_dir=Path(args.output_dir),
            ref_text=args.ref_text or "",
            model_name=args.model_name,
            max_chars_per_chunk=args.max_chars,
            pause_seconds=args.pause,
            progress=prog if args.worker else (lambda m: print(m, flush=True)),
            keep_chunks=bool(args.keep_chunks),
            resume=bool(args.resume),
            job_name=args.job_name,
            backend=args.backend,
        )
        if args.worker:
            jprint({"type": "result", "result": result})
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except NarrationError as e:
        if args.worker:
            jprint({"type": "error", "message": str(e)})
        else:
            print(f"ERROR: {e}")
        return 2
    except Exception as e:
        msg = f"Unexpected narration worker error: {e}"
        if args.worker:
            jprint({"type": "error", "message": msg})
        else:
            print(f"ERROR: {msg}")
        return 1


if __name__ == "__main__":
    raise SystemExit(_worker_main())
