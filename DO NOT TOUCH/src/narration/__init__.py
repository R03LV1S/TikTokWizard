"""Optional narration/voiceover module (separate from karaoke alignment/render)."""

from .generator import (
    NarrationError,
    check_f5_tts_available,
    check_mms_fa_available,
    check_narration_backend,
    check_narration_backends,
    detect_persian_script,
    ensure_narration_dirs,
    generate_narration,
    generate_narration_subprocess,
    get_narration_env_python,
    get_narration_paths,
    list_narration_jobs,
    list_scripts,
    list_voice_samples,
    resolve_backend,
)

__all__ = [
    "NarrationError",
    "check_f5_tts_available",
    "check_mms_fa_available",
    "check_narration_backend",
    "check_narration_backends",
    "detect_persian_script",
    "ensure_narration_dirs",
    "generate_narration",
    "generate_narration_subprocess",
    "get_narration_env_python",
    "get_narration_paths",
    "list_narration_jobs",
    "list_scripts",
    "list_voice_samples",
    "resolve_backend",
]
