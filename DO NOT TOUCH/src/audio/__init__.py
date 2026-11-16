"""Audio module for TikTok Wizard (v4.1)."""

from __future__ import annotations

from .dsp import (
    compute_f0_pitch_tracking,
    compute_onset_envelope,
    compute_rms_envelope,
    detect_echo_decay_trains,
    detect_stutter_spikes,
    extract_beat_grid_16th,
    normalize_lufs,
)
from .streams import (
    AudioStream,
    decode_audio_mono,
    decode_audio_stereo_and_mono,
    mix_streams,
    stereo_azimuth_decompose,
    true_duration_sec,
)

__all__ = [
    "AudioStream",
    "compute_f0_pitch_tracking",
    "compute_onset_envelope",
    "compute_rms_envelope",
    "decode_audio_mono",
    "decode_audio_stereo_and_mono",
    "detect_echo_decay_trains",
    "detect_stutter_spikes",
    "extract_beat_grid_16th",
    "mix_streams",
    "normalize_lufs",
    "stereo_azimuth_decompose",
    "true_duration_sec",
]
