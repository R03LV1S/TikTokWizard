"""Digital signal processing functions for the v4.1 audio alignment pipeline.

Provides:
  - Fast, mathematically accurate `-16 LUFS` loudness normalization (`normalize_lufs`)
    with `-1.0 dBFS` peak ceiling protection.
  - 10ms-hop Fundamental Frequency (`F0`) vocal pitch & voicing tracker
    (`compute_f0_pitch_tracking`) via FFT autocorrelation and parabolic peak
    interpolation across the human singing range (`80 Hz - 1000 Hz`).
  - 10ms-hop RMS energy, vocal-band onset strength, stutter spike detector,
    BPM-locked spectrogram echo-clone detector, and 16th-note beat-grid extractor.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.signal

TARGET_SR: int = 16000


def normalize_lufs(
    audio: np.ndarray,
    target_lufs: float = -16.0,
    peak_ceiling_db: float = -1.0,
) -> np.ndarray:
    """Normalize a waveform array to `target_lufs` (dBFS RMS equivalent) with peak ceiling.

    Fixes the sign inversion in `10 ** (target_lufs / 20.0)` and avoids O(N^2)
    full-signal Hilbert transforms on non-power-of-two array lengths.
    """
    arr = np.asarray(audio, dtype=np.float32)
    if arr.size == 0:
        return arr.copy()

    x = arr.astype(np.float64)
    # Gate out near-silent frames when estimating active vocal RMS
    sq = x * x
    mean_sq = float(np.mean(sq))
    if mean_sq <= 1e-12:
        return arr.copy()

    active_mask = sq >= (0.10 * mean_sq)
    rms = float(np.sqrt(np.mean(sq[active_mask]))) if np.any(active_mask) else float(np.sqrt(mean_sq))
    if rms <= 1e-9:
        return arr.copy()

    # Negative target_lufs (e.g. -16.0 LUFS) maps to 10^(-16/20) = 0.158489 target RMS
    eff_db = -abs(float(target_lufs)) if float(target_lufs) != 0.0 else -16.0
    target_rms = math.pow(10.0, eff_db / 20.0)
    gain = min(25.0, target_rms / rms)
    out = x * gain

    peak_limit = math.pow(10.0, -abs(float(peak_ceiling_db)) / 20.0)
    peak_actual = float(np.max(np.abs(out)))
    if peak_actual > peak_limit and peak_actual > 1e-9:
        out *= peak_limit / peak_actual

    return out.astype(np.float32)


def compute_rms_envelope(
    audio: np.ndarray, sr: int = TARGET_SR, hop_sec: float = 0.01
) -> np.ndarray:
    """Compute a 10ms-step sliding RMS envelope."""
    arr = np.asarray(audio, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.zeros(1, dtype=np.float32)

    hop = max(1, int(round(hop_sec * sr)))
    win = max(hop, int(round(0.025 * sr)))
    sq = arr * arr
    kernel = np.ones(win, dtype=np.float64) / float(win)
    smooth = np.convolve(sq, kernel, mode="same")
    return np.sqrt(np.maximum(0.0, smooth[::hop])).astype(np.float32)


def compute_onset_envelope(
    audio: np.ndarray, sr: int = TARGET_SR, hop_sec: float = 0.01
) -> np.ndarray:
    """Compute half-wave rectified spectral flux onset strength at 10ms hop in the vocal band."""
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if arr.size < 256:
        return np.zeros(max(1, arr.size // max(1, int(hop_sec * sr))), dtype=np.float32)

    nyq = 0.5 * float(sr)
    lo = max(0.02, min(0.85, 300.0 / nyq))
    hi = max(lo + 0.05, min(0.98, 4500.0 / nyq))
    sos = scipy.signal.butter(3, [lo, hi], btype="bandpass", output="sos")
    band = scipy.signal.sosfiltfilt(sos, arr.astype(np.float64)).astype(np.float32)

    hop = max(1, int(round(hop_sec * sr)))
    n_fft = max(256, 2 * hop)
    win = np.hanning(n_fft).astype(np.float32)
    _, _, z = scipy.signal.stft(
        band, fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop
    )
    mag = np.log1p(10.0 * np.abs(z))
    diff = np.diff(mag, axis=-1, prepend=mag[:, :1])
    flux = np.mean(np.maximum(0.0, diff), axis=0).astype(np.float32)
    peak = float(np.max(flux))
    if peak > 1e-9:
        flux /= peak
    return flux


def compute_f0_pitch_tracking(
    audio: np.ndarray,
    sr: int = TARGET_SR,
    fmin_hz: float = 80.0,
    fmax_hz: float = 1000.0,
    hop_sec: float = 0.01,
) -> np.ndarray:
    """Track human singing fundamental frequency `F0` (Hz) at 10ms frame steps.

    Uses FFT-accelerated normalized autocorrelation over lags corresponding to
    `[fmin_hz, fmax_hz]` (excluding lag 0) with parabolic interpolation,
    returning `0.0` for unvoiced/silent frames and `f0_hz` for voiced frames.
    """
    arr = np.asarray(audio, dtype=np.float64).reshape(-1)
    frame_size = max(64, int(round(0.032 * sr)))  # 32ms window (512 samples @ 16kHz)
    hop_size = max(1, int(round(hop_sec * sr)))   # 10ms hop (160 samples @ 16kHz)

    if arr.size < frame_size:
        return np.zeros(1, dtype=np.float32)

    min_lag = max(2, int(math.floor(float(sr) / float(max(fmax_hz, fmin_hz + 10.0)))))
    max_lag = min(frame_size - 2, int(math.ceil(float(sr) / float(max(40.0, fmin_hz)))))
    if max_lag <= min_lag:
        return np.zeros(max(1, (arr.size - frame_size) // hop_size + 1), dtype=np.float32)

    fft_len = 1 << int(math.ceil(math.log2(2 * frame_size - 1)))
    win = np.hanning(frame_size).astype(np.float64)

    global_rms = float(np.sqrt(np.mean(arr * arr)))
    voicing_rms_floor = max(1e-4, 0.08 * global_rms)

    pitches: List[float] = []
    for start in range(0, arr.size - frame_size + 1, hop_size):
        frame = arr[start : start + frame_size]
        frame_centered = (frame - np.mean(frame)) * win
        energy = float(np.sum(frame_centered * frame_centered))
        frame_rms = math.sqrt(energy / float(frame_size))

        if frame_rms < voicing_rms_floor or energy <= 1e-12:
            pitches.append(0.0)
            continue

        spec = np.fft.rfft(frame_centered, n=fft_len)
        acf = np.fft.irfft(spec * np.conj(spec), n=fft_len)[: max_lag + 2] / energy

        search_region = acf[min_lag : max_lag + 1]
        if search_region.size == 0:
            pitches.append(0.0)
            continue

        rel_idx = int(np.argmax(search_region))
        best_lag = min_lag + rel_idx
        peak_corr = float(acf[best_lag])

        # Voicing threshold: periodic human voice exhibits normalized ACF peak >= 0.32
        if peak_corr < 0.32:
            pitches.append(0.0)
            continue

        # Parabolic refinement around `best_lag`
        refined_lag = float(best_lag)
        if 1 <= best_lag < len(acf) - 1:
            y0 = float(acf[best_lag - 1])
            y1 = float(acf[best_lag])
            y2 = float(acf[best_lag + 1])
            denom = y0 - 2.0 * y1 + y2
            if abs(denom) > 1e-9:
                delta = 0.5 * (y0 - y2) / denom
                if -0.5 <= delta <= 0.5:
                    refined_lag = float(best_lag) + delta

        f0 = float(sr) / max(1.0, refined_lag)
        if fmin_hz <= f0 <= fmax_hz:
            pitches.append(f0)
        else:
            pitches.append(0.0)

    return np.asarray(pitches, dtype=np.float32)


def detect_stutter_spikes(
    onset_env: np.ndarray,
    word_start_sec: float,
    hard_min_sec: float = 0.0,
    hop_sec: float = 0.01,
) -> Tuple[float, bool]:
    """Check 40ms-220ms prior to `word_start_sec` for rapid consonant stutter spikes."""
    if onset_env is None or onset_env.size == 0:
        return word_start_sec, False

    scan_lo = max(hard_min_sec, word_start_sec - 0.22)
    scan_hi = max(scan_lo, word_start_sec - 0.03)
    i0 = max(0, int(round(scan_lo / hop_sec)))
    i1 = min(onset_env.size, int(round(scan_hi / hop_sec)))
    if i1 - i0 < 3:
        return word_start_sec, False

    seg = onset_env[i0:i1]
    thresh = max(0.25, float(np.median(onset_env)) * 2.5)
    peaks, _ = scipy.signal.find_peaks(seg, height=thresh, distance=max(1, int(0.04 / hop_sec)))
    if peaks.size >= 1:
        snapped = max(hard_min_sec, scan_lo + float(peaks[0]) * hop_sec)
        return snapped, True
    return word_start_sec, False


def extract_beat_grid_16th(
    audio: np.ndarray, sr: int = TARGET_SR
) -> Dict[str, Any]:
    """Extract track tempo (`bpm`), beat timestamps, and 16th-note subdivision grid."""
    dur = float(len(audio)) / float(max(1, sr))
    bpm = 120.0
    beat_times: List[float] = []

    try:
        import librosa

        tempo, beats = librosa.beat.beat_track(y=np.asarray(audio, dtype=np.float32), sr=sr)
        bpm_val = float(np.ndim(tempo) and np.asarray(tempo).reshape(-1)[0] or tempo)
        if 55.0 <= bpm_val <= 210.0:
            bpm = bpm_val
        beat_times = [float(t) for t in librosa.frames_to_time(beats, sr=sr)]
    except Exception:
        beat_times = []

    if len(beat_times) < 2:
        period = 60.0 / bpm
        beat_times = [round(k * period, 4) for k in range(int(dur / period) + 1)]

    subgrid_16th: List[float] = []
    for k in range(len(beat_times) - 1):
        b0 = beat_times[k]
        b1 = beat_times[k + 1]
        step = (b1 - b0) / 4.0
        for sub in range(4):
            subgrid_16th.append(round(b0 + sub * step, 4))
    if beat_times:
        subgrid_16th.append(round(beat_times[-1], 4))

    return {
        "bpm": round(bpm, 2),
        "beat_times": beat_times,
        "subgrid_16th": subgrid_16th,
    }


def detect_echo_decay_trains(
    rms_env: np.ndarray,
    burst_intervals: Sequence[Tuple[float, float]],
    bpm: float = 120.0,
    hop_sec: float = 0.01,
) -> List[bool]:
    """Identify which bursts in `burst_intervals` are decaying studio echo clones (`True`)."""
    n = len(burst_intervals)
    is_clone = [False] * n
    if n < 2 or rms_env is None or rms_env.size == 0:
        return is_clone

    beat_sec = 60.0 / max(40.0, min(220.0, float(bpm)))
    delay_candidates = [beat_sec * f for f in (0.25, 0.333, 0.5, 0.75, 1.0, 1.5, 2.0)]

    peaks: List[float] = []
    for st, en in burst_intervals:
        i0 = max(0, int(round(st / hop_sec)))
        i1 = min(rms_env.size, max(i0 + 1, int(round(en / hop_sec))))
        peaks.append(float(np.max(rms_env[i0:i1])) if i1 > i0 else 0.0)

    for k in range(1, n):
        dt = float(burst_intervals[k][0]) - float(burst_intervals[k - 1][0])
        near_grid = any(abs(dt - d) <= 0.06 for d in delay_candidates)
        decaying = peaks[k] < 0.82 * peaks[k - 1]
        if near_grid and decaying:
            is_clone[k] = True

    return is_clone
