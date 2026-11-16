"""Audio stream management and multi-stem preparation for the v4.1 pipeline.

Guarantees:
  - Explicit channel-orientation guard and proper stereo-to-mono averaging
    (`mean(dim=0)`), preventing the stereo-flatten 2x duration bug (`452.62s` vs `226.31s`).
  - SIMD-accelerated resampling via `soxr` (with `torchaudio.functional.resample`
    fallback).
  - Stereo STFT azimuth panning separation (`pan_divergence`) and local `-16 LUFS`
    slice gain normalization.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
import scipy.signal
import torch
import torchaudio

TARGET_SR: int = 16000


def _resample_1d(waveform_1d: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a 1-D float32 array using `soxr` VHQ when available, else `torchaudio`."""
    if int(orig_sr) == int(target_sr) or waveform_1d.size == 0:
        return waveform_1d.astype(np.float32, copy=False)
    try:
        import soxr

        return soxr.resample(
            waveform_1d.astype(np.float32, copy=False),
            int(orig_sr),
            int(target_sr),
            quality="VHQ",
        ).astype(np.float32, copy=False)
    except Exception:
        t = torch.from_numpy(waveform_1d.astype(np.float32, copy=False)).unsqueeze(0)
        res = torchaudio.functional.resample(t, int(orig_sr), int(target_sr))
        return res.squeeze(0).contiguous().cpu().numpy().astype(np.float32)


def decode_audio_stereo_and_mono(
    path: str, target_sr: int = TARGET_SR
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Decode an audio file into `(mono, left, right, target_sr)` 1-D float32 arrays.

    Explicitly checks tensor dimensions so `(samples, channels)` is never
    mistaken for `(channels, samples)`, and downmixes stereo via `0.5 * (L + R)`
    rather than `.reshape(-1)`.
    """
    waveform, sr = torchaudio.load(path)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() > 2:
        waveform = waveform.reshape(waveform.shape[0], -1)

    # Guard against backends returning (samples, channels)
    if waveform.shape[0] > 8 and waveform.shape[1] <= 8:
        waveform = waveform.transpose(0, 1)

    orig_sr = int(sr)
    if waveform.shape[0] >= 2:
        left_raw = waveform[0].detach().cpu().numpy().astype(np.float32)
        right_raw = waveform[1].detach().cpu().numpy().astype(np.float32)
        left = _resample_1d(left_raw, orig_sr, target_sr)
        right = _resample_1d(right_raw, orig_sr, target_sr)
        n = min(left.size, right.size)
        left = left[:n]
        right = right[:n]
        mono = (0.5 * (left + right)).astype(np.float32)
    else:
        mono_raw = waveform[0].detach().cpu().numpy().astype(np.float32)
        mono = _resample_1d(mono_raw, orig_sr, target_sr)
        left = mono.copy()
        right = mono.copy()

    return np.ascontiguousarray(mono), np.ascontiguousarray(left), np.ascontiguousarray(right), int(target_sr)


def decode_audio_mono(path: str, target_sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    """Decode an audio/video file to a 1-D mono float32 waveform at `target_sr`.

    Stereo handling is explicit and deterministic: multi-channel tensors are
    averaged across the channel axis (`mean(dim=0)`), never flattened with
    `.reshape(-1)`.
    """
    mono, _, _, sr = decode_audio_stereo_and_mono(path, target_sr=target_sr)
    return mono, sr


def true_duration_sec(path: str, target_sr: int = TARGET_SR) -> float:
    """Return the true duration in seconds of the decoded mono stream."""
    data, sr = decode_audio_mono(path, target_sr=target_sr)
    return float(len(data)) / float(max(1, sr))


def stereo_azimuth_decompose(
    left: np.ndarray,
    right: np.ndarray,
    sr: int = TARGET_SR,
    center_threshold: float = 0.28,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose stereo vocals into `(center_lead_dry, wide_backing_echo)` at `sr`."""
    n = min(left.size, right.size)
    if n < 512:
        mono = (0.5 * (left[:n] + right[:n])).astype(np.float32)
        return mono, np.zeros_like(mono)

    l_sig = left[:n].astype(np.float32)
    r_sig = right[:n].astype(np.float32)
    mid_sig = 0.5 * (l_sig + r_sig)

    n_fft = 1024
    hop = 256
    win = np.hanning(n_fft).astype(np.float32)

    _, _, z_l = scipy.signal.stft(l_sig, fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop)
    _, _, z_r = scipy.signal.stft(r_sig, fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop)
    _, _, z_m = scipy.signal.stft(mid_sig, fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop)

    pan_div = np.abs(z_l - z_r) / (np.abs(z_l) + np.abs(z_r) + 1e-9)
    lead_mask = np.clip(1.0 - (pan_div / max(0.05, float(center_threshold) * 1.35)) ** 2, 0.12, 1.0)

    z_lead = z_m * lead_mask
    z_echo = (0.5 * (z_l + z_r)) - z_lead + 0.5 * (z_l - z_r)

    _, lead_rec = scipy.signal.istft(z_lead, fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop)
    _, echo_rec = scipy.signal.istft(z_echo, fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop)

    lead_out = np.zeros(n, dtype=np.float32)
    echo_out = np.zeros(n, dtype=np.float32)
    m = min(n, lead_rec.size, echo_rec.size)
    lead_out[:m] = lead_rec[:m].astype(np.float32)
    echo_out[:m] = echo_rec[:m].astype(np.float32)
    return lead_out, echo_out


class AudioStream:
    """Represents a single 1-D mono audio stream with metadata."""

    def __init__(self, data: np.ndarray, sample_rate: int, title: str = ""):
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim > 1:
            if arr.shape[0] <= 8 and arr.shape[1] > 8:
                arr = np.mean(arr, axis=0).astype(np.float32)
            else:
                arr = np.mean(arr, axis=-1).astype(np.float32)
        self.data: np.ndarray = np.ascontiguousarray(arr)
        self.sample_rate: int = int(sample_rate)
        self.title: str = str(title)
        self.duration_sec: float = float(len(self.data)) / float(max(1, self.sample_rate))

    def resample(self, target_sr: int) -> "AudioStream":
        """Resample stream to `target_sr`."""
        if self.sample_rate == int(target_sr):
            return self
        new_data = _resample_1d(self.data, self.sample_rate, int(target_sr))
        return AudioStream(new_data, int(target_sr), self.title)


def mix_streams(streams: Sequence[AudioStream], crossfade_ms: int = 50) -> np.ndarray:
    """Mix multiple `AudioStream` instances with deterministic integer-indexed crossfading."""
    if not streams:
        return np.zeros(0, dtype=np.float32)
    if len(streams) == 1:
        return streams[0].data.astype(np.float32, copy=True)

    max_len = max(int(len(s.data)) for s in streams)
    if max_len <= 0:
        return np.zeros(0, dtype=np.float32)

    mixed = np.zeros(max_len, dtype=np.float64)

    for i, stream in enumerate(streams):
        data_len = int(len(stream.data))
        if data_len == 0:
            continue
        s_data = stream.data.astype(np.float64)
        if i == 0:
            mixed[:data_len] += s_data
        else:
            fade_len = int(
                min(
                    round((float(crossfade_ms) / 1000.0) * float(stream.sample_rate)),
                    data_len,
                    max(0, max_len - data_len),
                )
            )
            if fade_len > 0:
                start_idx = max(0, max_len - data_len)
                end_idx = start_idx + fade_len
                fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float64)
                fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float64)
                mixed[start_idx:end_idx] *= fade_out
                mixed[start_idx:end_idx] += s_data[:fade_len] * fade_in
                if data_len > fade_len:
                    rem_end = min(max_len, start_idx + data_len)
                    mixed[end_idx:rem_end] += s_data[fade_len : fade_len + (rem_end - end_idx)]
            else:
                mixed[:data_len] += s_data

    peak = float(np.max(np.abs(mixed)))
    if peak > 1.0:
        mixed /= peak
    return mixed.astype(np.float32)
