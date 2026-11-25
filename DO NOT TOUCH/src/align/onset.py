# align/onset.py
"""Lightweight onset/voicing evidence for karaoke word placement.

Universal, signal-based features: spectral flux onset, RMS, F0 voicing ratio.
No song-specific rules, no absolute timestamp heuristics.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..audio.dsp import compute_f0_pitch_tracking, compute_onset_envelope, compute_rms_envelope


def compute_features(sig: np.ndarray, sr: int = 16000, hop_sec: float = 0.01) -> Dict[str, Any]:
    sig = np.asarray(sig, dtype=np.float32).reshape(-1)
    rms = compute_rms_envelope(sig, sr=sr, hop_sec=hop_sec)
    onset = compute_onset_envelope(sig, sr=sr, hop_sec=hop_sec)
    f0 = compute_f0_pitch_tracking(sig, sr=sr, hop_sec=hop_sec)
    n = max(len(rms), len(onset), len(f0), 1)
    def _fit(x):
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if len(x) == n: return x
        if len(x) == 0: return np.zeros(n, dtype=np.float32)
        xi = np.linspace(0, 1, len(x))
        xo = np.linspace(0, 1, n)
        return np.interp(xo, xi, x).astype(np.float32)
    rms = _fit(rms); onset = _fit(onset); f0 = _fit(f0)
    voicing = (f0 > 0.0).astype(np.float32)
    return {"rms": rms, "onset": onset, "voicing": voicing, "f0": f0, "hop_sec": float(hop_sec), "sr": int(sr), "n": n}


def _norm01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.percentile(x, 10)) if x.size else 0.0
    hi = float(np.percentile(x, 95)) if x.size else 1.0
    if hi - lo <= 1e-9:
        hi = lo + 1e-9
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def score_window(features: Dict[str, Any], t0: float, t1: float, min_dur: float = 0.12) -> Tuple[Optional[float], float, Dict[str, Any]]:
    hop = float(features.get("hop_sec", 0.01))
    n = int(features.get("n", 0))
    if n <= 0 or t1 <= t0:
        return None, -1.0, {}
    rms = _norm01(features["rms"])
    onset = _norm01(features["onset"])
    voicing = np.asarray(features["voicing"], dtype=np.float32)
    comp = 0.45 * onset + 0.35 * rms + 0.20 * voicing
    i0 = max(0, int(round(float(t0) / hop)))
    i1 = min(n, int(round(float(t1) / hop)))
    if i1 - i0 < 2:
        return None, -1.0, {}
    win = int(max(1, round(min_dur / hop)))
    best_t, best_s, best_meta = None, -1.0, {}
    for s in range(i0, max(i0 + 1, i1 - win + 1)):
        e = min(n, s + win)
        seg_comp = comp[s:e]
        seg_on = onset[s:e]
        seg_rms = rms[s:e]
        seg_vo = voicing[s:e]
        score = float(np.mean(seg_comp))
        # onset at start is good for word onset
        start_on = float(np.max(seg_on[: max(1, len(seg_on)//3)]))
        score = 0.75 * score + 0.25 * start_on
        # penalize very low voicing when window has some signal
        vo_ratio = float(np.mean(seg_vo))
        if vo_ratio < 0.15 and float(np.mean(seg_rms)) > 0.2:
            score *= 0.85
        if score > best_s:
            best_s = score
            best_t = float(s) * hop
            best_meta = {"voicing_ratio": vo_ratio, "onset": float(np.mean(seg_on)), "rms": float(np.mean(seg_rms))}
    return best_t, float(best_s), best_meta


def best_peak_time(sig: np.ndarray, sr: int, t0: float, t1: float, min_dur: float = 0.12, hop_sec: float = 0.01) -> Tuple[Optional[float], float, Dict[str, Any]]:
    if sig is None or len(sig) == 0 or t1 <= t0:
        return None, -1.0, {}
    i0 = max(0, int(round(float(t0) * sr)))
    i1 = min(len(sig), int(round(float(t1) * sr)))
    if i1 - i0 < int(0.05 * sr):
        return None, -1.0, {}
    seg = np.asarray(sig[i0:i1], dtype=np.float32)
    feats = compute_features(seg, sr=sr, hop_sec=hop_sec)
    bt, sc, meta = score_window(feats, 0.0, float(len(seg)) / float(sr), min_dur=min_dur)
    if bt is None:
        return None, -1.0, {}
    return float(t0) + float(bt), float(sc), meta
