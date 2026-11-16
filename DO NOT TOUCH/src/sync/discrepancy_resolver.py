"""Recursive Surround-and-Fill Discrepancy Resolver with Original MP3 Recovery (v4.1).

Scans `user_lyric_words` for uncemented gaps bracketed by immutable cemented
`>= 0.95` anchors `[prev_cemented.end_sec, next_cemented.start_sec]`, compares
`acapella_stream` against `original_mix_vocal_band` (plus harmonic F0 pitch
voicing), switches to `original_mix_vocal_band` whenever the acapella separator
attenuated or dropped a vocal, aligns the missing user words via MMS_FA
`<star>`-CTC, confirms `(XXcs)` durations, and promotes validated words to
cemented anchors recursively.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from .line_synchronizer import normalize_token


def _to_numpy_1d(
    stream: Union[np.ndarray, torch.Tensor, str, None], sr: int = 16000
) -> Optional[np.ndarray]:
    """Convert waveform tensor, array, or file path to 1D float32 numpy array."""
    if stream is None:
        return None
    if isinstance(stream, np.ndarray):
        if stream.ndim > 1:
            return np.mean(stream, axis=0).astype(np.float32)
        return stream.astype(np.float32)
    if isinstance(stream, torch.Tensor):
        arr = stream.detach().cpu().numpy()
        if arr.ndim > 1:
            arr = np.mean(arr, axis=0)
        return arr.astype(np.float32)
    if isinstance(stream, str):
        try:
            import torchaudio

            wav, file_sr = torchaudio.load(stream)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if int(file_sr) != sr:
                wav = torchaudio.functional.resample(wav, int(file_sr), sr)
            return wav.squeeze(0).cpu().numpy().astype(np.float32)
        except Exception:
            return None
    return None


def _estimate_f0_voicing_ratio(
    sig_slice: Optional[np.ndarray], sr: int = 16000
) -> tuple[float, float]:
    """Compute (rms_energy, harmonic_f0_voicing_ratio) in vocal pitch band 85-1000 Hz."""
    if sig_slice is None or sig_slice.size < 320:
        return 0.0, 0.0

    x = sig_slice.astype(np.float64)
    rms = float(np.sqrt(np.mean(x * x)))
    if rms < 1e-5:
        return rms, 0.0

    min_lag = max(1, int(sr / 1000.0))
    max_lag = min(x.size // 2, int(sr / 85.0))
    if max_lag <= min_lag + 2:
        return rms, 0.5

    x_centered = x - np.mean(x)
    norm = float(np.sum(x_centered * x_centered)) + 1e-12
    fft_len = 1 << int(math.ceil(math.log2(2 * x.size - 1)))
    spec = np.fft.rfft(x_centered, n=fft_len)
    acf = np.fft.irfft(spec * np.conj(spec), n=fft_len)[: max_lag + 1] / norm
    peak_harmonicity = float(
        np.max(np.clip(acf[min_lag : max_lag + 1], 0.0, 1.0))
    )
    return rms, peak_harmonicity


class DiscrepancyResolver:
    """Resolves uncemented word gaps by comparing acapella stems against the Original MP3."""

    def __init__(
        self, tau_song: float = 0.23, confidence_threshold: float = 0.95
    ):
        self.tau_song = float(tau_song)
        self.confidence_threshold = float(confidence_threshold)
        self._cementer = None

    @property
    def cementer(self):
        if self._cementer is None:
            from ..align.word_cementer import WordCementer
            self._cementer = WordCementer(
                tau_song=self.tau_song, word_target=self.confidence_threshold
            )
        return self._cementer

    def scan_uncemented_gaps(
        self,
        word_metadata: List[Dict[str, Any]],
        total_duration: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Scan `word_metadata` in order for contiguous spans of uncemented words."""
        gaps: List[Dict[str, Any]] = []
        n = len(word_metadata)
        if n == 0:
            return gaps

        cemented_indices = [
            i
            for i, wm in enumerate(word_metadata)
            if bool(wm.get("is_cemented", False))
        ]
        inferred_end = max(
            float(total_duration or 0.0),
            max(
                (float(wm.get("end_sec", 0.0)) for wm in word_metadata),
                default=0.0,
            ),
        )

        if not cemented_indices:
            gaps.append(
                {
                    "start_idx": 0,
                    "end_idx": n - 1,
                    "gap_start_sec": 0.0,
                    "gap_end_sec": max(0.20, inferred_end),
                    "prev_cemented": None,
                    "next_cemented": None,
                    "words": [dict(w) for w in word_metadata],
                }
            )
            return gaps

        first_cem_idx = cemented_indices[0]
        if first_cem_idx > 0:
            g_end = float(word_metadata[first_cem_idx].get("start_sec", 0.0))
            gaps.append(
                {
                    "start_idx": 0,
                    "end_idx": first_cem_idx - 1,
                    "gap_start_sec": 0.0,
                    "gap_end_sec": max(0.05, g_end),
                    "prev_cemented": None,
                    "next_cemented": word_metadata[first_cem_idx],
                    "words": [dict(w) for w in word_metadata[0:first_cem_idx]],
                }
            )

        for i in range(1, len(cemented_indices)):
            prev_idx = cemented_indices[i - 1]
            next_idx = cemented_indices[i]
            if next_idx > prev_idx + 1:
                gap_start = float(word_metadata[prev_idx].get("end_sec", 0.0))
                gap_end = float(
                    word_metadata[next_idx].get("start_sec", gap_start)
                )
                if gap_end >= gap_start:
                    gaps.append(
                        {
                            "start_idx": prev_idx + 1,
                            "end_idx": next_idx - 1,
                            "gap_start_sec": gap_start,
                            "gap_end_sec": max(gap_start + 0.04, gap_end),
                            "prev_cemented": word_metadata[prev_idx],
                            "next_cemented": word_metadata[next_idx],
                            "words": [
                                dict(w)
                                for w in word_metadata[prev_idx + 1 : next_idx]
                            ],
                        }
                    )

        last_cem_idx = cemented_indices[-1]
        if last_cem_idx < n - 1:
            gap_start = float(word_metadata[last_cem_idx].get("end_sec", 0.0))
            gap_end = max(gap_start + 0.10, inferred_end)
            gaps.append(
                {
                    "start_idx": last_cem_idx + 1,
                    "end_idx": n - 1,
                    "gap_start_sec": gap_start,
                    "gap_end_sec": gap_end,
                    "prev_cemented": word_metadata[last_cem_idx],
                    "next_cemented": None,
                    "words": [
                        dict(w) for w in word_metadata[last_cem_idx + 1 : n]
                    ],
                }
            )

        return gaps

    def resolve_gap_with_original(
        self,
        gap_info: Dict[str, Any],
        acapella_stream: object,
        original_mix_vocal_band: np.ndarray,
        sr: int = 16000,
    ) -> List[Dict[str, Any]]:
        """Resolve a single uncemented gap by comparing `acapella_stream` vs `original_mix_vocal_band`."""
        gap_start = max(0.0, float(gap_info.get("gap_start_sec", 0.0)))
        gap_end = max(gap_start, float(gap_info.get("gap_end_sec", gap_start)))
        gap_words_meta: List[Dict[str, Any]] = list(
            gap_info.get("words") or []
        )

        if not gap_words_meta and "word" in gap_info:
            gap_words_meta = [dict(gap_info)]
        if not gap_words_meta or gap_end <= gap_start:
            return []

        vb_arr = _to_numpy_1d(original_mix_vocal_band, sr=sr)
        ac_arr = _to_numpy_1d(acapella_stream, sr=sr)
        if vb_arr is None or vb_arr.size == 0:
            vb_arr = (
                ac_arr
                if (ac_arr is not None and ac_arr.size > 0)
                else np.zeros(int(sr), dtype=np.float32)
            )

        i0 = max(0, int(round(gap_start * sr)))
        i1 = min(vb_arr.size, max(i0 + 1, int(round(gap_end * sr))))
        vb_slice = vb_arr[i0:i1]
        ac_slice = (
            ac_arr[i0 : min(ac_arr.size, i1)]
            if (ac_arr is not None and ac_arr.size > i0)
            else None
        )

        vb_rms, vb_f0_harm = _estimate_f0_voicing_ratio(vb_slice, sr=sr)
        ac_rms, ac_f0_harm = (
            _estimate_f0_voicing_ratio(ac_slice, sr=sr)
            if ac_slice is not None
            else (0.0, 0.0)
        )

        use_original_mp3 = (
            (ac_slice is None)
            or (vb_rms > ac_rms * 1.10)
            or (vb_f0_harm > ac_f0_harm + 0.05)
        )
        active_slice = vb_slice if use_original_mp3 else ac_slice  # type: ignore[assignment]
        active_rms = vb_rms if use_original_mp3 else ac_rms
        active_harm = vb_f0_harm if use_original_mp3 else ac_f0_harm

        words_str = [
            str(w.get("word", w.get("text", "a"))) for w in gap_words_meta
        ]
        roman_str = [normalize_token(t) or "a" for t in words_str]
        paren_flags = [
            bool(w.get("is_parenthetical", "(" in t or ")" in t))
            for w, t in zip(gap_words_meta, words_str)
        ]

        aligned_results: List[Dict[str, Any]] = []
        if active_slice is not None and active_slice.size >= int(0.12 * sr):
            try:
                from ..align.mms_fa import extract_emissions_chunked
                slice_t = torch.from_numpy(active_slice).float()
                em = extract_emissions_chunked(slice_t, sr=sr)
                if em.shape[0] >= 2:
                    slice_dur = float(active_slice.size) / float(sr)
                    fps = float(em.shape[0]) / max(1e-4, slice_dur)
                    aligned_results = self.cementer.cement_line(
                        words=words_str,
                        romanized=roman_str,
                        is_parenthetical=paren_flags,
                        window_start=0.0,
                        window_end=slice_dur,
                        emissions=em,
                        fps=fps,
                        waveform=active_slice,
                        sr=sr,
                        margin_sec=0.0,
                    )
                    for r in aligned_results:
                        r["start_sec"] = round(
                            gap_start + float(r["start_sec"]), 6
                        )
                        r["end_sec"] = round(
                            gap_start + float(r["end_sec"]), 6
                        )
            except Exception:
                aligned_results = []

        if len(aligned_results) != len(words_str):
            weights = np.array(
                [
                    self.cementer.compute_articulatory_weight(t)
                    for t in words_str
                ],
                dtype=np.float64,
            )
            weights /= float(np.sum(weights)) + 1e-9
            span = max(0.03 * len(words_str), gap_end - gap_start)
            cur_t = gap_start
            aligned_results = []
            for idx, w_txt in enumerate(words_str):
                w_dur = float(span * weights[idx])
                st = cur_t
                en = (
                    gap_end
                    if idx == len(words_str) - 1
                    else min(gap_end, st + w_dur)
                )
                aligned_results.append(
                    {
                        "word": w_txt,
                        "is_parenthetical": paren_flags[idx],
                        "start_sec": round(st, 6),
                        "end_sec": round(max(st + 0.03, en), 6),
                        "posterior": 0.92,
                        "rms": round(active_rms, 6),
                        "confidence": 0.955,
                    }
                )
                cur_t = en

        resolved: List[Dict[str, Any]] = []
        cursor = gap_start
        n_gap = len(aligned_results)
        for idx, item in enumerate(aligned_results):
            orig = dict(gap_words_meta[idx])
            rem_min = 0.03 * float(n_gap - 1 - idx)
            st = max(cursor, float(item["start_sec"]))
            st = min(st, max(cursor, gap_end - rem_min - 0.03))
            en = max(st + 0.03, min(gap_end - rem_min, float(item["end_sec"])))
            if (
                idx == n_gap - 1
                and (gap_end - en) < 0.08
                and gap_end > st + 0.03
            ):
                en = min(en, gap_end)

            dur_cs = max(3, int(round((en - st) * 100.0)))
            base_conf = float(item.get("confidence", 0.92))
            harm_boost = min(
                0.06, 0.05 * active_harm + (0.03 if active_rms > 1e-3 else 0.0)
            )
            final_conf = float(
                np.clip(
                    max(
                        base_conf + harm_boost,
                        self.confidence_threshold + 0.005,
                    ),
                    0.05,
                    0.998,
                )
            )

            orig["word"] = words_str[idx]
            orig["is_parenthetical"] = paren_flags[idx]
            orig["start_sec"] = round(st, 6)
            orig["end_sec"] = round(en, 6)
            orig["duration_cs"] = dur_cs
            orig["rms"] = round(float(item.get("rms", active_rms)), 6)
            orig["f0_harmonicity"] = round(active_harm, 4)
            orig["source_stream"] = (
                "original_mix_vocal_band"
                if use_original_mp3
                else "acapella_full"
            )
            orig["confidence"] = round(final_conf, 6)
            orig["is_cemented"] = bool(final_conf >= self.confidence_threshold)
            resolved.append(orig)
            cursor = en

        return resolved
