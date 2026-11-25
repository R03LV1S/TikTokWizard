"""Word cementing via real MMS_FA <star>-CTC alignment and lexical merit bounds (v4.1.1).

Includes:
  - Per-line <star>-CTC alignment against full-song or local slice emissions.
  - Fresh per-gap -16 LUFS neural emission extraction (`cement_gap_fresh_audio`)
    so quiet/reverb-washed discrepancy gaps are never drowned out by 20-second
    chunk normalization.
  - Multi-evidence acoustic confidence combining MMS_FA character peak posterior,
    Forward-Backward Viterbi convergence stability, local vocal SNR, and
    self-calibrated syllable merit bounds.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..audio.dsp import normalize_lufs
from ..sync.discrepancy_resolver import _estimate_f0_voicing_ratio
from .mms_fa import align_words_star_ctc, extract_emissions_chunked, get_mms_fa


def _clean_alpha(text: str) -> str:
    t = unicodedata.normalize("NFKD", (text or "").lower())
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", t)


def _count_syllables(word_text: str) -> int:
    key = _clean_alpha(word_text)
    if not key:
        return 1
    groups = re.findall(r"[aeiouy]+", key)
    return max(1, len(groups))


class WordCementer:
    """Aligns and cements words inside anchored line and discrepancy windows."""

    def __init__(
        self,
        bundle: object = None,
        tau_song: float = 0.23,
        word_target: float = 0.95,
        backend: str = "mms_fa",
    ):
        self.bundle = bundle
        self.tau_song = float(tau_song)
        self.word_target = float(word_target)
        self.backend = str(backend or "mms_fa")

    def compute_articulatory_weight(self, word_text: str) -> float:
        """Universal articulatory phoneme-class weight across open/closed vowels and consonants."""
        key = _clean_alpha(word_text)
        if not key:
            return 0.50
        weight = 0.0
        prev_vowel = False
        for ch in key:
            if ch in "aeo":
                weight += 0.42 if not prev_vowel else 0.28
                prev_vowel = True
            elif ch in "iuy":
                weight += 0.34 if not prev_vowel else 0.22
                prev_vowel = True
            elif ch in "mnlr":
                weight += 0.16
                prev_vowel = False
            else:
                weight += 0.09
                prev_vowel = False
        return float(np.clip(0.35 + weight, 0.40, 2.80))

    def compute_merit_bounds(
        self,
        word_text: str,
        is_line_final: bool = False,
        is_parenthetical: bool = False,
    ) -> Tuple[float, float, float]:
        """Return (min_merit_sec, typical_sec, max_merit_sec) scaled by tau_song."""
        key = _clean_alpha(word_text)
        n_chars = max(1, len(key))
        n_syl = _count_syllables(word_text)
        tau = float(np.clip(self.tau_song, 0.10, 0.32))
        is_short_unit = n_syl == 1 and n_chars <= 3

        if is_short_unit and not is_line_final and not is_parenthetical:
            min_m = max(0.03, 0.16 * tau)
            typ_m = max(0.07, 0.50 * tau)
            max_m = max(0.32, 1.95 * tau)
        else:
            min_m = max(0.04, n_syl * 0.22 * tau + 0.003 * n_chars)
            typ_m = max(0.11, n_syl * 0.80 * tau + 0.008 * n_chars)
            # Sustained sung/belted notes can be 3-4s+; never cap below real vocal duration.
            max_m = float(np.clip(n_syl * 6.0 * tau, 1.25, 8.0))

        return float(min_m), float(typ_m), float(max_m)

    def cement_words(
        self,
        words: Sequence[str],
        waveform: torch.Tensor,
        sr: int = 16000,
    ) -> List[Dict[str, Any]]:
        """Align a word sequence directly against a waveform slice."""
        if not words:
            return []
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0)
        wav_np = waveform.detach().cpu().numpy().astype(np.float32)
        dur = max(0.10, float(wav_np.size) / float(max(1, sr)))
        emissions = extract_emissions_chunked(waveform, sr=sr)
        fps = float(emissions.shape[0]) / dur if dur > 0 and emissions.shape[0] > 0 else 50.0
        roman = [_clean_alpha(w) or "a" for w in words]
        paren = ["(" in w or ")" in w for w in words]
        return self.cement_line(
            words=words,
            romanized=roman,
            is_parenthetical=paren,
            window_start=0.0,
            window_end=dur,
            emissions=emissions,
            fps=fps,
            waveform=wav_np,
            sr=sr,
            margin_sec=0.05,
        )

    def cement_gap_fresh_audio(
        self,
        words: Sequence[str],
        romanized: Sequence[str],
        is_parenthetical: Sequence[bool],
        search_lo: float,
        search_hi: float,
        full_waveform: np.ndarray,
        sr: int = 16000,
    ) -> List[Dict[str, Any]]:
        """Run fresh locally -16 LUFS normalized MMS_FA inference on an isolated gap slice.

        Eliminates 20-second chunk normalization suppression on quiet/ad-lib/reverb phrases.
        """
        if not words or full_waveform is None or full_waveform.size == 0:
            return []

        n_samples = int(full_waveform.shape[0])
        total_dur = float(n_samples) / float(max(1, sr))
        s0 = max(0.0, min(float(search_lo), total_dur - 0.08))
        s1 = max(s0 + 0.08, min(float(search_hi), total_dur))

        i0 = max(0, min(int(round(s0 * sr)), n_samples - 1))
        i1 = max(i0 + 1, min(int(round(s1 * sr)), n_samples))
        raw_slice = full_waveform[i0:i1].astype(np.float32)

        # Pad ultra-short slices to at least 0.24s so MMS_FA conv layers have sufficient frames
        min_samples = int(0.24 * sr)
        if raw_slice.size < min_samples:
            raw_slice = np.pad(raw_slice, (0, min_samples - raw_slice.size))

        leveled_slice = normalize_lufs(raw_slice, target_lufs=-16.0, peak_ceiling_db=-0.8)
        if self.backend == "echora":
            from .echora_backend import extract_emissions as _echora_em
            local_em = _echora_em(leveled_slice, sr=sr, device="cpu")
            if local_em.dim() == 3:
                local_em = local_em[0]
        elif self.backend == "tunestudio":
            from .tunestudio_backend import extract_emissions as _ts_em
            local_em = _ts_em(leveled_slice, sr=sr, device="cpu")
            if local_em.dim() == 3:
                local_em = local_em[0]
        else:
            slice_t = torch.from_numpy(leveled_slice).float()
            local_em = extract_emissions_chunked(
                slice_t, sr=sr, chunk_sec=max(4.0, (s1 - s0) + 1.0), overlap_sec=0.5
            )
        if local_em.shape[0] < 2:
            return []

        eff_dur = float(raw_slice.size) / float(sr)
        local_fps = float(local_em.shape[0]) / max(1e-4, eff_dur)

        local_aligned = self.cement_line(
            words=words,
            romanized=romanized,
            is_parenthetical=is_parenthetical,
            window_start=0.0,
            window_end=s1 - s0,
            emissions=local_em,
            fps=local_fps,
            waveform=leveled_slice[: max(1, i1 - i0)],
            sr=sr,
            margin_sec=0.0,
        )
        for item in local_aligned:
            item["start_sec"] = round(s0 + float(item["start_sec"]), 6)
            item["end_sec"] = round(s0 + float(item["end_sec"]), 6)
        return local_aligned

    def _rebalance_and_trim_line_words(
        self,
        aligned: List[Dict[str, Any]],
        win_lo: float,
        win_hi: float,
        waveform: np.ndarray,
        sr: int,
    ) -> None:
        """Enforce lexical merit bounds, trim mid-sentence breath stretches, and prevent overlaps."""
        n_words = len(aligned)
        if n_words == 0:
            return

        span_avail = max(0.05 * n_words, win_hi - win_lo)
        merits = [
            self.compute_merit_bounds(
                w["word"],
                is_line_final=(idx == n_words - 1),
                is_parenthetical=bool(w.get("is_parenthetical", False)),
            )
            for idx, w in enumerate(aligned)
        ]

        cursor = win_lo
        for idx, w in enumerate(aligned):
            min_m, _, max_m = merits[idx]
            rem_min = sum(m[0] for m in merits[idx + 1 :])
            latest_start = max(cursor, win_hi - rem_min - min_m)

            st = float(np.clip(w["start_sec"], cursor, latest_start))
            en = float(max(st + min_m, min(win_hi - rem_min, float(w["end_sec"]))))

            if (en - st) > max_m and idx < n_words - 1 and not w.get("is_parenthetical", False):
                i0 = max(0, int(round(st * sr)))
                i1 = min(waveform.size, int(round(en * sr)))
                if i1 - i0 > int(0.08 * sr):
                    hop = max(1, int(0.01 * sr))
                    frames = [
                        float(np.sqrt(np.mean(waveform[k : k + hop].astype(np.float64) ** 2)))
                        for k in range(i0, i1 - hop, hop)
                    ]
                    if frames:
                        peak_f = max(frames) + 1e-9
                        min_frames_keep = max(3, int(round(min_m / 0.01)))
                        max_frames_keep = max(min_frames_keep + 1, int(round(max_m / 0.01)))
                        cut_f = min(len(frames), max_frames_keep)
                        for fi in range(min_frames_keep, min(len(frames), max_frames_keep)):
                            if frames[fi] < 0.22 * peak_f:
                                cut_f = fi
                                break
                        en = min(en, st + cut_f * 0.01)

            if (en - st) > max_m:
                en = st + max_m

            w["start_sec"] = st
            w["end_sec"] = en
            cursor = en

        total_word_dur = sum(float(w["end_sec"]) - float(w["start_sec"]) for w in aligned)
        min_line_span = min(span_avail, max(0.32, sum(m[0] for m in merits)))
        if total_word_dur < min_line_span and span_avail >= min_line_span:
            weights = np.array(
                [self.compute_articulatory_weight(w["word"]) for w in aligned],
                dtype=np.float64,
            )
            weights /= np.sum(weights) + 1e-9
            t_cur = max(win_lo, float(aligned[0]["start_sec"]))
            if t_cur + min_line_span > win_hi:
                t_cur = max(win_lo, win_hi - min_line_span)
            for idx, w in enumerate(aligned):
                dur_w = max(merits[idx][0], float(min_line_span * weights[idx]))
                w["start_sec"] = t_cur
                w["end_sec"] = min(win_hi, t_cur + dur_w)
                t_cur = w["end_sec"]
                w["_linear_fallback"] = True

    def cement_line(
        self,
        words: Sequence[str],
        romanized: Sequence[str],
        is_parenthetical: Sequence[bool],
        window_start: float,
        window_end: float,
        emissions: torch.Tensor,
        fps: float,
        waveform: np.ndarray,
        sr: int,
        margin_sec: float = 0.20,
        parenthetical_echo_allow: bool = True,
    ) -> List[Dict[str, Any]]:
        """Align a line's words with <star>-CTC and compute calibrated acoustic confidence."""
        if emissions is None or emissions.shape[0] == 0 or not words:
            return []

        total_frames = int(emissions.shape[0])
        total_dur = float(waveform.shape[0]) / float(max(1, sr))
        w0 = max(0.0, float(window_start) - float(margin_sec))
        w1 = min(total_dur, float(window_end) + float(margin_sec))
        if w1 <= w0 + 0.04:
            w1 = min(total_dur, w0 + 0.40)

        f0 = int(round(w0 * fps))
        f1 = int(round(w1 * fps))

        target_char_count = sum(max(1, len(rom)) for rom in romanized) + len(romanized) + 1
        min_frames = target_char_count + 6

        if (f1 - f0) < min_frames:
            needed = min_frames - (f1 - f0)
            pad_left = needed // 2
            pad_right = needed - pad_left
            f0 = max(0, f0 - pad_left)
            f1 = min(total_frames, f1 + pad_right)
            if (f1 - f0) < min_frames:
                if f0 == 0:
                    f1 = min(total_frames, min_frames)
                elif f1 == total_frames:
                    f0 = max(0, total_frames - min_frames)

        f0 = max(0, min(f0, total_frames - 1))
        f1 = max(f0 + 1, min(f1, total_frames))
        slice_em = emissions[f0:f1]
        sec_per_frame = 1.0 / float(fps)

        # Vocal-activity gate: bleed floor max(0.035,0.35*p90_rms) + hysteresis/morph closing
        silence_mask = None
        try:
            from .mms_fa import compute_vocal_activity_mask
            hop_sec = 1.0 / max(1e-6, float(fps))
            active_full, gate_info = compute_vocal_activity_mask(waveform, sr=sr, hop_sec=hop_sec)
            if not hasattr(self, '_vocal_gate_logged'):
                import logging
                logging.getLogger(__name__).info(f"[vocal_gate] masked={gate_info['masked_fraction']*100:.1f}% frames={gate_info['frames']} p90={gate_info['p90']:.5f} thresh={gate_info['thresh']:.5f}")
                self._vocal_gate_logged = True
            sl = slice_em.shape[0]
            seg_active = active_full[f0:f0+sl] if f0+sl <= len(active_full) else np.pad(active_full[f0:], (0, max(0,f0+sl-len(active_full))), constant_values=False)
            silence_mask = ~seg_active[:sl]
        except Exception:
            silence_mask = None

        use_echora = self.backend == "echora" and int(slice_em.shape[-1]) == 32
        use_tunestudio = self.backend == "tunestudio" and int(slice_em.shape[-1]) == 29
        if use_echora:
            from .echora_backend import align_words_ctc as _echora_align
            aligned = _echora_align(
                slice_em, words, romanized, is_parenthetical,
                window_start_sec=float(f0) * sec_per_frame,
                sec_per_frame=sec_per_frame,
            )
            source_label = "echora_karaoke_ctc"
        else:
            aligned = align_words_star_ctc(
                slice_em,
                words,
                romanized,
                is_parenthetical,
                window_start_sec=float(f0) * sec_per_frame,
                sec_per_frame=sec_per_frame,
                silence_frame_mask=silence_mask,
            )
            source_label = "tunestudio_singing_ctc" if use_tunestudio else "mms_fa_star_ctc"

        win_lo = max(0.0, float(window_start) - 0.06)
        win_hi = min(total_dur, max(win_lo + 0.12, float(window_end) + 0.06))
        self._rebalance_and_trim_line_words(aligned, win_lo, win_hi, waveform, sr)

        n_samples = int(waveform.shape[0])
        win_i0 = max(0, min(int(round(w0 * sr)), n_samples - 1))
        win_i1 = max(win_i0 + 1, min(int(round(w1 * sr)), n_samples))
        win_seg = waveform[win_i0:win_i1].astype(np.float64)
        win_rms_ref = (
            float(np.sqrt(np.mean(win_seg * win_seg)) + 1e-5) if win_seg.size else 0.02
        )

        out: List[Dict[str, Any]] = []
        n_words = len(aligned)
        for idx, w in enumerate(aligned):
            s = max(0.0, float(w["start_sec"]))
            e = max(s + 0.03, float(w["end_sec"]))
            i0 = max(0, min(int(round(s * sr)), n_samples - 1))
            i1 = max(i0 + 1, min(int(round(e * sr)), n_samples))
            seg = waveform[i0:i1].astype(np.float64)
            rms = float(np.sqrt(np.mean(seg * seg))) if seg.size else 0.0

            # Estimate F0 voicing ratio (85-1000 Hz)
            seg_rms, f0_harmonicity = _estimate_f0_voicing_ratio(waveform[i0:i1], sr=sr)

            # Clitic / short function word detection (Spanish function words)
            clean_w = re.sub(r"[^\w]+", "", (w["word"] or "").lower())
            is_clitic = (len(clean_w) <= 3 and clean_w in {
                "pa", "y", "i", "de", "el", "la", "los", "las", "se", "que", "en",
                "al", "un", "una", "a", "te", "me", "le", "lo", "mi", "su", "si"
            })

            # Stratified energy factor
            e_ref = max(win_rms_ref * 0.30, 0.008) if is_clitic else max(win_rms_ref * 0.45, 0.012)
            min_energy = 0.45 if is_clitic else 0.30
            energy_factor = float(np.clip(rms / max(1e-4, e_ref), min_energy, 1.0))

            posterior = float(w.get("posterior", 0.80))
            fb_jitter = float(w.get("fb_jitter_sec", 0.02))
            fb_convergence = float(math.exp(-max(0.0, fb_jitter) / 0.065))

            min_m, _, max_m = self.compute_merit_bounds(
                w["word"],
                is_line_final=(idx == n_words - 1),
                is_parenthetical=bool(w.get("is_parenthetical", False)),
            )
            dur = e - s
            merit_ok = (min_m - 0.03) <= dur <= (max_m + 0.03)
            merit_score = 1.0 if merit_ok else 0.80

            # Calibrated 5-pillar acoustic confidence
            f0_factor = float(np.clip(f0_harmonicity * 2.8, 0.55, 1.0)) if rms > 0.005 else 0.60
            if use_echora:
                calibrated_post = float(posterior ** 0.55)
                raw_conf = (
                    0.45 * calibrated_post
                    + 0.10 * fb_convergence
                    + 0.15 * f0_factor
                    + 0.20 * energy_factor
                    + 0.10 * merit_score
                )
            else:
                calibrated_post = float(posterior ** 0.50) if is_clitic else float(posterior ** 0.58)
                raw_conf = (
                    0.35 * calibrated_post
                    + 0.25 * fb_convergence
                    + 0.15 * f0_factor
                    + 0.15 * energy_factor
                    + 0.10 * merit_score
                )
            # High-confidence bonus only when evidence is genuinely strong
            if (fb_jitter <= 0.005 and f0_harmonicity >= 0.25 and merit_ok
                    and raw_conf >= 0.92 and posterior >= 0.75):
                raw_conf = max(raw_conf, min(0.998, raw_conf + 0.02))

            # Evidence floor: low posterior cannot be high-confidence cemented
            if posterior < 0.55:
                raw_conf = min(raw_conf, self.word_target - 0.05)
            elif posterior < 0.70:
                raw_conf = min(raw_conf, self.word_target - 0.01)
            confidence = float(np.clip(round(raw_conf, 4), 0.05, 0.998))
            anchor_tier = None
            reject_reason = None
            if w.get("_linear_fallback"):
                confidence = min(confidence, self.word_target - 0.05)
                anchor_tier = "placement_unverified"
                reject_reason = "linear_fallback_unverified"

            out.append(
                {
                    "word": w["word"],
                    "is_parenthetical": bool(w["is_parenthetical"]),
                    "start_sec": round(s, 6),
                    "end_sec": round(e, 6),
                    "duration_cs": max(3, int(round((e - s) * 100.0))),
                    "posterior": round(posterior, 6),
                    "fb_jitter_sec": round(fb_jitter, 5),
                    "rms": round(rms, 6),
                    "confidence": round(confidence, 6),
                    "is_cemented": bool(confidence >= self.word_target),
                    "anchor_tier": anchor_tier,
                    "reject_reason": reject_reason,
                    "confidence_source": source_label,
                }
            )
        return out
