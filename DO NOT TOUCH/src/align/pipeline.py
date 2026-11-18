# pipeline.py
"""End-to-end greenfield alignment pipeline (real acoustic alignment v4.1 consolidated).

Flow:
  1. Holy-grail UTF-8 ingest of the user lyric (immutable surface text, 1:1 lines).
  2. True stereo-aware decoding + 4 synchronized 16 kHz mono stems
     (original_mix_vocal_band, acapella_full, acapella_lead_dry, acapella_backing_echo)
     with duration invariant verification against stereo-flatten bugs.
  3. Dual-stream ASR word timestamps -> self-calibrated tau_song -> monotonic
     lyric<->ASR word anchors with parenthetical echo-clone filtering.
  4. Chunked MMS_FA 50fps log-posterior emissions over both acapella_full and
     original_mix_vocal_band.
  5. Per-line <star>-CTC word cementing with Forward-Backward Viterbi convergence,
     local -16 LUFS normalization, and self-calibrated lexical merit bounds.
  6. Multi-pass recursive Surround-and-Fill discrepancy resolution via
     DiscrepancyResolver bracketed by cemented >= 0.95 anchors
     [prev_cemented.end_sec, next_cemented.start_sec], switching to
     original_mix_vocal_band whenever the acapella stem attenuated vocals.
  7. Sub-frame (2ms) consonant/sibilant transient snapping, 1:1 export
     (karaoke.ass, lyrics_synced.txt, alignment_manifest.json), and the
     anti-fabrication verification gate.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
import scipy.signal
import torch
import torchaudio

from ..audio.dsp import compute_rms_envelope
from ..audio.streams import (
    decode_audio_mono,
    decode_audio_stereo_and_mono,
    stereo_azimuth_decompose,
)
from ..subtitles.ass_writer import write_ass
from ..subtitles.lrc_exporter import export_lrc
from ..sync.discrepancy_resolver import DiscrepancyResolver
from ..sync.ingestor import parse_user_lyric, romanize_word
from ..sync.line_synchronizer import (
    align_words_monotonic,
    build_line_windows,
    normalize_token,
    transcribe_vocal,
    _token_similarity_score,
)
from .integrity_verifier import IntegrityVerifier
from .mms_fa import clean_phonetic_token, extract_emissions_chunked
from .transient_snapper import snap_words_transients_2ms
from .word_cementer import WordCementer

FPS: float = 50.0
TARGET_SR: int = 16000


def _count_syllables_universal(token: str) -> int:
    clean = normalize_token(token) or romanize_word(token)
    if not clean:
        return 1
    vowel_groups = re.findall(r"[aeiouy]+", clean.lower())
    return max(1, len(vowel_groups))


def _zero_phase_bandpass(
    sig: np.ndarray, lo_hz: float, hi_hz: float, sr: int = TARGET_SR, order: int = 4
) -> np.ndarray:
    if sig.size < 32:
        return sig.astype(np.float32, copy=False)
    nyq = 0.5 * float(sr)
    lo = max(20.0 / nyq, min(0.95, lo_hz / nyq))
    hi = max(lo + 0.02, min(0.99, hi_hz / nyq))
    sos = scipy.signal.butter(order, [lo, hi], btype="bandpass", output="sos")
    return scipy.signal.sosfiltfilt(sos, sig.astype(np.float64)).astype(np.float32)


class UniversalEngine:
    """The clean-slate 5-step universal engine (real acoustics v4.1)."""

    def __init__(
        self,
        tau_song: float = 0.23,
        word_target: float = 0.95,
        whisper_size: str = "small",
    ):
        self.tau_song = float(tau_song)
        self.word_target = float(word_target)
        self.whisper_size = str(whisper_size)
        self.cementer = WordCementer(tau_song=self.tau_song, word_target=self.word_target)
        self.resolver = DiscrepancyResolver(
            tau_song=self.tau_song, confidence_threshold=self.word_target
        )
        self.verifier = IntegrityVerifier(word_target=self.word_target)

    # ------------------------------------------------------------------ #
    def _build_stems(self, mp3_path: str, stems_dir: str) -> Dict[str, str]:
        """Build 4 synchronized 16 kHz mono stems with stereo-flatten validation."""
        os.makedirs(stems_dir, exist_ok=True)
        mono, left, right, _ = decode_audio_stereo_and_mono(mp3_path, target_sr=TARGET_SR)
        true_dur = float(mono.size) / float(TARGET_SR)

        paths = {
            "original_mix_vocal_band": os.path.join(stems_dir, "original_mix_vocal_band.wav"),
            "acapella_full": os.path.join(stems_dir, "acapella_full.wav"),
            "acapella_lead_dry": os.path.join(stems_dir, "acapella_lead_dry.wav"),
            "acapella_backing_echo": os.path.join(stems_dir, "acapella_backing_echo.wav"),
        }

        cache_valid = True
        for p in paths.values():
            if not os.path.isfile(p):
                cache_valid = False
                break
            try:
                info = torchaudio.info(p)
                cached_dur = float(info.num_frames) / float(max(1, info.sample_rate))
                if abs(cached_dur - true_dur) > 0.50 or info.num_channels != 1:
                    cache_valid = False
                    break
            except Exception:
                cache_valid = False
                break

        if cache_valid:
            return paths

        # (a) original_mix_vocal_band: Mid-channel weighted 220-4200 Hz zero-phase vocal band
        mid = 0.5 * (left + right)
        side = 0.5 * (left - right)
        center_focus = (0.80 * mid + 0.20 * mono).astype(np.float32)
        vb = _zero_phase_bandpass(center_focus, 220.0, 4200.0, TARGET_SR, order=4)
        vb = (vb / (float(np.max(np.abs(vb))) + 1e-9) * 0.89).astype(np.float32)
        torchaudio.save(
            paths["original_mix_vocal_band"],
            torch.from_numpy(vb).unsqueeze(0),
            TARGET_SR,
        )

        # (b) acapella_full via audio-separator on original stereo MP3, else Mid-Side fallback
        acapella_16k: Optional[np.ndarray] = None
        try:
            from audio_separator.separator import Separator

            sep = Separator(output_dir=stems_dir, output_format="wav")
            sep.load_model()
            outs = sep.separate(mp3_path)
            for o in outs or []:
                cand_path = o if os.path.isabs(str(o)) else os.path.join(stems_dir, str(o))
                if "vocal" in str(o).lower() and os.path.isfile(cand_path):
                    arr, _ = decode_audio_mono(cand_path, target_sr=TARGET_SR)
                    if abs(arr.size / float(TARGET_SR) - true_dur) <= 1.0:
                        acapella_16k = arr
                        break
        except Exception:
            acapella_16k = None

        if acapella_16k is None:
            mid_vocal = _zero_phase_bandpass(mid, 160.0, 7200.0, TARGET_SR, order=4)
            side_vocal = _zero_phase_bandpass(side, 220.0, 6800.0, TARGET_SR, order=4)
            acapella_16k = (0.75 * mid_vocal + 0.25 * side_vocal).astype(np.float32)

        if acapella_16k.size > mono.size:
            acapella_16k = acapella_16k[: mono.size]
        elif acapella_16k.size < mono.size:
            acapella_16k = np.pad(acapella_16k, (0, mono.size - acapella_16k.size))

        acapella_16k = (
            acapella_16k / (float(np.max(np.abs(acapella_16k))) + 1e-9) * 0.89
        ).astype(np.float32)
        torchaudio.save(
            paths["acapella_full"],
            torch.from_numpy(acapella_16k).unsqueeze(0),
            TARGET_SR,
        )

        # (c) acapella_lead_dry & (d) acapella_backing_echo via STFT azimuth decomposition
        lead_raw, echo_raw = stereo_azimuth_decompose(left, right, sr=TARGET_SR)
        lead_16k = _zero_phase_bandpass(
            0.65 * acapella_16k + 0.35 * lead_raw, 150.0, 7600.0, TARGET_SR
        )
        echo_16k = _zero_phase_bandpass(echo_raw, 180.0, 7600.0, TARGET_SR)
        lead_16k = (lead_16k / (float(np.max(np.abs(lead_16k))) + 1e-9) * 0.89).astype(np.float32)
        echo_16k = (echo_16k / (float(np.max(np.abs(echo_16k))) + 1e-9) * 0.89).astype(np.float32)

        torchaudio.save(
            paths["acapella_lead_dry"],
            torch.from_numpy(lead_16k).unsqueeze(0),
            TARGET_SR,
        )
        torchaudio.save(
            paths["acapella_backing_echo"],
            torch.from_numpy(echo_16k).unsqueeze(0),
            TARGET_SR,
        )

        return paths

    # ------------------------------------------------------------------ #
    def _calibrate_tau_song(
        self,
        flat_words: Sequence[str],
        mapping: Dict[int, int],
        asr_words: Sequence[Tuple[float, float, str]],
    ) -> float:
        per_syl: List[float] = []
        for w_idx, asr_idx in mapping.items():
            if 0 <= w_idx < len(flat_words) and 0 <= asr_idx < len(asr_words):
                st, en, _ = asr_words[asr_idx]
                dur = float(en) - float(st)
                if 0.04 <= dur <= 1.25:
                    n_syl = _count_syllables_universal(flat_words[w_idx])
                    per_syl.append(dur / float(max(1, n_syl)))
        if len(per_syl) >= 8:
            return float(np.clip(np.median(per_syl), 0.10, 0.30))
        return self.tau_song

    # ------------------------------------------------------------------ #
    def _filter_echo_clones_in_asr(
        self,
        flat_words: Sequence[str],
        asr_words: List[Tuple[float, float, str]],
        lead_sig: Optional[np.ndarray] = None,
        echo_sig: Optional[np.ndarray] = None,
        sr: int = TARGET_SR,
    ) -> List[Tuple[float, float, str]]:
        if not asr_words:
            return asr_words

        # If stereo azimuth decomposed stems are provided, apply universal acoustic ratio check:
        # Reverb tails have high backing/echo energy (ratio > 1.40) decaying after the lead vocal
        if lead_sig is not None and echo_sig is not None and lead_sig.size > 0:
            filtered: List[Tuple[float, float, str]] = []
            last_seen_en: Dict[str, float] = {}
            for st, en, txt in asr_words:
                tok = normalize_token(txt) or romanize_word(txt)
                prev_end = last_seen_en.get(tok, -100.0)
                time_since_last = float(st) - prev_end

                i0 = max(0, min(int(round(float(st) * sr)), lead_sig.size - 1))
                i1 = max(i0 + 1, min(int(round(float(en) * sr)), lead_sig.size))
                r_lead = float(np.sqrt(np.mean(lead_sig[i0:i1].astype(np.float64)**2))) if i1 > i0 else 0.0
                r_echo = float(np.sqrt(np.mean(echo_sig[i0:i1].astype(np.float64)**2))) if i1 > i0 else 0.0
                ratio = r_echo / max(1e-4, r_lead)

                # Acoustic echo clone: same token appearing within reverb decay time (<= 6.0s)
                # where backing echo energy significantly dominates lead dry vocal
                if 0.0 <= time_since_last <= 6.0 and ratio > 1.40:
                    continue

                filtered.append((float(st), float(en), txt))
                last_seen_en[tok] = float(en)
            return filtered

        max_user_run: Dict[str, int] = {}
        cur_tok = ""
        cur_len = 0
        for w in flat_words:
            nk = normalize_token(w) or romanize_word(w)
            if not nk:
                continue
            if nk == cur_tok:
                cur_len += 1
            else:
                cur_tok = nk
                cur_len = 1
            if cur_len > max_user_run.get(nk, 0):
                max_user_run[nk] = cur_len

        filtered_lex: List[Tuple[float, float, str]] = []
        run_tok = ""
        run_count = 0
        prev_end = -10.0

        for st, en, txt in asr_words:
            nk = normalize_token(txt) or romanize_word(txt)
            if nk and nk == run_tok and (float(st) - prev_end) <= 2.2:
                run_count += 1
            else:
                run_tok = nk
                run_count = 1
            prev_end = float(en)

            allowed = max(2, max_user_run.get(nk, 1))
            if run_count <= allowed:
                filtered_lex.append((float(st), float(en), txt))

        return filtered_lex

    # ------------------------------------------------------------------ #
    def _resolve_discrepancies_recursive(
        self,
        all_words: List[Dict[str, Any]],
        flat_roman: Sequence[str],
        windows: List[Dict[str, Any]],
        emissions_acapella: torch.Tensor,
        emissions_mix_vb: torch.Tensor,
        fps: float,
        acapella_sig: np.ndarray,
        mix_vb_sig: np.ndarray,
        sr: int,
        dur: float,
        log: Callable[[str], None],
    ) -> None:
        """Step 6: Recursive Surround-and-Fill Discrepancy Resolver (v4.3 4-Phase Architecture)."""
        rms_ac = compute_rms_envelope(acapella_sig, sr, hop_sec=0.01)
        rms_vb = compute_rms_envelope(mix_vb_sig, sr, hop_sec=0.01)
        ac_med = float(np.median(rms_ac) + 1e-6)
        vb_med = float(np.median(rms_vb) + 1e-6)

        n_words = len(all_words)
        for pass_idx in range(1, 6):
            gaps = self.resolver.scan_uncemented_gaps(all_words, total_duration=dur)
            uncemented_count = sum(
                1
                for w in all_words
                if not w.get("is_cemented", False)
                or float(w.get("confidence", 0.0)) < self.word_target
            )
            if not gaps or uncemented_count == 0:
                break

            newly_cemented = 0
            for gap in gaps:
                idx_a = int(gap["start_idx"])
                idx_b = int(gap["end_idx"])
                line_a = int(all_words[idx_a]["line_index"])
                line_b = int(all_words[idx_b]["line_index"])

                t_left = float(gap["gap_start_sec"])
                t_right = float(gap["gap_end_sec"])

                # Dynamic acoustic search window expansion
                pad = 0.08 * pass_idx
                line_win_lo = max(0.0, float(windows[line_a]["start_sec"]) - 0.50)
                line_win_hi = min(dur, float(windows[line_b]["end_sec"]) + 0.50)
                prev_bound = (
                    float(all_words[idx_a - 1]["end_sec"]) if idx_a > 0 else line_win_lo
                )
                next_bound = (
                    float(all_words[idx_b + 1]["start_sec"])
                    if idx_b < n_words - 1
                    else line_win_hi
                )

                search_lo = max(prev_bound, t_left - pad)
                search_hi = min(next_bound, t_right + pad)

                min_needed = 0.06 * float(idx_b - idx_a + 1)
                if search_hi - search_lo < min_needed:
                    mid_t = 0.5 * (t_left + t_right)
                    search_lo = max(prev_bound, mid_t - 0.5 * min_needed)
                    search_hi = min(next_bound, mid_t + 0.5 * min_needed)

                gap_words = [all_words[k]["word"] for k in range(idx_a, idx_b + 1)]
                gap_roman = [flat_roman[k] for k in range(idx_a, idx_b + 1)]
                gap_paren = [
                    bool(all_words[k].get("is_parenthetical", False))
                    for k in range(idx_a, idx_b + 1)
                ]

                best_cand: List[Dict[str, Any]] = []
                use_vb = False

                if pass_idx <= 2:
                    # Stratum A: Fresh MMS_FA on gap
                    cand_ac = self.cementer.cement_gap_fresh_audio(
                        gap_words,
                        gap_roman,
                        gap_paren,
                        search_lo,
                        search_hi,
                        acapella_sig,
                        sr=sr,
                    )
                    cand_vb = self.cementer.cement_gap_fresh_audio(
                        gap_words,
                        gap_roman,
                        gap_paren,
                        search_lo,
                        search_hi,
                        mix_vb_sig,
                        sr=sr,
                    )
                    mean_ac = (
                        float(np.mean([c["confidence"] for c in cand_ac]))
                        if cand_ac
                        else 0.0
                    )
                    mean_vb = (
                        float(np.mean([c["confidence"] for c in cand_vb]))
                        if cand_vb
                        else 0.0
                    )
                    use_vb = (mean_vb > mean_ac + 0.01)
                    best_cand = cand_vb if use_vb else cand_ac
                    active_rms = rms_vb if use_vb else rms_ac
                    active_med = vb_med if use_vb else ac_med

                if (not best_cand or len(best_cand) != len(gap_words)) and pass_idx >= 3:
                    # Stratum B: Dual-Stream Harmonic Fallback with original mix
                    gap_query = dict(gap)
                    gap_query["gap_start_sec"] = search_lo
                    gap_query["gap_end_sec"] = search_hi
                    best_cand = self.resolver.resolve_gap_with_original(
                        gap_query, acapella_sig, mix_vb_sig, sr=sr
                    )
                    use_vb = True
                    active_rms = rms_vb
                    active_med = vb_med

                if (not best_cand or len(best_cand) != len(gap_words) or pass_idx >= 5):
                    # Stratum C: Irreducible Bounded-Island Solver
                    gap_query = dict(gap)
                    gap_query["gap_start_sec"] = search_lo
                    gap_query["gap_end_sec"] = search_hi
                    best_cand = self.resolver.resolve_bounded_gap_island(
                        gap_query, mix_vb_sig, sr=sr
                    )
                    use_vb = True
                    active_rms = rms_vb
                    active_med = vb_med

                if not best_cand or len(best_cand) != len(gap_words):
                    continue

                for offset, cand in enumerate(best_cand):
                    target_idx = idx_a + offset
                    orig_w = all_words[target_idx]
                    c_st = float(cand["start_sec"])
                    c_en = float(cand["end_sec"])

                    p_bound = (
                        float(all_words[target_idx - 1]["end_sec"])
                        if target_idx > 0
                        else 0.0
                    )
                    n_bound = (
                        float(all_words[target_idx + 1]["start_sec"])
                        if (
                            target_idx + 1 < n_words
                            and all_words[target_idx + 1].get("is_cemented", False)
                        )
                        else dur
                    )
                    c_st = max(p_bound, c_st)
                    c_en = max(
                        c_st + 0.03,
                        min(n_bound if n_bound > c_st + 0.03 else c_en, c_en),
                    )

                    w_f0 = max(0, int(round(c_st * 100)))
                    w_f1 = min(active_rms.size, max(w_f0 + 1, int(round(c_en * 100))))
                    local_snr = (
                        float(np.mean(active_rms[w_f0:w_f1])) / active_med
                        if w_f1 > w_f0
                        else 1.0
                    )
                    energy_bonus = min(0.06, 0.025 * max(0.0, local_snr - 1.0))
                    new_conf = min(
                        0.998, round(float(cand["confidence"]) + energy_bonus, 4)
                    )

                    orig_conf = float(orig_w.get("confidence", 0.0))
                    target_line = int(orig_w.get("line_index", line_a))
                    line_start_bound = float(windows[target_line]["start_sec"]) - 0.50
                    line_end_bound = float(windows[target_line]["end_sec"]) + 0.50

                    within_line_bounds = (
                        line_start_bound <= c_st and c_en <= line_end_bound + 1.0
                    )

                    # Monotonic Quality Overwrite Gate
                    can_overwrite = False
                    if orig_w.get("is_cemented", False):
                        can_overwrite = within_line_bounds and (new_conf > orig_conf + 0.005)
                    else:
                        can_overwrite = within_line_bounds and (
                            new_conf >= self.word_target
                            or new_conf >= orig_conf + 0.005
                            or (new_conf >= 0.85 and orig_conf < 0.85)
                            or (pass_idx >= 4 and new_conf >= 0.90)
                            or pass_idx >= 5
                        )

                    if can_overwrite:
                        orig_w["start_sec"] = round(c_st, 4)
                        orig_w["end_sec"] = round(c_en, 4)
                        orig_w["duration_cs"] = max(
                            3, int(round((c_en - c_st) * 100))
                        )
                        orig_w["confidence"] = new_conf
                        orig_w["is_cemented"] = bool(new_conf >= self.word_target)
                        orig_w["source_stream"] = (
                            "original_mix_vocal_band" if use_vb else "acapella_full"
                        )
                        if orig_w["is_cemented"]:
                            newly_cemented += 1

            log(
                f"[6.{pass_idx}] recursive surround-and-fill pass cemented={newly_cemented}"
            )
            remaining = sum(
                1
                for w in all_words
                if not w.get("is_cemented", False)
                or float(w.get("confidence", 0.0)) < self.word_target
            )
            if remaining == 0:
                break

    # ------------------------------------------------------------------ #
    def run(
        self,
        lyric_path: str,
        mp3_path: str,
        out_dir: str,
        log: Callable[[str], None] = print,
    ) -> Dict[str, Any]:
        os.makedirs(out_dir, exist_ok=True)
        stems_dir = os.path.join(out_dir, "stems")

        # --- Step 1: Holy-Grail UTF-8 Ingest ---
        with open(lyric_path, "r", encoding="utf-8-sig") as f:
            raw = f.read()
        lines, parsed_words = parse_user_lyric(raw)
        log(f"[1] ingested lines={len(lines)} words={len(parsed_words)}")

        line_word_slices: List[Tuple[int, int]] = []
        flat_words: List[str] = []
        flat_roman: List[str] = []
        flat_paren: List[bool] = []

        for li, ln in enumerate(lines):
            ws = len(flat_words)
            l_objs = [w for w in parsed_words if w.line_index == li]
            if l_objs:
                for w_obj in l_objs:
                    flat_words.append(w_obj.text)
                    flat_roman.append(
                        clean_phonetic_token(w_obj.text) or normalize_token(w_obj.text) or w_obj.romanized_key or "a"
                    )
                    flat_paren.append(bool(w_obj.is_parenthetical))
            else:
                for t in ln.split():
                    flat_words.append(t)
                    flat_roman.append(clean_phonetic_token(t) or normalize_token(t) or romanize_word(t) or "a")
                    flat_paren.append("(" in t or ")" in t)
            line_word_slices.append((ws, len(flat_words)))
        log(f"[1] flattened words={len(flat_words)}")

        # --- Step 2: Synchronized 16 kHz Mono Stems ---
        stems = self._build_stems(mp3_path, stems_dir)
        mix_vb_sig, sr = decode_audio_mono(stems["original_mix_vocal_band"], TARGET_SR)
        acapella_sig, _ = decode_audio_mono(stems["acapella_full"], TARGET_SR)
        lead_sig, _ = decode_audio_mono(stems["acapella_lead_dry"], TARGET_SR)
        echo_sig, _ = decode_audio_mono(stems["acapella_backing_echo"], TARGET_SR)
        dur = float(len(mix_vb_sig)) / float(sr)
        log(f"[2] stems built; true duration={dur:.2f}s")

        # --- Step 3: Dual-Stream ASR Anchors + Echo-Clone Filter + tau_song ---
        asr_lead, asr_info = transcribe_vocal(
            stems["acapella_lead_dry"], model_size=self.whisper_size
        )
        asr_vb, _ = transcribe_vocal(
            stems["original_mix_vocal_band"], model_size=self.whisper_size
        )

        all_candidates: List[Tuple[float, float, str]] = []
        for st, en, txt in asr_lead:
            all_candidates.append((float(st), float(en), str(txt)))
        for st, en, txt in asr_vb:
            all_candidates.append((float(st), float(en), str(txt)))
        all_candidates.sort(key=lambda x: (x[0], x[1]))

        # Universal dual-stream deduplication: keep both streams, deduplicating only overlapping synonyms
        merged_asr: List[Tuple[float, float, str]] = []
        for c_st, c_en, c_txt in all_candidates:
            c_norm = normalize_token(c_txt)
            is_dupe = False
            for p_st, p_en, p_txt in merged_asr[-4:]:
                p_norm = normalize_token(p_txt)
                overlap = max(0.0, min(c_en, p_en) - max(c_st, p_st))
                if overlap > 0.10:
                    sim = _token_similarity_score(c_norm, p_norm)
                    if sim >= 70.0:
                        is_dupe = True
                        break
            if not is_dupe:
                merged_asr.append((c_st, c_en, c_txt))
        merged_asr.sort(key=lambda x: (x[0], x[1]))

        asr_words = self._filter_echo_clones_in_asr(
            flat_words, merged_asr, lead_sig=lead_sig, echo_sig=echo_sig, sr=sr
        )
        log(
            f"[3] ASR raw_lead={len(asr_lead)} raw_vb={len(asr_vb)} merged={len(merged_asr)} "
            f"echo_filtered={len(asr_words)} lang={asr_info.get('language')}"
        )

        word_to_line: List[int] = [
            li for li, (ws, we) in enumerate(line_word_slices) for _ in range(ws, we)
        ]
        mapping = align_words_monotonic(
            flat_words,
            [w[2] for w in asr_words],
            word_to_line=word_to_line,
            asr_times=[(w[0], w[1]) for w in asr_words],
        )
        self.tau_song = self._calibrate_tau_song(flat_words, mapping, asr_words)
        self.cementer.tau_song = self.tau_song
        self.resolver.tau_song = self.tau_song
        self.resolver.cementer.tau_song = self.tau_song

        rms_vb = compute_rms_envelope(mix_vb_sig, sr, hop_sec=0.01)
        windows = build_line_windows(
            lines, line_word_slices, mapping, asr_words, dur, vocal_rms=rms_vb, rms_hop_sec=0.01
        )
        log(
            f"[3] anchored {len(windows)} line windows; calibrated tau_song={self.tau_song:.3f}s"
        )

        # --- Step 4: Dual-Stream Chunked MMS_FA Emissions ---
        emissions_acapella = extract_emissions_chunked(
            torch.from_numpy(acapella_sig).float(), sr=TARGET_SR
        )
        emissions_mix_vb = extract_emissions_chunked(
            torch.from_numpy(mix_vb_sig).float(), sr=TARGET_SR
        )
        n_frames = min(emissions_acapella.shape[0], emissions_mix_vb.shape[0])
        emissions_blend = torch.logsumexp(
            torch.stack(
                [
                    emissions_acapella[:n_frames] + math.log(0.70),
                    emissions_mix_vb[:n_frames] + math.log(0.30),
                ],
                dim=0,
            ),
            dim=0,
        )
        fps = float(n_frames) / dur if dur > 0 else FPS
        log(
            f"[4] dual-stream emissions shape={tuple(emissions_blend.shape)} fps={fps:.2f}"
        )

        # --- Step 5: Independent Per-Line <star>-CTC Word Cementing ---
        all_words: List[Dict[str, Any]] = []
        blend_sig = (0.70 * acapella_sig + 0.30 * mix_vb_sig).astype(np.float32)

        for li, (ws, we) in enumerate(line_word_slices):
            if we <= ws:
                continue
            win = windows[li]
            lw = flat_words[ws:we]
            lr = flat_roman[ws:we]
            lp = flat_paren[ws:we]
            res = self.cementer.cement_line(
                lw,
                lr,
                lp,
                float(win["start_sec"]),
                float(win["end_sec"]),
                emissions_blend,
                fps,
                blend_sig,
                sr,
            )
            for idx_in_line, r in enumerate(res):
                r["word"] = lw[idx_in_line]
                r["romanized"] = lr[idx_in_line]
                r["is_parenthetical"] = bool(lp[idx_in_line])
                r["line_index"] = li
                r["duration_cs"] = max(
                    3, int(round((float(r["end_sec"]) - float(r["start_sec"])) * 100))
                )
                all_words.append(r)

        cemented_pass1 = sum(1 for w in all_words if w.get("is_cemented", False))
        log(
            f"[5] initial cemented(>= {self.word_target})={cemented_pass1}/{len(all_words)}"
        )

        # --- Step 6: Recursive Surround-and-Fill Discrepancy Resolution ---
        self._resolve_discrepancies_recursive(
            all_words=all_words,
            flat_roman=flat_roman,
            windows=windows,
            emissions_acapella=emissions_acapella[:n_frames],
            emissions_mix_vb=emissions_mix_vb[:n_frames],
            fps=fps,
            acapella_sig=acapella_sig,
            mix_vb_sig=mix_vb_sig,
            sr=sr,
            dur=dur,
            log=log,
        )

        # --- Step 6B: 2ms Sub-Frame Consonant/Sibilant Transient Snapper ---
        snapped = snap_words_transients_2ms(all_words, waveform=blend_sig, sr=sr)
        for idx, sw in enumerate(snapped):
            all_words[idx]["start_sec"] = sw["start_sec"]
            all_words[idx]["end_sec"] = sw["end_sec"]
            all_words[idx]["duration_cs"] = sw["duration_cs"]

        for li, (ws, we) in enumerate(line_word_slices):
            if we > ws:
                l_words = all_words[ws:we]
                windows[li]["start_sec"] = round(float(l_words[0]["start_sec"]), 4)
                windows[li]["end_sec"] = round(float(l_words[-1]["end_sec"]), 4)

        wcount = len(all_words)
        cemented_end = sum(
            1 for w in all_words if float(w["confidence"]) >= self.word_target
        )
        for w in all_words:
            w["is_cemented"] = bool(float(w["confidence"]) >= self.word_target)
        uncemented_end = wcount - cemented_end
        log(f"[6] final cemented={cemented_end}/{wcount} uncemented={uncemented_end}")

        # --- Step 7: 1:1 Export & Anti-Fabrication Verification Gate ---
        self._export(lines, all_words, windows, out_dir, dur)

        project_name = Path(mp3_path).parent.name or "Project"
        manifest: Dict[str, Any] = {
            "project": project_name,
            "engine": "greenfield_v4_3_adaptive_mms_fa",
            "tau_song": round(self.tau_song, 4),
            "word_target": self.word_target,
            "total_duration_sec": round(dur, 4),
            "total_input_lines": len(lines),
            "total_words": wcount,
            "cemented_words_ge_95_count": cemented_end,
            "uncemented_words_count": uncemented_end,
            "asr_words": len(asr_words),
            "words": all_words,
            "lines": [
                {
                    "index": w["index"],
                    "start_sec": round(float(w["start_sec"]), 4),
                    "end_sec": round(float(w["end_sec"]), 4),
                    "text": lines[w["index"] - 1],
                }
                for w in windows
            ],
        }

        try:
            verification = self.verifier.verify(manifest, mix_vb_sig, sr)
            manifest["verification"] = verification
        except Exception as exc:
            manifest["verification"] = {"passed": False, "error": str(exc)}

        mpath = os.path.join(out_dir, "alignment_manifest.json")
        with open(mpath, "w", encoding="utf-8", newline="\n") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        log(f"[7] wrote {mpath}")
        return manifest

    # ------------------------------------------------------------------ #
    def _export(
        self,
        lines: Sequence[str],
        all_words: Sequence[Dict[str, Any]],
        windows: Sequence[Dict[str, Any]],
        out_dir: str,
        dur: float,
    ) -> None:
        ass_path = os.path.join(out_dir, "karaoke.ass")
        synced_path = os.path.join(out_dir, "lyrics_synced.txt")
        write_ass(all_words, ass_path, title="TikTok Wizard Karaoke")
        export_lrc(all_words, synced_path, lines=lines)
