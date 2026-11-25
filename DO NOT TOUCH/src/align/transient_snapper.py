"""Transient snap-and-1-to-1 export: 2ms consonant/sibilant boundary refinement (v4.1).

Refines consonant-onset and sibilant-release boundaries at 2ms (0.2cs) precision
strictly within surrounding cemented word bounds, and exports 1:1 unmerged ASS
Dialogue lines (1 lyric line = 1 ASS Dialogue line, never 1 word per Dialogue line).
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import scipy.signal


TARGET_SR: int = 16000
_CONSONANT_STARTS = set("bcdfghjklmnpqrstvwxz")
_CONSONANT_ENDS = set("szxcftkpdnrl")


def _clean_roman(word: str) -> str:
    t = unicodedata.normalize("NFKD", (word or "").lower())
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return re.sub(r"[^a-z]+", "", t)


def _zero_phase_high_band(sig: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Extract 3800-7800 Hz consonant/sibilant transient envelope."""
    if sig is None or sig.size < 64:
        return np.zeros(0, dtype=np.float32)
    nyq = 0.5 * float(sr)
    lo = max(0.05, min(0.90, 3800.0 / nyq))
    hi = max(lo + 0.04, min(0.98, 7800.0 / nyq))
    sos = scipy.signal.butter(3, [lo, hi], btype="bandpass", output="sos")
    filtered = scipy.signal.sosfiltfilt(sos, sig.astype(np.float64))
    return np.abs(filtered).astype(np.float32)


def snap_words_transients_2ms(
    cemented_words: List[Dict[str, Any]],
    waveform: Optional[np.ndarray] = None,
    sr: int = TARGET_SR,
) -> List[Dict[str, Any]]:
    """Refine word start/end boundaries at 2ms resolution inside cemented neighbor bounds."""
    if not cemented_words:
        return []

    refined: List[Dict[str, Any]] = [dict(w) for w in cemented_words]
    n_words = len(refined)

    has_audio = (
        waveform is not None
        and isinstance(waveform, np.ndarray)
        and waveform.size >= sr
    )
    sib_env = (
        _zero_phase_high_band(waveform, sr=sr)
        if has_audio
        else np.zeros(0, dtype=np.float32)
    )
    sib_thresh = 0.0
    dur_total = (
        float(sib_env.size) / float(sr)
        if sib_env.size > 0
        else float(refined[-1].get("end_sec", 0.0)) + 1.0
    )
    if sib_env.size > 0:
        med = float(np.median(sib_env) + 1e-6)
        sib_thresh = max(med * 2.6, float(np.percentile(sib_env, 78)) * 0.45)

    step_samples = max(1, int(round(0.002 * sr)))  # 2ms = 32 samples @ 16kHz

    for idx, w in enumerate(refined):
        st = float(w.get("start_sec", 0.0))
        en = float(w.get("end_sec", st + 0.05))
        roman = _clean_roman(str(w.get("romanized") or w.get("word", "")))

        hard_min = float(refined[idx - 1].get("end_sec", 0.0)) if idx > 0 else 0.0
        hard_max = (
            float(refined[idx + 1].get("start_sec", dur_total))
            if idx + 1 < n_words
            else dur_total
        )

        if sib_env.size > 0 and roman:
            # 1. Leading plosive/fricative onset snap (up to 40ms left, quantized to 2ms grid)
            if roman[0] in _CONSONANT_STARTS:
                scan_lo = max(hard_min, st - 0.040)
                i_lo = max(0, int(round(scan_lo * sr)))
                i_hi = min(sib_env.size, int(round(st * sr)))
                if i_hi - i_lo >= step_samples:
                    seg = sib_env[i_lo:i_hi]
                    above = np.where(seg >= sib_thresh)[0]
                    if above.size > 0:
                        cand_st = scan_lo + (float(above[0] // step_samples) * 0.002)
                        if hard_min <= cand_st < en - 0.030:
                            st = cand_st

            # 2. Trailing sibilant/consonant release snap (up to 50ms right, quantized to 2ms grid)
            if roman[-1] in _CONSONANT_ENDS:
                scan_hi = min(hard_max, en + 0.050)
                i_lo = max(0, int(round(en * sr)))
                i_hi = min(sib_env.size, int(round(scan_hi * sr)))
                if i_hi - i_lo >= step_samples:
                    seg = sib_env[i_lo:i_hi]
                    above = np.where(seg >= sib_thresh)[0]
                    if above.size > 0:
                        cand_en = en + (float((above[-1] // step_samples) + 1) * 0.002)
                        if st + 0.030 < cand_en <= hard_max:
                            en = cand_en
        elif roman:
            # Deterministic 2ms grid micro-snap within available slack when waveform is omitted
            if roman[0] in _CONSONANT_STARTS and (st - hard_min) >= 0.006:
                st = max(hard_min, round((st - 0.004) / 0.002) * 0.002)
            if roman[-1] in _CONSONANT_ENDS and (hard_max - en) >= 0.006:
                en = min(hard_max, round((en + 0.004) / 0.002) * 0.002)

        st = max(hard_min, round(st / 0.002) * 0.002)
        en = max(
            st + 0.030,
            min(hard_max if hard_max > st + 0.030 else en, round(en / 0.002) * 0.002),
        )

        w["start_sec"] = round(float(st), 4)
        w["end_sec"] = round(float(en), 4)
        w["duration_cs"] = max(3, int(round((w["end_sec"] - w["start_sec"]) * 100.0)))

    return refined


def transient_snap(
    ass_lines: List[Dict[str, Any]],
    cemented_words: List[Dict[str, Any]],
    waveform: Optional[np.ndarray] = None,
    sr: int = TARGET_SR,
) -> List[Dict[str, Any]]:
    """Apply 2ms transient snap refinement to ASS dialogue line dicts using cemented words."""
    snapped_words = snap_words_transients_2ms(cemented_words, waveform=waveform, sr=sr)

    # Group snapped words by line_index if available
    by_line: Dict[int, List[Dict[str, Any]]] = {}
    for w in snapped_words:
        if "line_index" in w:
            li = int(w["line_index"])
            by_line.setdefault(li, []).append(w)

    refined_lines: List[Dict[str, Any]] = []
    for li, line in enumerate(ass_lines):
        adjusted = dict(line)
        l_words = by_line.get(li)

        if l_words:
            new_st_sec = float(l_words[0]["start_sec"])
            new_en_sec = float(l_words[-1]["end_sec"])
            if "start_ms" in adjusted or "end_ms" in adjusted:
                adjusted["start_ms"] = int(round(new_st_sec * 1000.0))
                adjusted["end_ms"] = int(round(new_en_sec * 1000.0))
            if "start_sec" in adjusted or "end_sec" in adjusted:
                adjusted["start_sec"] = round(new_st_sec, 4)
                adjusted["end_sec"] = round(new_en_sec, 4)
        else:
            # Match by temporal overlap if line_index was not populated on words
            has_ms = "start_ms" in adjusted or "end_ms" in adjusted
            l_st = (
                float(adjusted.get("start_ms", 0)) / 1000.0
                if has_ms
                else float(adjusted.get("start_sec", 0.0))
            )
            l_en = (
                float(adjusted.get("end_ms", 0)) / 1000.0
                if has_ms
                else float(adjusted.get("end_sec", l_st))
            )
            overlapping = [
                w
                for w in snapped_words
                if float(w.get("end_sec", 0.0)) >= l_st - 0.05
                and float(w.get("start_sec", 0.0)) <= l_en + 0.05
            ]
            if overlapping:
                new_st_sec = min(l_st, float(overlapping[0]["start_sec"]))
                new_en_sec = max(l_en, float(overlapping[-1]["end_sec"]))
                if has_ms:
                    adjusted["start_ms"] = int(round(new_st_sec * 1000.0))
                    adjusted["end_ms"] = int(round(new_en_sec * 1000.0))
                if "start_sec" in adjusted or "end_sec" in adjusted:
                    adjusted["start_sec"] = round(new_st_sec, 4)
                    adjusted["end_sec"] = round(new_en_sec, 4)

        refined_lines.append(adjusted)

    return refined_lines


def _fmt_ass_time(seconds: float) -> str:
    s = max(0.0, float(seconds))
    total_cs = int(round(s * 100.0))
    h = total_cs // 360000
    rem = total_cs % 360000
    m = rem // 6000
    rem %= 6000
    sec = rem // 100
    cs = rem % 100
    return f"{h}:{m:02d}:{sec:02d}.{cs:02d}"


def export_ass(
    cemented_words: List[Dict[str, Any]],
    output_path: str,
    lines: Optional[Sequence[str]] = None,
) -> None:
    """Export cemented word timings to 1:1 unmerged ASS format (1 line = 1 Dialogue)."""
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    has_line_idx = any("line_index" in w for w in cemented_words)

    if has_line_idx:
        for w in cemented_words:
            li = int(w.get("line_index", 0))
            grouped.setdefault(li, []).append(w)
    else:
        for idx, w in enumerate(cemented_words):
            grouped[idx] = [w]

    line_indices = sorted(grouped.keys())
    dialogue_lines: List[str] = []

    for li in line_indices:
        lwords = grouped[li]
        if not lwords:
            continue
        start_sec = float(lwords[0].get("start_sec", 0.0))
        end_sec = max(
            start_sec + 0.05, float(lwords[-1].get("end_sec", start_sec + 0.20))
        )

        parts: List[str] = []
        prev_end = start_sec
        for w in lwords:
            w_st = float(w.get("start_sec", prev_end))
            w_en = max(w_st + 0.03, float(w.get("end_sec", w_st + 0.10)))
            gap_cs = int(round((w_st - prev_end) * 100.0))
            if gap_cs >= 3:
                parts.append(f"{{\\k{gap_cs}}}")
            cs = max(3, int(round((w_en - w_st) * 100.0)))
            parts.append(f"{{\\k{cs}}}{w.get('word', '')}")
            prev_end = w_en

        dialogue_lines.append(
            f"Dialogue: 0,{_fmt_ass_time(start_sec)},{_fmt_ass_time(end_sec)},Default,,0,0,0,,{' '.join(parts)}"
        )

    header = [
        "[Script Info]",
        "Title: TikTok Wizard Karaoke Output",
        "ScriptType: v4.00+",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Default,Arimo,68,&H0000FFFF,&H00FFFFFF,&H00000000,&H64000000,"
        "-1,0,0,0,100,100,0,0,1,3,0,2,40,40,180,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        for h_line in header:
            f.write(h_line + "\n")
        for d_line in dialogue_lines:
            f.write(d_line + "\n")
