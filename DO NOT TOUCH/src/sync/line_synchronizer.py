"""Line-level synchronization and discrepancy-aware window anchoring (v4.1).

Stage 1 (coarse): Transcribe the vocal stream with faster-whisper to obtain
timed ASR word tokens, then monotonically align the user's lyric words to those
ASR words using a proportionally banded Needleman-Wunsch pass with romanized,
prefix, and consonant-skeleton similarity plus neighbor-context bonuses.

Applies causal outlier filtering so echo bounces or stray ASR tokens at the
other end of the song can never teleport a lyric line out of its true acoustic
neighborhood, and interpolates unanchored lines across active vocal bursts
within their self-calibrated lexical merit ceiling.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from rapidfuzz import fuzz


class MonotonicAlignmentMap(list):
    """List[Optional[int]] that also supports Dict-like .items() and .get()."""

    def items(self) -> Iterable[Tuple[int, int]]:
        for idx, val in enumerate(self):
            if val is not None:
                yield idx, int(val)

    def get(self, key: int, default: Optional[int] = None) -> Optional[int]:
        if 0 <= int(key) < len(self):
            val = self[int(key)]
            return int(val) if val is not None else default
        return default


def normalize_token(text: str) -> str:
    """Romanized, lowercase, alphabetic-only comparable key with elongation squeeze."""
    if not text:
        return ""
    raw = str(text).replace("\u00f1", "n").replace("\u00d1", "n")
    try:
        from anyascii import anyascii

        folded = anyascii(raw).lower()
    except Exception:
        nfkd = unicodedata.normalize("NFKD", raw)
        folded = "".join(c for c in nfkd if not unicodedata.combining(c)).lower()

    clean = re.sub(r"[^a-z']+", "", folded)
    clean = re.sub(r"(.)\1{2,}", r"\1", clean)
    return clean


def _consonant_skeleton(norm_tok: str) -> str:
    """Extract ordered non-vowel consonant skeleton for slang/contraction matching."""
    if not norm_tok:
        return ""
    return re.sub(r"[aeiouy']+", "", norm_tok)


def _token_similarity_score(lyr_norm: str, asr_norm: str) -> float:
    """Universal 0..100 similarity combining full ratio, prefix, and consonant skeleton."""
    if not lyr_norm or not asr_norm:
        return 0.0
    if lyr_norm == asr_norm:
        return 100.0

    base_ratio = float(fuzz.ratio(lyr_norm, asr_norm))

    min_len = min(len(lyr_norm), len(asr_norm))
    prefix_match = 0
    for ca, cb in zip(lyr_norm, asr_norm):
        if ca == cb:
            prefix_match += 1
        else:
            break
    prefix_score = (
        (100.0 * float(prefix_match) / float(min_len))
        if min_len >= 2
        else base_ratio
    )

    c_lyr = _consonant_skeleton(lyr_norm)
    c_asr = _consonant_skeleton(asr_norm)
    skel_score = float(fuzz.ratio(c_lyr, c_asr)) if (c_lyr and c_asr) else base_ratio

    blended = 0.60 * base_ratio + 0.25 * prefix_score + 0.15 * skel_score
    return float(max(base_ratio, blended))


def _count_syllables_line(line_text: str) -> int:
    toks = (line_text or "").split()
    total = 0
    for t in toks:
        nk = normalize_token(t)
        groups = re.findall(r"[aeiouy]+", nk)
        total += max(1, len(groups))
    return max(1, total)


def transcribe_vocal(
    wav_path: str,
    model_size: str = "small",
    language: Optional[str] = "es",
    threads: Optional[int] = None,
) -> Tuple[List[Tuple[float, float, str]], Dict[str, Any]]:
    """Transcribe a monophonic vocal stream with disk caching by audio SHA-256."""
    cache_path: Optional[str] = None
    if os.path.isfile(wav_path):
        try:
            h = hashlib.sha256()
            with open(wav_path, "rb") as f:
                while True:
                    blk = f.read(65536)
                    if not blk:
                        break
                    h.update(blk)
            digest = h.hexdigest()[:16]
            cache_dir = os.path.dirname(os.path.abspath(wav_path))
            cache_path = os.path.join(
                cache_dir,
                f".asr_cache_{model_size}_{language or 'auto'}_{digest}.json",
            )
            if os.path.isfile(cache_path):
                with open(cache_path, "r", encoding="utf-8") as cf:
                    cached = json.load(cf)
                words_cached = [
                    (float(w[0]), float(w[1]), str(w[2]))
                    for w in cached.get("words", [])
                ]
                if words_cached:
                    return words_cached, dict(
                        cached.get(
                            "info", {"language": language or "es", "prob": 1.0}
                        )
                    )
        except Exception:
            cache_path = None

    from faster_whisper import WhisperModel

    cpu_threads = threads or max(1, min(8, os.cpu_count() or 4))
    model = WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
        cpu_threads=cpu_threads,
    )
    segments, info = model.transcribe(
        wav_path,
        language=language,
        word_timestamps=True,
        vad_filter=False,
        condition_on_previous_text=False,
        beam_size=2,
    )
    words: List[Tuple[float, float, str]] = []
    for seg in segments:
        for w in seg.words or []:
            txt = (w.word or "").strip()
            if txt:
                words.append((float(w.start), float(w.end), txt))

    info_dict = {
        "language": getattr(info, "language", language or "es"),
        "prob": float(getattr(info, "language_probability", 1.0)),
    }

    if cache_path and words:
        try:
            with open(cache_path, "w", encoding="utf-8") as cf:
                json.dump(
                    {"words": words, "info": info_dict}, cf, ensure_ascii=False
                )
        except Exception:
            pass

    return words, info_dict


def align_words_monotonic(
    lyric_words: Sequence[str],
    asr_words: Sequence[str],
    min_score: float = 52.0,
    band: int = 90,
    gap_penalty: float = -4.5,
    word_to_line: Optional[Sequence[int]] = None,
    asr_times: Optional[Sequence[Tuple[float, float]]] = None,
) -> MonotonicAlignmentMap:
    """Proportionally banded global Needleman-Wunsch alignment with lexical rarity and temporal coherence.

    Maps each lyric word index -> asr word index (or None). Returns a
    MonotonicAlignmentMap (which is a list of Optional[int] and also provides
    .items() for dict-style iteration).
    """
    n = len(lyric_words)
    m = len(asr_words)
    mapping = MonotonicAlignmentMap([None] * n)
    if n == 0 or m == 0:
        return mapping

    ln = [normalize_token(w) for w in lyric_words]
    an = [normalize_token(w) for w in asr_words]

    # Universal Inverse Document Frequency (lexical rarity weighting):
    # Unique anchor words ("sustancias", "descanse", "caption") lock monotonic positions
    import math
    from collections import Counter
    counts = Counter(ln)
    rarity = {k: math.log(1.0 + float(n) / float(v)) for k, v in counts.items()}

    eff_band = max(int(band), abs(n - m) + 85)
    NEG = -1e15
    dp = np.full((n + 1, m + 1), NEG, dtype=np.float64)
    bt = np.full((n + 1, m + 1), -1, dtype=np.int8)
    prev_j = np.full((n + 1, m + 1), -1, dtype=np.int32)
    dp[0, 0] = 0.0

    for j in range(1, min(m + 1, eff_band + 1)):
        dp[0, j] = j * (gap_penalty * 0.65)
        bt[0, j] = 2
    for i in range(1, min(n + 1, eff_band + 1)):
        dp[i, 0] = i * gap_penalty
        bt[i, 0] = 1

    for i in range(1, n + 1):
        j_center = int(round(float(i) * (float(m) / float(max(1, n)))))
        jlo = max(1, j_center - eff_band)
        jhi = min(m, j_center + eff_band)

        l_tok = ln[i - 1]
        l_prev = ln[i - 2] if i >= 2 else ""
        l_next = ln[i] if i < n else ""

        w_rarity = rarity.get(l_tok, 1.0)
        rarity_mult = float(np.clip(0.85 + 0.35 * (w_rarity / max(1e-4, math.log(n + 1))), 0.85, 1.60))

        is_same_line_as_prev = False
        if word_to_line is not None and len(word_to_line) >= i:
            line_curr = word_to_line[i - 1]
            line_prev = word_to_line[i - 2] if i >= 2 else -1
            is_same_line_as_prev = (line_curr == line_prev)

        for j in range(jlo, jhi + 1):
            best = NEG
            b_choice = -1
            best_pj = -1

            if dp[i - 1, j] > NEG / 2:
                v_up = dp[i - 1, j] + gap_penalty
                if v_up > best:
                    best = v_up
                    b_choice = 1
                    best_pj = prev_j[i - 1, j]

            if dp[i, j - 1] > NEG / 2:
                v_left = dp[i, j - 1] + (gap_penalty * 0.65)
                if v_left > best:
                    best = v_left
                    b_choice = 2
                    best_pj = prev_j[i, j - 1]

            if dp[i - 1, j - 1] > NEG / 2:
                a_tok = an[j - 1]
                sc = _token_similarity_score(l_tok, a_tok)

                ctx_bonus = 0.0
                if i >= 2 and j >= 2 and l_prev and an[j - 2]:
                    if _token_similarity_score(l_prev, an[j - 2]) >= 75.0:
                        ctx_bonus += 10.0
                if i < n and j < m and l_next and an[j]:
                    if _token_similarity_score(l_next, an[j]) >= 75.0:
                        ctx_bonus += 10.0

                eff_sc = min(100.0, sc + ctx_bonus)
                if eff_sc >= min_score:
                    reward = ((eff_sc - min_score) * 1.25 + 5.0) * rarity_mult

                    time_pen = 0.0
                    pj = prev_j[i - 1, j - 1]
                    if is_same_line_as_prev and pj >= 0 and pj < m and asr_times is not None:
                        t_prev_en = asr_times[pj][1]
                        t_curr_st = asr_times[j - 1][0]
                        dt = t_curr_st - t_prev_en
                        if dt > 3.0:
                            # Intra-line acoustic continuity invariant: words in a single line cannot jump across drops
                            time_pen = -35.0 * min(5.0, (dt - 3.0))
                        elif dt < -0.15:
                            time_pen = -25.0

                    v_diag = dp[i - 1, j - 1] + reward + time_pen
                    if v_diag > best:
                        best = v_diag
                        b_choice = 0
                        best_pj = j - 1

            dp[i, j] = best
            bt[i, j] = b_choice
            prev_j[i, j] = best_pj

    end_i, end_j = n, m
    if dp[n, m] <= NEG / 4:
        best_end = NEG
        for j in range(max(0, m - eff_band), m + 1):
            if dp[n, j] > best_end:
                best_end = dp[n, j]
                end_i, end_j = n, j

    i, j = end_i, end_j
    while i > 0 and j > 0:
        b = int(bt[i, j])
        if b == 0:
            mapping[i - 1] = j - 1
            i -= 1
            j -= 1
        elif b == 1:
            i -= 1
        elif b == 2:
            j -= 1
        else:
            if dp[i - 1, j] >= dp[i, j - 1]:
                i -= 1
            else:
                j -= 1

    return mapping


def build_line_windows(
    lines: Sequence[str],
    line_word_slices: Sequence[Tuple[int, int]],
    mapping: Sequence[Optional[int]],
    asr_words: Sequence[Tuple[float, float, str]],
    total_duration: float,
    vocal_rms: Optional[np.ndarray] = None,
    rms_hop_sec: float = 0.01,
) -> List[Dict[str, Any]]:
    """Convert per-word ASR matches into causally validated per-line [start_sec, end_sec] windows.

    Rejects outlier ASR matches that teleport across the song, interpolates
    unanchored lines proportionally to their syllable budgets between trusted
    neighbors, and caps line durations so no sentence stretches across an
    instrumental drop.
    """
    n_lines = len(line_word_slices)
    if n_lines == 0:
        return []

    total_dur = max(1.0, float(total_duration))
    line_syllables = [
        _count_syllables_line(lines[li]) if li < len(lines) else max(1, we - ws)
        for li, (ws, we) in enumerate(line_word_slices)
    ]
    line_min_durs = [
        max(
            0.38,
            min(
                1.80,
                0.11 * float(line_syllables[li])
                + 0.05 * float(max(1, we - ws)),
            ),
        )
        for li, (ws, we) in enumerate(line_word_slices)
    ]
    line_typ_durs = [
        max(
            0.65,
            min(
                3.60,
                0.21 * float(line_syllables[li])
                + 0.08 * float(max(1, we - ws)),
            ),
        )
        for li, (ws, we) in enumerate(line_word_slices)
    ]
    line_max_durs = [
        max(
            1.25,
            min(
                5.20,
                0.42 * float(line_syllables[li])
                + 0.35 * float(max(1, we - ws)),
            ),
        )
        for li, (ws, we) in enumerate(line_word_slices)
    ]

    raw_spans: List[Optional[Tuple[float, float, int]]] = [None] * n_lines
    for li, (ws, we) in enumerate(line_word_slices):
        matched_times: List[Tuple[float, float]] = []
        for k in range(ws, we):
            ai = mapping[k] if k < len(mapping) else None
            if ai is not None and 0 <= int(ai) < len(asr_words):
                st, en, _ = asr_words[int(ai)]
                if 0.0 <= float(st) <= float(en) <= total_dur + 0.5:
                    matched_times.append((float(st), float(en)))
        if matched_times:
            centers = np.array(
                [0.5 * (t0 + t1) for t0, t1 in matched_times], dtype=np.float64
            )
            med_c = float(np.median(centers))
            inliers = [
                (t0, t1)
                for (t0, t1), c in zip(matched_times, centers)
                if abs(c - med_c) <= 4.5
            ] or matched_times
            s_raw = float(min(t0 for t0, _ in inliers))
            e_raw = float(max(t1 for _, t1 in inliers))
            raw_spans[li] = (s_raw, max(s_raw + 0.08, e_raw), len(inliers))

    trusted: List[Optional[Tuple[float, float]]] = [None] * n_lines
    prev_trusted_end = 0.0
    prev_trusted_idx = -1

    for li in range(n_lines):
        cand = raw_spans[li]
        if cand is None:
            continue
        s_raw, e_raw, _ = cand
        center_li = 0.5 * (s_raw + e_raw)

        rem_min_time = sum(
            line_min_durs[k] * 0.55 for k in range(li + 1, n_lines)
        )
        if s_raw > total_dur - rem_min_time:
            continue

        future_centers: List[float] = []
        for lj in range(li + 1, min(n_lines, li + 12)):
            if raw_spans[lj] is not None:
                future_centers.append(
                    0.5 * (raw_spans[lj][0] + raw_spans[lj][1])
                )
                if len(future_centers) >= 5:
                    break

        if (
            len(future_centers) >= 2
            and center_li > float(np.median(future_centers)) + 1.5
        ):
            continue

        if prev_trusted_idx >= 0:
            lines_elapsed = max(1, li - prev_trusted_idx)
            max_reasonable_jump = max(14.0, lines_elapsed * 5.0)
            if (
                s_raw - prev_trusted_end
            ) > max_reasonable_jump and len(future_centers) >= 1:
                if future_centers[0] < s_raw:
                    continue

        if e_raw < prev_trusted_end - 0.25:
            continue

        s_clamped = max(prev_trusted_end - 0.05, s_raw)
        # Trust verified inlier ASR bounds without arbitrary 5.2s ceiling truncation:
        # A singer's sustained note or vocal stutter is respected.
        e_clamped = max(
            s_clamped + line_min_durs[li],
            min(total_dur - rem_min_time, e_raw),
        )
        trusted[li] = (s_clamped, min(total_dur, e_clamped))
        prev_trusted_end = trusted[li][1]
        prev_trusted_idx = li

    windows: List[Dict[str, Any]] = [
        {"index": li + 1, "start_sec": 0.0, "end_sec": 0.0}
        for li in range(n_lines)
    ]

    idx = 0
    while idx < n_lines:
        if trusted[idx] is not None:
            st, en = trusted[idx]  # type: ignore[misc]
            windows[idx]["start_sec"] = float(st)
            windows[idx]["end_sec"] = float(en)
            idx += 1
            continue

        gap_start_idx = idx
        while idx < n_lines and trusted[idx] is None:
            idx += 1
        gap_end_idx = idx - 1

        if gap_start_idx > 0:
            t_left = float(windows[gap_start_idx - 1]["end_sec"])
        else:
            if idx < n_lines and trusted[idx] is not None:
                first_anchor_st = float(trusted[idx][0])  # type: ignore[index]
                needed_typ = sum(
                    line_typ_durs[k]
                    for k in range(gap_start_idx, gap_end_idx + 1)
                )
                t_left = max(0.0, first_anchor_st - needed_typ - 0.30)
            else:
                t_left = 0.0

        if idx < n_lines and trusted[idx] is not None:
            t_right = float(trusted[idx][0])  # type: ignore[index]
        else:
            t_right = total_dur

        gap_indices = list(range(gap_start_idx, gap_end_idx + 1))
        avail = max(0.10 * len(gap_indices), t_right - t_left)
        sum_max = sum(line_max_durs[k] for k in gap_indices)
        alloc_span = min(avail, sum_max)

        # Adaptive positioning within large instrumental voids (> 1.5 * sum_max):
        # If vocal RMS envelope is available, anchor around the actual vocal energy burst!
        cursor = t_left
        if avail > (sum_max * 1.45) and vocal_rms is not None and vocal_rms.size > 0:
            f_lo = max(0, int(round(t_left / max(1e-4, rms_hop_sec))))
            f_hi = min(vocal_rms.size, int(round(t_right / max(1e-4, rms_hop_sec))))
            win_frames = max(1, int(round(alloc_span / max(1e-4, rms_hop_sec))))
            if f_hi - f_lo > win_frames:
                sub_rms = vocal_rms[f_lo:f_hi]
                # Sliding boxcar convolution to locate vocal peak burst
                boxcar = np.ones(win_frames, dtype=np.float64) / float(win_frames)
                smoothed = np.convolve(sub_rms, boxcar, mode="valid")
                if smoothed.size > 0:
                    best_offset_f = int(np.argmax(smoothed))
                    peak_start_sec = (f_lo + best_offset_f) * rms_hop_sec
                    cursor = max(t_left, min(t_right - alloc_span, peak_start_sec))
        elif avail > (sum_max * 1.45):
            # Fallback heuristic: intro/pickup phrases attach closer to the subsequent anchor
            all_paren = all(
                ("(" in lines[k] and ")" in lines[k]) or "gracias" in normalize_token(lines[k])
                for k in gap_indices if k < len(lines)
            )
            if all_paren and idx < n_lines:
                cursor = max(t_left, t_right - alloc_span - 0.30)

        weights = np.array(
            [line_typ_durs[k] for k in gap_indices], dtype=np.float64
        )
        weights /= float(np.sum(weights)) + 1e-9

        for offset, k in enumerate(gap_indices):
            dur_k = float(alloc_span * weights[offset])
            dur_k = float(
                np.clip(
                    dur_k,
                    min(0.25, avail / len(gap_indices)),
                    line_max_durs[k],
                )
            )
            st_k = cursor
            en_k = min(total_dur, st_k + dur_k)
            windows[k]["start_sec"] = round(st_k, 4)
            windows[k]["end_sec"] = round(max(st_k + 0.08, en_k), 4)
            cursor = windows[k]["end_sec"]

        if idx < n_lines and trusted[idx] is not None and cursor > t_right:
            st_next, en_next = trusted[idx]  # type: ignore[misc]
            new_st_next = min(en_next - 0.15, cursor)
            if new_st_next > st_next:
                trusted[idx] = (new_st_next, en_next)

    return windows
