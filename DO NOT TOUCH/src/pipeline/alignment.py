#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShotCut TikTok Automator — Forced Alignment Engine

Uses faster-whisper with word-level timestamps, combined with global
Needleman-Wunsch sequence alignment to map user-written TikTok lyrics
to the exact audio timing.

Core principle: Lyrics are the visual ground truth; audio is aligned to the text.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .schema import KaraokeEntry, KaraokeMeta, KaraokeWord, LyricLine, LyricWord

try:
    from faster_whisper import WhisperModel as _WhisperModel
    WHISPER_AVAILABLE = True
except Exception:
    _WhisperModel = None
    WHISPER_AVAILABLE = False


def normalize_word(text: str) -> str:
    """Lowercase + strip accents and punctuation for lyrics matching."""
    if not text:
        return ""
    # Strip accents
    nfkd = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    clean = re.sub(r"[^a-zA-Z0-9]", "", stripped.lower())
    # Common Spanish colloquialisms and Whisper reverb mishearings
    if clean in ("pa", "pal"):
        return "para"
    if clean in ("ciancia", "ciancias", "gracia", "gracias"):
        return "gracias"
    if clean in ("empadezcance", "descanse"):
        return "descanse"
    return clean


def word_similarity(w1: str, w2: str) -> float:
    """Return similarity score 0.0 - 1.0 between two words."""
    n1, n2 = normalize_word(w1), normalize_word(w2)
    if not n1 or not n2:
        return 0.0
    if n1 == n2:
        return 1.0
    if n1 in n2 or n2 in n1:
        return 0.85
    return SequenceMatcher(None, n1, n2).ratio()


def score_pair(w1: str, w2: str) -> float:
    """Scoring function for Needleman-Wunsch alignment."""
    n1, n2 = normalize_word(w1), normalize_word(w2)
    if not n1 or not n2:
        return -0.5
    if n1 == n2:
        weight = 1.0 if len(n1) <= 2 else 2.0
        return 3.0 * weight
    ratio = SequenceMatcher(None, n1, n2).ratio()
    if ratio >= 0.75:
        return 2.0 * ratio
    return -1.0


def transcribe_words(
    audio_path: Path,
    language: Optional[str] = "es",
    model_size: str = "small",
    device: str = "auto",
    compute_type: str = "int8",
) -> Optional[List[dict]]:
    """
    Run faster-whisper to extract words with precise start/end timestamps.
    Caches results in .whisper_cache.json next to the audio file for instant re-runs.
    """
    if not WHISPER_AVAILABLE or _WhisperModel is None:
        return None

    audio_path = Path(audio_path)
    # Check DO NOT TOUCH/_Projects/<name>/.data first, then fallback to local .data
    dnt_data = audio_path.resolve().parents[2] / "DO NOT TOUCH" / "_Projects" / audio_path.parent.name / ".data"
    dot_data = dnt_data if dnt_data.exists() else (audio_path.parent / ".data")
    cache_path = dot_data / ".whisper_cache.json"

    # 1. Check cache (look in dnt_data, then .data/, then legacy parent directory)
    legacy_cache = audio_path.parent / ".whisper_cache.json"
    effective_cache = cache_path if cache_path.exists() else (legacy_cache if legacy_cache.exists() else cache_path)

    if effective_cache.exists():
        try:
            cached_data = json.loads(effective_cache.read_text(encoding="utf-8"))
            words = []
            for seg in cached_data:
                for w in seg.get("words", []):
                    words.append({
                        "word": str(w["word"]).strip(),
                        "start": float(w["start"]),
                        "end": float(w["end"]),
                        "prob": float(w.get("prob", 1.0)),
                    })
            if words:
                print(f"Loaded {len(words)} transcribed words from cache.")
                return words
        except Exception:
            pass

    # 2. Run faster-whisper
    try:
        print(f"Loading Whisper model ({model_size}, {device}, {compute_type})...")
        model = _WhisperModel(model_size, device=device, compute_type=compute_type)
        print("Transcribing audio (extracting word-level timestamps)...")
        segments, _info = model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,
            condition_on_previous_text=False,
        )

        all_segments = []
        words: List[dict] = []

        for seg in segments:
            seg_words = []
            if seg.words:
                for w in seg.words:
                    w_dict = {
                        "word": str(w.word).strip(),
                        "start": round(float(w.start), 2),
                        "end": round(float(w.end), 2),
                        "prob": round(float(w.probability), 2),
                    }
                    seg_words.append(w_dict)
                    words.append(w_dict)

            all_segments.append({
                "text": seg.text.strip(),
                "start": round(float(seg.start), 2),
                "end": round(float(seg.end), 2),
                "words": seg_words,
            })

        # Save to cache in dot_data
        if all_segments:
            try:
                dot_data.mkdir(parents=True, exist_ok=True)
                cache_path = dot_data / ".whisper_cache.json"
                cache_path.write_text(json.dumps(all_segments, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass

        return words if words else None
    except Exception as e:
        print(f"Whisper transcription error: {e}")
        return None


def align_lyric_words_dp(
    all_lyric_words: List[dict],
    whisper_words: List[dict],
) -> List[dict]:
    """
    Global Needleman-Wunsch dynamic programming alignment between
    lyrics words and transcribed audio words.
    """
    N = len(all_lyric_words)
    M = len(whisper_words)

    if N == 0:
        return []
    if M == 0:
        return all_lyric_words

    dp = [[0.0] * (M + 1) for _ in range(N + 1)]
    backtrack = [[0] * (M + 1) for _ in range(N + 1)]

    GAP_PENALTY = -0.8

    for i in range(1, N + 1):
        dp[i][0] = i * GAP_PENALTY
        backtrack[i][0] = 2

    for j in range(1, M + 1):
        dp[0][j] = j * GAP_PENALTY
        backtrack[0][j] = 3

    for i in range(1, N + 1):
        lw = all_lyric_words[i - 1]["text"]
        for j in range(1, M + 1):
            ww = whisper_words[j - 1]["word"]
            match_score = score_pair(lw, ww)
            score_diag = dp[i - 1][j - 1] + match_score
            score_up = dp[i - 1][j] + GAP_PENALTY
            score_left = dp[i][j - 1] + GAP_PENALTY

            if score_diag >= score_up and score_diag >= score_left:
                dp[i][j] = score_diag
                backtrack[i][j] = 1
            elif score_up >= score_left:
                dp[i][j] = score_up
                backtrack[i][j] = 2
            else:
                dp[i][j] = score_left
                backtrack[i][j] = 3

    # Backtrack
    i, j = N, M
    alignment = []
    while i > 0 or j > 0:
        bt = backtrack[i][j]
        if bt == 1:
            alignment.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif bt == 2:
            alignment.append((i - 1, None))
            i -= 1
        elif bt == 3:
            alignment.append((None, j - 1))
            j -= 1
        else:
            break

    alignment.reverse()

    # Assign matches
    for l_idx, w_idx in alignment:
        if l_idx is not None:
            lw = all_lyric_words[l_idx]
            if w_idx is not None:
                ww = whisper_words[w_idx]
                sim = SequenceMatcher(
                    None, normalize_word(lw["text"]), normalize_word(ww["word"])
                ).ratio()
                if sim >= 0.55:
                    lw["start"] = ww["start"]
                    lw["end"] = max(ww["end"], ww["start"] + 0.1)
                    lw["matched"] = True
                    lw["confidence"] = sim
                    lw["matched_to"] = ww["word"]
                else:
                    lw["start"] = None
                    lw["end"] = None
                    lw["matched"] = False
                    lw["confidence"] = 0.0
            else:
                lw["start"] = None
                lw["end"] = None
                lw["matched"] = False
                lw["confidence"] = 0.0

    return all_lyric_words


def interpolate_and_sanitize(
    lyric_lines_grouped: Dict[int, List[dict]],
    total_duration: float,
) -> List[dict]:
    """
    Interpolate missing timestamps smoothly across all words and lines.
    Guarantees every single word has a valid, non-zero, monotonically
    increasing timestamp.
    """
    sorted_line_keys = sorted(lyric_lines_grouped.keys())

    # 1. Line-level bounds: establish anchor start and end for each line
    line_bounds: Dict[int, Tuple[float, float]] = {}

    for l_idx in sorted_line_keys:
        words = lyric_lines_grouped[l_idx]
        matched = [w for w in words if w.get("matched") and w.get("start") is not None]
        if matched:
            line_bounds[l_idx] = (matched[0]["start"], matched[-1]["end"])

    # Interpolate bounds for lines that had NO matches at all
    for idx_i, l_idx in enumerate(sorted_line_keys):
        if l_idx not in line_bounds:
            # Look backwards for previous known bound
            prev_end = 0.0
            for prev_k in reversed(sorted_line_keys[:idx_i]):
                if prev_k in line_bounds:
                    prev_end = line_bounds[prev_k][1]
                    break

            # Look forwards for next known bound
            next_start = total_duration
            for next_k in sorted_line_keys[idx_i + 1 :]:
                if next_k in line_bounds:
                    next_start = line_bounds[next_k][0]
                    break

            # Span available
            span = max(0.6, next_start - prev_end)
            est_dur = min(2.5, max(0.8, span * 0.5))
            est_start = prev_end + 0.1
            est_end = min(total_duration, est_start + est_dur)
            line_bounds[l_idx] = (est_start, est_end)

    # 2. Interpolate words within each line
    all_final_words: List[dict] = []
    prev_word_end = 0.0

    for l_idx in sorted_line_keys:
        words = lyric_lines_grouped[l_idx]
        lb_start, lb_end = line_bounds[l_idx]
        n_words = len(words)
        if n_words == 0:
            continue

        # If all words matched, use their timestamps directly
        # For unmatched words in the line, interpolate between matched neighbors
        # Find matched indices
        matched_indices = [i for i, w in enumerate(words) if w.get("matched") and w.get("start") is not None]

        if not matched_indices:
            # Evenly space all words in line across [lb_start, lb_end]
            dur_per_word = max(0.1, (lb_end - lb_start) / n_words)
            for i, w in enumerate(words):
                w_start = lb_start + i * dur_per_word
                w_end = w_start + dur_per_word
                w["start"] = w_start
                w["end"] = w_end
                w["confidence"] = 0.5
        else:
            # Fill before first matched
            first_m = matched_indices[0]
            if first_m > 0:
                first_m_start = words[first_m]["start"]
                avail = max(0.1, first_m_start - lb_start)
                dur = avail / first_m
                for i in range(first_m):
                    words[i]["start"] = lb_start + i * dur
                    words[i]["end"] = words[i]["start"] + dur
                    words[i]["confidence"] = 0.6

            # Fill between matched pairs
            for k in range(len(matched_indices) - 1):
                i1 = matched_indices[k]
                i2 = matched_indices[k + 1]
                if i2 - i1 > 1:
                    t1 = words[i1]["end"]
                    t2 = words[i2]["start"]
                    avail = max(0.1, t2 - t1)
                    dur = avail / (i2 - i1)
                    for idx, i in enumerate(range(i1 + 1, i2)):
                        words[i]["start"] = t1 + idx * dur
                        words[i]["end"] = words[i]["start"] + dur
                        words[i]["confidence"] = 0.6

            # Fill after last matched
            last_m = matched_indices[-1]
            if last_m < n_words - 1:
                last_m_end = words[last_m]["end"]
                avail = max(0.1, lb_end - last_m_end)
                dur = avail / (n_words - 1 - last_m)
                for idx, i in enumerate(range(last_m + 1, n_words)):
                    words[i]["start"] = last_m_end + idx * dur
                    words[i]["end"] = words[i]["start"] + dur
                    words[i]["confidence"] = 0.6

        # Enforce strict monotonicity and minimum duration
        for w in words:
            st = float(w["start"])
            en = float(w["end"])
            if st < prev_word_end:
                st = prev_word_end
            if en <= st:
                en = st + 0.1
            w["start"] = round(st, 2)
            w["end"] = round(en, 2)
            prev_word_end = w["end"]

            all_final_words.append({
                "word": w["text"],
                "start": w["start"],
                "end": w["end"],
                "line": l_idx,
                "confidence": w.get("confidence", 0.8),
            })

    return all_final_words


def align(
    audio_path: Path,
    lyrics_path: Path,
    total_duration: float,
    offset_seconds: float = 0.0,
    vocal_regions: Optional[List[Tuple[float, float]]] = None,
    ffmpeg_path: str = "ffmpeg",
    model_size: str = "small",
    device: str = "auto",
    compute_type: str = "int8",
) -> Optional[Tuple[List[dict], dict]]:
    """
    Main entry point: forced alignment using faster-whisper + Needleman-Wunsch DP.

    Returns:
        (word_entries, karaoke_meta_dict) or None if alignment fails.
    """
    audio_path = Path(audio_path)
    lyrics_path = Path(lyrics_path)

    # 1. Read lyrics
    raw_lyrics = lyrics_path.read_text(encoding="utf-8")
    lines = [ln.rstrip("\r") for ln in raw_lyrics.split("\n")]
    non_empty_lines = [(i, ln.strip()) for i, ln in enumerate(lines) if ln.strip()]

    if not non_empty_lines:
        print("Lyrics file is empty.")
        return None

    # 2. Extract word tokens from lyrics
    all_lyric_words = []
    lyric_lines_grouped: Dict[int, List[dict]] = {}

    for line_idx, line_text in non_empty_lines:
        words = line_text.split()
        lyric_lines_grouped[line_idx] = []
        for w_idx, w in enumerate(words):
            item = {
                "line_idx": line_idx,
                "w_idx": w_idx,
                "text": w,
                "start": None,
                "end": None,
                "matched": False,
            }
            all_lyric_words.append(item)
            lyric_lines_grouped[line_idx].append(item)

    # 3. Transcribe audio with faster-whisper
    whisper_words = transcribe_words(
        audio_path=audio_path,
        language="es",
        model_size=model_size,
        device=device,
        compute_type=compute_type,
    )

    if not whisper_words:
        print("Whisper transcription returned no words.")
        return None

    # 4. DP Global Alignment
    print("Aligning lyrics with Needleman-Wunsch dynamic programming...")
    align_lyric_words_dp(all_lyric_words, whisper_words)

    # 5. Interpolate and sanitize all timestamps
    final_word_entries = interpolate_and_sanitize(lyric_lines_grouped, total_duration)

    matched_count = sum(1 for w in all_lyric_words if w.get("matched"))
    conf_mean = round(float(np.mean([w.get("confidence", 0.0) for w in all_lyric_words])), 3) if all_lyric_words else 0.0

    meta = {
        "schema_version": "shotcut_tiktok.karaoke_meta.v2",
        "timing_method": "forced_alignment_dp",
        "word_count": len(final_word_entries),
        "matched_words": matched_count,
        "alignment_confidence_mean": conf_mean,
        "lyrics_checksum": "",
        "generated_at": "",
        "vocal_regions": [],
        "total_vocal_seconds": 0.0,
    }

    print(f"Alignment successful: {matched_count}/{len(all_lyric_words)} words matched directly ({conf_mean*100:.1f}% confidence).")
    return final_word_entries, meta
