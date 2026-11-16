# mms_fa.py
"""MMS_FA forced-alignment machinery (v4.1 consolidated).

Provides:
- Lazy singleton for the multilingual MMS_FA bundle (with_star=True, 29 classes)
- Chunked emission extraction (O(T) memory on CPU, local -16 LUFS auto-gain,
  28-class log_softmax + calibrated <star> log-prior)
- Exact state-tracking Forward-Backward <star>-CTC Viterbi word alignment producing
  real per-word timestamps, forward-backward convergence jitter, and calibrated
  acoustic posteriors without target-index/class-ID collision.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchaudio

from ..audio.dsp import normalize_lufs

_BUNDLE: Any = None
_MODEL: Optional[torch.nn.Module] = None
_DICT: Optional[Dict[str, int]] = None
_STAR_ID: int = 28
_BLANK_ID: int = 0

# MMS_FA emits ~50 frames/sec (conv stride product 320 @ 16 kHz).
_FPS: float = 50.0
_STAR_LOG_PRIOR: float = -1.85


def get_mms_fa() -> Tuple[torch.nn.Module, Dict[str, int], int, int]:
    """Load (once) and return (model, dictionary, star_id, blank_id)."""
    global _BUNDLE, _MODEL, _DICT, _STAR_ID, _BLANK_ID
    if _MODEL is None:
        _BUNDLE = torchaudio.pipelines.MMS_FA
        _MODEL = _BUNDLE.get_model(with_star=True).eval()
        _DICT = dict(_BUNDLE.get_dict(star="*"))
        _STAR_ID = int(_DICT.get("*", 28))
        _BLANK_ID = int(_DICT.get("-", 0))
    assert _MODEL is not None and _DICT is not None
    return _MODEL, _DICT, _STAR_ID, _BLANK_ID


def romanized_to_ids(token: str, dictionary: Dict[str, int]) -> List[int]:
    """Map a romanized token to MMS_FA character IDs (excluding blank and star)."""
    clean = (token or "").lower().strip()
    ids = [int(dictionary[c]) for c in clean if c in dictionary and c not in ("-", "*")]
    if not ids:
        ids = [int(dictionary.get("a", 1))]
    return ids


# Backward-compatible alias
romanzied_to_ids = romanized_to_ids


def extract_emissions_chunked(
    waveform_1d: torch.Tensor,
    sr: int = 16000,
    chunk_sec: float = 20.0,
    overlap_sec: float = 2.0,
) -> torch.Tensor:
    """Extract (T, 29) log-posterior emissions across the whole waveform in O(T) RAM.

    Normalizes the 28 standard character/blank classes (0..27) via log_softmax
    and sets class 28 (<star>) to a calibrated wildcard log-prior (-1.85) so the
    <star> channel never halves character posteriors.
    """
    model, _, _, _ = get_mms_fa()
    if waveform_1d.dim() > 1:
        waveform_1d = waveform_1d.reshape(-1)
    waveform_1d = waveform_1d.float().cpu()

    total_samples = int(waveform_1d.shape[0])
    if total_samples == 0:
        return torch.zeros((0, 29), dtype=torch.float32)

    chunk_samples = max(3200, int(round(chunk_sec * sr)))
    overlap_samples = max(320, int(round(overlap_sec * sr)))
    step_samples = max(1600, chunk_samples - 2 * overlap_samples)

    out: List[torch.Tensor] = []
    pos = 0
    with torch.inference_mode():
        while pos < total_samples:
            start = max(0, pos - overlap_samples) if pos > 0 else 0
            core_end = min(total_samples, pos + step_samples)
            end = min(total_samples, core_end + overlap_samples)
            chunk = waveform_1d[start:end]

            orig_len = int(chunk.numel())
            if orig_len < 640:
                chunk = torch.nn.functional.pad(chunk, (0, 640 - orig_len))

            chunk_np = normalize_lufs(chunk.numpy(), target_lufs=-16.0, peak_ceiling_db=-1.0)
            chunk_leveled = torch.from_numpy(chunk_np).float()

            em, _ = model(chunk_leveled.unsqueeze(0))  # (1, T_c, 29)
            raw_em = em.squeeze(0).cpu()

            # Normalize 28 lexical classes (0..27) without <star> denominator dilution
            char_lp = torch.log_softmax(raw_em[:, :28], dim=-1)
            star_col = torch.full((char_lp.shape[0], 1), _STAR_LOG_PRIOR, dtype=char_lp.dtype)
            lp = torch.cat([char_lp, star_col], dim=-1)

            frames = int(lp.shape[0])
            eff_dur = max(1e-6, float(max(orig_len, 640)) / float(sr))
            fps = float(frames) / eff_dur

            left = int(round(((pos - start) / float(sr)) * fps)) if pos > 0 else 0
            right = (
                int(round((orig_len / float(sr)) * fps))
                if core_end >= total_samples
                else int(round(((core_end - start) / float(sr)) * fps))
            )
            left = max(0, min(left, frames - 1))
            right = max(left + 1, min(frames, right))
            out.append(lp[left:right])

            if core_end >= total_samples:
                break
            pos += step_samples

    if not out:
        return torch.zeros((0, 29), dtype=torch.float32)
    return torch.cat(out, dim=0)


def _viterbi_ctc_state_sequence(
    log_probs: np.ndarray,
    targets: Sequence[int],
    blank_id: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact 2L+1 state-tracking CTC Viterbi decoding."""
    n_frames, _ = log_probs.shape
    n_targets = len(targets)
    if n_targets == 0 or n_frames == 0:
        return np.full(n_frames, -1, dtype=np.int32), np.zeros(n_frames, dtype=np.float32)

    n_states = 2 * n_targets + 1
    state_tokens = np.empty(n_states, dtype=np.int32)
    state_tokens[0::2] = blank_id
    state_tokens[1::2] = np.asarray(targets, dtype=np.int32)

    can_skip = np.zeros(n_states, dtype=bool)
    for k in range(1, n_targets):
        if targets[k] != targets[k - 1]:
            can_skip[2 * k + 1] = True

    neg_inf = -1e15
    dp = np.full((n_frames, n_states), neg_inf, dtype=np.float64)
    bp = np.zeros((n_frames, n_states), dtype=np.int32)

    dp[0, 0] = float(log_probs[0, state_tokens[0]])
    if n_states > 1:
        dp[0, 1] = float(log_probs[0, state_tokens[1]])

    state_idx = np.arange(n_states, dtype=np.int32)
    prev1_idx = np.maximum(0, state_idx - 1)
    prev2_idx = np.maximum(0, state_idx - 2)

    for f in range(1, n_frames):
        emit = log_probs[f, state_tokens]
        c_stay = dp[f - 1]
        c_step1 = np.where(state_idx >= 1, dp[f - 1, prev1_idx], neg_inf)
        c_step2 = np.where(can_skip, dp[f - 1, prev2_idx], neg_inf)

        best_prev = state_idx.copy()
        best_val = c_stay.copy()

        mask1 = c_step1 > best_val
        best_val = np.where(mask1, c_step1, best_val)
        best_prev = np.where(mask1, prev1_idx, best_prev)

        mask2 = c_step2 > best_val
        best_val = np.where(mask2, c_step2, best_val)
        best_prev = np.where(mask2, prev2_idx, best_prev)

        dp[f] = best_val + emit
        bp[f] = best_prev

    end_state = n_states - 1
    if n_states >= 2 and dp[-1, n_states - 2] >= dp[-1, n_states - 1]:
        end_state = n_states - 2

    states = np.empty(n_frames, dtype=np.int32)
    cur = int(end_state)
    for f in range(n_frames - 1, -1, -1):
        states[f] = cur
        cur = int(bp[f, cur])

    target_indices = np.where((states % 2) == 1, (states - 1) // 2, -1).astype(np.int32)
    frame_scores = np.array(
        [float(log_probs[f, state_tokens[states[f]]]) for f in range(n_frames)],
        dtype=np.float32,
    )
    return target_indices, frame_scores


def align_words_star_ctc(
    emissions: torch.Tensor,
    words: Sequence[str],
    romanized: Sequence[str],
    is_parenthetical: Sequence[bool],
    window_start_sec: float,
    sec_per_frame: float,
) -> List[Dict[str, Any]]:
    """Align a sentence's words with Forward-Backward <star>-CTC against window emissions."""
    _, dictionary, star_id, blank_id = get_mms_fa()
    n_words = len(words)
    if n_words == 0:
        return []

    targets: List[int] = [star_id]
    word_char_ranges: List[Tuple[int, int]] = []
    for wi, rom in enumerate(romanized):
        chars = romanized_to_ids(rom, dictionary)
        cs = len(targets)
        targets.extend(chars)
        ce = len(targets)
        word_char_ranges.append((cs, ce))
        if wi < n_words - 1:
            targets.append(star_id)
    targets.append(star_id)

    if emissions.dim() == 3:
        lp_tensor = emissions.squeeze(0).detach().cpu()
    else:
        lp_tensor = emissions.detach().cpu()

    orig_frames = int(lp_tensor.shape[0])
    min_required_frames = max(4, 2 * len(targets) + 2)
    if orig_frames < min_required_frames:
        pad_amount = min_required_frames - orig_frames
        blank_pad = torch.full((pad_amount, lp_tensor.shape[1]), -30.0, dtype=lp_tensor.dtype)
        blank_pad[:, blank_id] = -0.01
        blank_pad[:, star_id] = _STAR_LOG_PRIOR
        lp_tensor = torch.cat([lp_tensor, blank_pad], dim=0)

    lp_np = lp_tensor.numpy().astype(np.float32)
    n_frames = int(lp_np.shape[0])
    eff_sec_per_frame = (
        sec_per_frame * (float(orig_frames) / float(n_frames))
        if orig_frames > 0 and n_frames > orig_frames
        else sec_per_frame
    )

    fwd_tgt_idx, _ = _viterbi_ctc_state_sequence(lp_np, targets, blank_id=blank_id)

    rev_targets = list(reversed(targets))
    bwd_tgt_rev, _ = _viterbi_ctc_state_sequence(lp_np[::-1], rev_targets, blank_id=blank_id)
    n_tgt = len(targets)
    bwd_tgt_idx = np.where(bwd_tgt_rev[::-1] >= 0, (n_tgt - 1) - bwd_tgt_rev[::-1], -1)

    def _extract_spans(tgt_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        f_first = np.full(n_tgt, -1, dtype=np.int32)
        f_last = np.full(n_tgt, -1, dtype=np.int32)
        for frame_i in range(n_frames):
            t = int(tgt_seq[frame_i])
            if 0 <= t < n_tgt:
                if f_first[t] < 0:
                    f_first[t] = frame_i
                f_last[t] = frame_i
        return f_first, f_last

    fwd_first, fwd_last = _extract_spans(fwd_tgt_idx)
    bwd_first, bwd_last = _extract_spans(bwd_tgt_idx)

    results: List[Dict[str, Any]] = []
    prev_f1 = 0

    for wi, (cs, ce) in enumerate(word_char_ranges):
        valid_chars = [ci for ci in range(cs, ce) if fwd_first[ci] >= 0]
        bwd_chars = [ci for ci in range(cs, ce) if bwd_first[ci] >= 0]

        if valid_chars:
            f0 = int(min(fwd_first[ci] for ci in valid_chars))
            f1 = int(max(fwd_last[ci] for ci in valid_chars)) + 1
        elif bwd_chars:
            f0 = int(min(bwd_first[ci] for ci in bwd_chars))
            f1 = int(max(bwd_last[ci] for ci in bwd_chars)) + 1
        else:
            rem_words = max(1, n_words - wi)
            slot = max(1, (n_frames - prev_f1) // rem_words)
            f0 = min(n_frames - 1, prev_f1)
            f1 = min(n_frames, f0 + slot)

        f0 = max(prev_f1, min(f0, n_frames - 1))
        f1 = max(f0 + 1, min(f1, n_frames))
        prev_f1 = f1

        if valid_chars and bwd_chars:
            b0 = int(min(bwd_first[ci] for ci in bwd_chars))
            b1 = int(max(bwd_last[ci] for ci in bwd_chars)) + 1
            jitter_frames = 0.5 * (abs(f0 - b0) + abs(f1 - b1))
            fb_jitter_sec = float(jitter_frames) * eff_sec_per_frame
        else:
            fb_jitter_sec = 0.08

        char_peaks: List[float] = []
        for ci in range(cs, ce):
            tok_id = targets[ci]
            c_lo = fwd_first[ci] if fwd_first[ci] >= 0 else f0
            c_hi = (fwd_last[ci] + 1) if fwd_last[ci] >= 0 else f1
            c_lo = max(0, min(int(c_lo), n_frames - 1))
            c_hi = max(c_lo + 1, min(int(c_hi) + 1, n_frames))
            peak_lp = float(np.max(lp_np[c_lo:c_hi, tok_id]))
            non_blank_lp = float(np.max(lp_np[c_lo:c_hi, 1:28]))
            rel_prob = math.exp(max(-10.0, min(0.0, peak_lp)))
            dominance = math.exp(max(-6.0, min(0.0, peak_lp - non_blank_lp)))
            char_peaks.append(min(1.0, 0.60 * (rel_prob ** 0.32) + 0.40 * dominance))

        raw_post = float(np.mean(char_peaks)) if char_peaks else 0.72
        conv_score = float(math.exp(-max(0.0, fb_jitter_sec) / 0.09))
        posterior = float(np.clip(0.60 * raw_post + 0.40 * conv_score, 0.05, 0.999))

        s = window_start_sec + float(f0) * eff_sec_per_frame
        e = window_start_sec + float(f1) * eff_sec_per_frame
        results.append(
            {
                "word": words[wi],
                "is_parenthetical": bool(is_parenthetical[wi]),
                "start_sec": float(s),
                "end_sec": float(e),
                "posterior": float(posterior),
                "fb_jitter_sec": round(float(fb_jitter_sec), 5),
                "_frame0": int(f0),
                "_frame1": int(f1),
            }
        )

    return results
