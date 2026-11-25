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
import re
import unicodedata
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
_MODEL_DEVICE: str = "cpu"


def _resolve_mms_device(preferred: Optional[str] = None) -> str:
    try:
        from ..core.hardware import get_torch_device

        return get_torch_device(preferred)
    except Exception:
        if preferred in ("cuda", "mps", "xpu", "cpu"):
            return preferred
        return "cpu"


# MMS_FA emits ~50 frames/sec (conv stride product 320 @ 16 kHz).
_FPS: float = 50.0
_STAR_LOG_PRIOR: float = -1.85


def get_mms_fa(device: Optional[str] = None) -> Tuple[torch.nn.Module, Dict[str, int], int, int]:
    """Load (once) and return (model, dictionary, star_id, blank_id)."""
    global _BUNDLE, _MODEL, _DICT, _STAR_ID, _BLANK_ID, _MODEL_DEVICE
    target_dev = _resolve_mms_device(device)
    if _MODEL is None:
        _BUNDLE = torchaudio.pipelines.MMS_FA
        _MODEL = _BUNDLE.get_model(with_star=True).eval()
        _DICT = dict(_BUNDLE.get_dict(star="*"))
        _STAR_ID = int(_DICT.get("*", 28))
        _BLANK_ID = int(_DICT.get("-", 0))
    if target_dev != _MODEL_DEVICE:
        try:
            _MODEL = _MODEL.to(target_dev)
            _MODEL_DEVICE = target_dev
        except Exception:
            _MODEL = _MODEL.to("cpu")
            _MODEL_DEVICE = "cpu"
    assert _MODEL is not None and _DICT is not None
    return _MODEL, _DICT, _STAR_ID, _BLANK_ID


_SPANISH_NUMBER_WORDS: Dict[str, str] = {
    "0": "cero",
    "1": "uno",
    "2": "dos",
    "3": "tres",
    "4": "cuatro",
    "5": "cinco",
    "6": "seis",
    "7": "siete",
    "8": "ocho",
    "9": "nueve",
    "10": "diez",
    "20": "veinte",
    "30": "treinta",
    "38": "treinta y ocho",
    "40": "cuarenta",
    "48": "cuarenta y ocho",
    "50": "cincuenta",
    "9c": "nueve ce",
    "9C": "nueve ce",
}


def compute_vocal_activity_mask(
    acapella_16k: np.ndarray, sr: int = 16000, hop_sec: float = 0.02
) -> tuple[np.ndarray, dict]:
    """Vocal-speech activity gate on acapella stem.
    Returns (active_mask[T], info). active_mask True = vocal active (do NOT penalize).
    Bleed floor: RMS < max(0.015, 0.25 * p90_rms) considered silent.
    Hysteresis/morphology: bridge <0.35s consonant dips, reject <0.20s blips.
    """
    sig = np.asarray(acapella_16k, dtype=np.float32).reshape(-1)
    hop = max(1, int(round(sr * hop_sec)))
    n = int(np.ceil(len(sig) / hop))
    if n <= 0:
        return np.zeros(0, dtype=bool), {
            "masked_fraction": 0.0,
            "p90": 0.0,
            "thresh": 0.0,
            "percentile_used": 0.0,
        }
    idxs = np.arange(0, len(sig), hop)
    ends = np.append(idxs[1:], len(sig))
    lens = np.maximum(1, ends - idxs)
    sq = sig.astype(np.float64) ** 2
    rms = np.sqrt(np.add.reduceat(sq, idxs) / lens)
    p90 = float(np.percentile(rms[rms > 0], 90)) if np.any(rms > 0) else 0.0
    thresh = max(0.015, 0.25 * p90)
    # hysteresis: enter at thresh, stay at 0.6*thresh
    active = np.zeros(n, dtype=bool)
    on = False
    low = 0.6 * thresh
    for i in range(n):
        if on:
            on = rms[i] >= low
        else:
            on = rms[i] >= thresh
        active[i] = on
    # morphological closing: bridge consonant dips < 0.35s
    gap_max = int(round(0.35 / hop_sec))
    i = 0
    while i < n:
        if not active[i]:
            j = i
            while j < n and not active[j]:
                j += 1
            if j - i <= gap_max and i > 0 and j < n:
                active[i:j] = True
            i = j
        else:
            i += 1
    # reject isolated active blips < 0.20s
    blip_max = int(round(0.20 / hop_sec))
    i = 0
    while i < n:
        if active[i]:
            j = i
            while j < n and active[j]:
                j += 1
            if j - i <= blip_max:
                left_gap = i == 0 or not np.any(active[max(0, i - gap_max) : i])
                right_gap = j == n or not np.any(active[j : min(n, j + gap_max)])
                if left_gap and right_gap:
                    active[i:j] = False
            i = j
        else:
            i += 1
    masked_fraction = float(np.mean(~active)) if n else 0.0
    return active, {
        "masked_fraction": masked_fraction,
        "p90": p90,
        "thresh": thresh,
        "frames": n,
        "percentile_used": 90,
    }


def clean_phonetic_token(token: str, lang: Optional[str] = None) -> str:
    """Normalize a lyric token to acoustic Spanish/Latin character representation."""
    t = (token or "").lower().strip()
    t = re.sub(r"^[^\w]+|[^\w']+$", "", t, flags=re.UNICODE)
    effective_lang = lang or "es"
    try:
        from .number_words import expand_token_numbers
        t = expand_token_numbers(t, lang=effective_lang)
    except Exception:
        if t in _SPANISH_NUMBER_WORDS:
            t = _SPANISH_NUMBER_WORDS[t]
    # NFKD fold accents/ñ before ASCII stripping (qué->que, días->dias, enseñarme->ensenarme)
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.replace("ñ", "n")
    if lang and str(lang).lower().startswith("es"):
        t = t.replace("ll", "y")
    if t == "y":
        t = "i"
    # Strip unvoiced orthographic apostrophe (e.g. "pa'" -> "pa")
    t = t.replace("'", "")
    t = re.sub(r"[^a-z0-9 ]+", "", t)
    return t or "a"


def romanized_to_ids(
    token: str, dictionary: Dict[str, int], lang: Optional[str] = None
) -> List[int]:
    """Map a romanized token to MMS_FA character IDs (excluding blank, star, and unvoiced apostrophe)."""
    clean = clean_phonetic_token(token, lang=lang)
    ids = [
        int(dictionary[c])
        for c in clean
        if c in dictionary and c not in ("-", "*", "'")
    ]
    if not ids:
        ids = [int(dictionary.get("a", 1))]
    return ids


# Backward-compatible alias
romanzied_to_ids = romanized_to_ids


_CONV_STRIDE = 320
_CONV_OVERHEAD = 80  # receptive_field(400) - stride(320)


def extract_emissions_chunked(
    waveform_1d: torch.Tensor,
    sr: int = 16000,
    chunk_sec: float = 20.0,
    overlap_sec: float = 2.0,
    device: Optional[str] = None,
) -> torch.Tensor:
    """Extract (T, 29) log-posterior emissions with exact stride-320 framing and bounded song-level gain.

    No per-chunk +28dB silence boost. Frame count follows Wav2Vec2 conv stack:
    T = floor((N - 400) / 320) + 1  <=>  N = 320*T + 80.
    """
    global _MODEL_DEVICE
    model, _, _, _ = get_mms_fa(device=device)
    active_dev = _MODEL_DEVICE
    if waveform_1d.dim() > 1:
        waveform_1d = waveform_1d.reshape(-1)
    wav = waveform_1d.detach().float().cpu()

    total_samples = int(wav.numel())
    if total_samples == 0:
        return torch.zeros((0, 29), dtype=torch.float32)

    # Bounded song-level gain only (never amplify near-silence gaps into fake vocals).
    rms = float(torch.sqrt(torch.mean(wav * wav) + 1e-12))
    if rms > 1e-4:
        target_rms = 10.0 ** (-20.0 / 20.0)  # -20 dBFS conservative song-level target
        gain = min(4.0, max(0.25, target_rms / rms))  # cap at +12 dB, floor at -12 dB
        wav = torch.clamp(wav * gain, -1.0, 1.0)

    total_frames = max(1, (total_samples - _CONV_OVERHEAD) // _CONV_STRIDE)
    needed_samples = total_frames * _CONV_STRIDE + _CONV_OVERHEAD
    if total_samples < needed_samples:
        wav = torch.nn.functional.pad(wav, (0, needed_samples - total_samples))

    chunk_frames = max(100, int(round(chunk_sec * sr / _CONV_STRIDE)))
    ctx_frames = max(25, int(round(overlap_sec * sr / _CONV_STRIDE)))

    out: List[torch.Tensor] = []
    f_core = 0
    with torch.inference_mode():
        while f_core < total_frames:
            f_core_end = min(total_frames, f_core + chunk_frames)
            f_start = max(0, f_core - ctx_frames)
            f_end = min(total_frames, f_core_end + ctx_frames)

            s0 = f_start * _CONV_STRIDE
            s1 = f_end * _CONV_STRIDE + _CONV_OVERHEAD
            chunk = wav[s0:s1]

            try:
                em, _ = model(chunk.unsqueeze(0).to(active_dev))  # (1, T_c, 29)
                raw_em = em.squeeze(0).cpu()
            except Exception:
                if active_dev != "cpu":
                    model = model.to("cpu")
                    _MODEL_DEVICE = "cpu"
                    active_dev = "cpu"
                    em, _ = model(chunk.unsqueeze(0))
                    raw_em = em.squeeze(0).cpu()
                else:
                    raise

            # Normalize 28 lexical classes (0..27) without <star> denominator dilution
            char_lp = torch.log_softmax(raw_em[:, :28], dim=-1)
            star_col = torch.full(
                (char_lp.shape[0], 1), _STAR_LOG_PRIOR, dtype=char_lp.dtype
            )
            lp = torch.cat([char_lp, star_col], dim=-1)

            left_trim = f_core - f_start
            keep = f_core_end - f_core
            if lp.shape[0] < left_trim + keep:
                # Rare conv-edge mismatch: pad with blank/star-neutral frame
                pad_n = left_trim + keep - lp.shape[0]
                pad = torch.full((pad_n, lp.shape[1]), -30.0, dtype=lp.dtype)
                pad[:, 0] = -0.01
                pad[:, 28] = _STAR_LOG_PRIOR
                lp = torch.cat([lp, pad], dim=0)
            out.append(lp[left_trim : left_trim + keep])
            f_core = f_core_end

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
        return np.full(n_frames, -1, dtype=np.int32), np.zeros(
            n_frames, dtype=np.float32
        )

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
    silence_frame_mask: "np.ndarray | None" = None,
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
    # Vocal-silence character masking: penalize non-blank lexical chars on silent frames
    if silence_frame_mask is not None:
        sm = np.asarray(silence_frame_mask, dtype=bool).reshape(-1)
        Tlp = int(lp_tensor.shape[0])
        if sm.shape[0] >= Tlp:
            sm = sm[:Tlp]
        else:
            sm = np.pad(sm, (0, Tlp - sm.shape[0]), constant_values=False)
        if sm.any():
            lp_np = lp_tensor.numpy().astype(np.float32, copy=True)
            lexical = np.ones(lp_np.shape[1], dtype=bool)
            lexical[blank_id] = False
            if 0 <= star_id < lp_np.shape[1]:
                lexical[star_id] = False
            lp_np[sm][:, lexical] -= 6.0
            lp_np = lp_np - np.logaddexp.reduce(lp_np, axis=1, keepdims=True)
            lp_tensor = torch.from_numpy(lp_np)

    orig_frames = int(lp_tensor.shape[0])
    min_required_frames = max(4, 2 * len(targets) + 2)
    if orig_frames < min_required_frames:
        pad_amount = min_required_frames - orig_frames
        blank_pad = torch.full(
            (pad_amount, lp_tensor.shape[1]), -30.0, dtype=lp_tensor.dtype
        )
        blank_pad[:, blank_id] = -0.01
        blank_pad[:, star_id] = _STAR_LOG_PRIOR
        lp_tensor = torch.cat([lp_tensor, blank_pad], dim=0)

    lp_np = lp_tensor.numpy().astype(np.float32)
    n_frames = int(lp_np.shape[0])
    frame_limit = int(orig_frames)  # never rescale timing for padded frames
    eff_sec_per_frame = float(sec_per_frame)

    fwd_tgt_idx, _ = _viterbi_ctc_state_sequence(lp_np, targets, blank_id=blank_id)

    rev_targets = list(reversed(targets))
    bwd_tgt_rev, _ = _viterbi_ctc_state_sequence(
        lp_np[::-1], rev_targets, blank_id=blank_id
    )
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
            slot = max(1, (frame_limit - prev_f1) // rem_words)
            f0 = min(frame_limit - 1, prev_f1)
            f1 = min(frame_limit, f0 + slot)

        f0 = max(prev_f1, min(f0, frame_limit - 1))
        f1 = max(f0 + 1, min(f1, frame_limit))
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
            c_lo = max(0, min(int(c_lo), frame_limit - 1))
            c_hi = max(c_lo + 1, min(int(c_hi), frame_limit))
            peak_lp = float(np.max(lp_np[c_lo:c_hi, tok_id]))
            non_blank_lp = float(np.max(lp_np[c_lo:c_hi, 1:28]))
            rel_prob = math.exp(max(-10.0, min(0.0, peak_lp)))
            dominance = math.exp(max(-6.0, min(0.0, peak_lp - non_blank_lp)))
            char_peaks.append(min(1.0, 0.70 * (rel_prob**0.30) + 0.30 * dominance))

        raw_post = float(np.mean(char_peaks)) if char_peaks else 0.75
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
