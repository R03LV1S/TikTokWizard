# align/echora_backend.py
"""Optional Echora MMS karaoke-finetuned forced aligner (singing-primary candidate).

Multilingual lyrics/karaoke finetune on MMS wav2vec2 CTC. Same emission shape family
as MMS_FA, different vocab/weights. Lazy-loaded. License: personal/non-commercial per
model card; treat as optional backend.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch

_MODEL = None
_PROCESSOR = None
_VOCAB = None
_ID2CHAR = None
_BLANK_ID = 0
_DEVICE = "cpu"


def _resolve_echora_device(device: Optional[str] = None) -> str:
    try:
        from ..core.hardware import get_torch_device

        return get_torch_device(device)
    except Exception:
        return device if device in ("cuda", "mps", "xpu", "cpu") else "cpu"


def _load(
    model_id: str = "hcX02/echora-mms-300m-multilingual-lyrics-forced-aligner",
    device: Optional[str] = None,
):
    global _MODEL, _PROCESSOR, _VOCAB, _ID2CHAR, _BLANK_ID, _DEVICE
    target_dev = _resolve_echora_device(device)
    if _MODEL is not None and _DEVICE == target_dev:
        return _MODEL, _PROCESSOR, _VOCAB, _BLANK_ID
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    if _PROCESSOR is None:
        _PROCESSOR = Wav2Vec2Processor.from_pretrained(model_id)
    if _MODEL is None:
        _MODEL = Wav2Vec2ForCTC.from_pretrained(model_id, ignore_mismatched_sizes=False)
        _MODEL.eval()
    try:
        _MODEL.to(target_dev)
        _DEVICE = target_dev
    except Exception:
        _MODEL.to("cpu")
        _DEVICE = "cpu"
    vocab = _PROCESSOR.tokenizer.get_vocab()
    # CTC blank is usually pad or 0 depending; for this model '|'=0, [PAD]=29
    # Wav2Vec2ForCTC uses config.pad_token_id as ctc blank in many setups; check.
    blank = getattr(_MODEL.config, "pad_token_id", None)
    if blank is None:
        blank = vocab.get("[PAD]", 0)
    _BLANK_ID = int(blank)
    _VOCAB = dict(vocab)
    _ID2CHAR = {int(i): ch for ch, i in vocab.items()}
    return _MODEL, _PROCESSOR, _VOCAB, _BLANK_ID


def _normalize_word(w: str) -> str:
    import unicodedata, re

    w = unicodedata.normalize("NFKD", str(w or "")).lower()
    w = "".join(ch for ch in w if not unicodedata.combining(ch))
    w = re.sub(r"[^a-z']+", "", w)
    return w


def extract_emissions(
    audio_16k: np.ndarray, sr: int = 16000, device: Optional[str] = None
) -> torch.Tensor:
    """Chunked emission extraction with exact stride-320 framing (20s chunks, 2s overlap)."""
    global _DEVICE
    model, processor, _, _ = _load(device=device)
    active_dev = _DEVICE
    if audio_16k is None or len(audio_16k) == 0:
        return torch.zeros((1, model.config.vocab_size), dtype=torch.float32)
    sig = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
    sr = int(sr)
    stride = 320
    chunk_sec = 20.0
    overlap_sec = 2.0
    chunk_samples = int(chunk_sec * sr)
    overlap_samples = int(overlap_sec * sr)
    if chunk_samples <= 0 or len(sig) <= chunk_samples:
        inp = processor(sig, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.inference_mode():
            try:
                logits = model(inp.input_values.to(active_dev)).logits[0]
            except Exception:
                model.to("cpu")
                _DEVICE = "cpu"
                active_dev = "cpu"
                logits = model(inp.input_values).logits[0]
        return logits.detach().cpu().float().clone()
    # Match MMS_FA framing: include conv overhead for chunk start context
    overhead = 80
    all_logits = []
    cursor = 0
    while cursor < len(sig):
        core_end = min(len(sig), cursor + chunk_samples)
        win_start = max(0, cursor - overhead)
        win_end = min(len(sig), core_end + overhead)
        chunk = sig[win_start:win_end]
        inp = processor(chunk, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.inference_mode():
            try:
                logits = model(inp.input_values.to(active_dev)).logits[0].detach().cpu().float().clone()
            except Exception:
                if active_dev != "cpu":
                    model.to("cpu")
                    _DEVICE = "cpu"
                    active_dev = "cpu"
                    logits = model(inp.input_values).logits[0].detach().cpu().float().clone()
                else:
                    raise
        # Trim to core frames using exact stride-320
        trim_start = int((cursor - win_start) // stride)
        core_frames = int(np.ceil((core_end - cursor) / stride))
        trim_end = min(logits.shape[0], trim_start + core_frames)
        core_logits = logits[trim_start:trim_end]
        if all_logits and overlap_samples > 0 and cursor > 0:
            overlap_frames = int(np.ceil(overlap_samples / stride))
            ov = min(overlap_frames, core_logits.shape[0], all_logits[-1].shape[0])
            if ov > 0:
                prev = all_logits[-1].clone()
                fade = torch.linspace(0.0, 1.0, ov).unsqueeze(-1)
                prev[-ov:] = prev[-ov:] * (1.0 - fade) + core_logits[:ov] * fade
                all_logits[-1] = prev
                core_logits = core_logits[ov:]
        if core_logits.shape[0] > 0:
            all_logits.append(core_logits)
        if core_end >= len(sig):
            break
        cursor = core_end - overlap_samples
    if not all_logits:
        return torch.zeros((1, model.config.vocab_size), dtype=torch.float32)
    return torch.cat(all_logits, dim=0)


def align_words_ctc(
    emissions: torch.Tensor,
    words: Sequence[str],
    romanized: Sequence[str],
    is_parenthetical: Sequence[bool],
    window_start_sec: float,
    sec_per_frame: float,
) -> List[Dict[str, Any]]:
    """CTC forced alignment over emissions. Returns same shape as mms_fa.align_words_star_ctc."""
    try:
        model, _, vocab, blank_id = _load()
    except Exception:
        return []
    if emissions.dim() == 3:
        log_probs = torch.log_softmax(emissions[0], dim=-1)
    else:
        log_probs = torch.log_softmax(emissions, dim=-1)
    n_frames = int(log_probs.shape[0])
    if n_frames == 0 or not words:
        return []

    sep_id = int(vocab.get("|", blank_id))
    targets: List[int] = []
    word_spans: List[Tuple[int, int]] = []
    for wi, rom in enumerate(romanized):
        txt = _normalize_word(rom) or _normalize_word(words[wi]) or "a"
        s = len(targets)
        for ch in txt:
            tid = int(vocab.get(ch, vocab.get("[UNK]", blank_id)))
            if targets and targets[-1] == tid:
                # CTC collapse: separate identical adjacent targets with blank in path; keep target as-is
                pass
            targets.append(tid)
        e = len(targets)
        if e == s:
            targets.append(int(vocab.get("[UNK]", blank_id))); e = len(targets)
        word_spans.append((s, e))
        if wi < len(words) - 1:
            # word separator helps; only add if different from neighbors to avoid issues
            if not targets or targets[-1] != sep_id:
                targets.append(sep_id)
    if not targets:
        return []

    t_tensor = torch.tensor(targets, dtype=torch.int32)
    try:
        import torchaudio.functional as F
        labels, scores = F.forced_align(log_probs.unsqueeze(0), t_tensor.unsqueeze(0), blank=int(blank_id))
        path = labels[0].cpu().numpy().tolist()
        frame_scores = scores[0].cpu().numpy().astype(np.float64)
    except Exception:
        # fallback to naive
        path = [int(blank_id)] * n_frames
        frame_scores = np.zeros(n_frames, dtype=np.float64)

    # Map frame -> target index via CTC path decoding (NOT vocab-id lookup).
    target_pos = [-1] * len(path)
    ti = -1
    prev_blank = True
    for t, lab in enumerate(path):
        lab_i = int(lab)
        if lab_i == int(blank_id):
            prev_blank = True
            continue
        if ti == -1 or prev_blank or (ti < len(targets) and lab_i != int(targets[ti])):
            ti += 1
        prev_blank = False
        if 0 <= ti < len(targets):
            target_pos[t] = ti

    target_to_word = {}
    for wi, (ws, we) in enumerate(word_spans):
        for tti in range(ws, we):
            target_to_word[tti] = wi

    word_frames: Dict[int, List[int]] = {i: [] for i in range(len(words))}
    word_scores: Dict[int, List[float]] = {i: [] for i in range(len(words))}
    for t, tti in enumerate(target_pos):
        if tti < 0:
            continue
        wi = target_to_word.get(int(tti), -1)
        if wi >= 0:
            word_frames[wi].append(t)
            word_scores[wi].append(float(frame_scores[t]) if t < len(frame_scores) else -5.0)

    out = []
    for wi, wtxt in enumerate(words):
        fr = word_frames.get(wi, [])
        scs = word_scores.get(wi, [])
        if fr:
            st_f, en_f = min(fr), max(fr) + 1
            st = window_start_sec + st_f * sec_per_frame
            en = window_start_sec + en_f * sec_per_frame
            # posterior from mean log-prob score (scores are log probs)
            post = float(np.exp(np.clip(np.mean(scs), -20, 0))) if scs else 0.4
        else:
            # proportional fallback
            span = n_frames * sec_per_frame
            st = window_start_sec + (wi / max(1, len(words))) * span
            en = st + max(0.08, span / max(1, len(words)))
            post = 0.25
        conf = min(0.99, max(0.0, float(post ** 0.65)))
        out.append({
            "word": wtxt,
            "is_parenthetical": bool(is_parenthetical[wi]) if wi < len(is_parenthetical) else False,
            "start_sec": round(float(st), 6),
            "end_sec": round(float(max(st + 0.03, en)), 6),
            "duration_cs": max(3, int(round((max(st + 0.03, en) - st) * 100))),
            "posterior": round(float(post), 6),
            "fb_jitter_sec": 0.01,
            "rms": 0.0,
            "confidence": round(float(conf), 6),
            "is_cemented": False,
            "confidence_source": "echora_karaoke_ctc",
        })
    return out
