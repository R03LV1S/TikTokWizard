# align/tunestudio_backend.py
"""Optional Tune Studio singing forced aligner (MMS-family, ONNX).

Singing-finetuned MMS-300m (tuneperfect/tunestudio-aligner). Same char family as
torchaudio MMS_FA; logits remapped to MMS_FA 29-class layout for drop-in use.
License: CC BY-NC 4.0 (non-commercial, attribution). Optional/flagged backend.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import os
import numpy as np
import torch

_SESS = None
_SESS_PATH = None

# Tune Studio vocab (31): 0<blank>,1<pad>,2</s>,3<unk>,4 a,5 i,6 e,7 n,8 o,9 u,10 t,11 s,12 r,13 m,14 k,15 l,16 d,17 g,18 h,19 y,20 b,21 p,22 w,23 c,24 v,25 j,26 z,27 f,28 ',29 q,30 x
# MMS_FA vocab (29):    0 -,1 a,2 i,3 e,4 n,5 o,6 u,7 t,8 s,9 r,10 m,11 k,12 l,13 d,14 g,15 h,16 y,17 b,18 p,19 w,20 c,21 v,22 j,23 z,24 f,25 ',26 q,27 x,28 *
_TS_TO_MMS = {
    0: 0,  # blank
    4: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5,
    9: 6,
    10: 7,
    11: 8,
    12: 9,
    13: 10,
    14: 11,
    15: 12,
    16: 13,
    17: 14,
    18: 15,
    19: 16,
    20: 17,
    21: 18,
    22: 19,
    23: 20,
    24: 21,
    25: 22,
    26: 23,
    27: 24,
    28: 25,
    29: 26,
    30: 27,
}
_STAR_ID = 28
_N_MMS = 29
_STAR_LOG_PRIOR = -1.85


def _get_session(model_id: str = "tuneperfect/tunestudio-aligner"):
    global _SESS, _SESS_PATH
    if _SESS is not None:
        return _SESS
    from huggingface_hub import hf_hub_download
    import onnxruntime as ort

    path = os.environ.get("TW_TUNESTUDIO_ONNX") or hf_hub_download(
        repo_id=model_id, filename="aligner.onnx"
    )
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
    opts.inter_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    # Prefer DirectML if available
    avail = ort.get_available_providers()
    if "DmlExecutionProvider" in avail:
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
    _SESS = ort.InferenceSession(path, sess_options=opts, providers=providers)
    _SESS_PATH = path
    return _SESS


def _remap_logits_to_mms(logits: np.ndarray) -> torch.Tensor:
    """Remap Tune Studio logits [T,31] to MMS_FA log-posterior emissions [T,29]."""
    # logits: [T,31], raw CTC logits
    t, c = logits.shape
    out = np.full((t, _N_MMS), -30.0, dtype=np.float32)
    # log_softmax over the 31 classes first? ctc-forced-aligner appends star after log_softmax.
    x = logits.astype(np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    lse = np.log(np.sum(np.exp(x), axis=-1, keepdims=True) + 1e-12)
    lsm = (x - lse).astype(np.float32)  # [T,31]
    for ts_i, mms_i in _TS_TO_MMS.items():
        if ts_i < c:
            out[:, mms_i] = lsm[:, ts_i]
    # fold pad/eos/unk mass into blank mildly? keep blank as-is from ts blank.
    # set star
    out[:, _STAR_ID] = _STAR_LOG_PRIOR
    # renormalize non-star to keep distribution sane
    nonstar = out[:, :28]
    lse2 = np.log(
        np.sum(np.exp(nonstar.astype(np.float64)), axis=-1, keepdims=True) + 1e-12
    )
    out[:, :28] = (nonstar.astype(np.float64) - lse2).astype(np.float32)
    return torch.from_numpy(out)



def _run_aligner(audio_16k: np.ndarray, sr: int = 16000, want_hidden: bool = False):
    sess = _get_session()
    audio = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
    if len(audio) == 0:
        z = torch.zeros((0, _N_MMS), dtype=torch.float32)
        return (z, None) if want_hidden else z
    win_s, ctx_s = 30.0, 2.0
    win, ctx, step, total = int(win_s * sr), int(ctx_s * sr), int(30.0 * sr), len(audio)
    emissions, hiddens = [], []
    inp_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    logits_name = next((n for n in out_names if "logit" in n.lower()), out_names[0])
    hidden_name = next((n for n in out_names if "hidden" in n.lower()), None)
    req = [logits_name] + ([hidden_name] if want_hidden and hidden_name else [])
    pos = 0
    while pos < total:
        s = max(0, pos - ctx)
        e = min(total, pos + step + ctx)
        chunk = audio[s:e]
        if len(chunk) < 320:
            chunk = np.pad(chunk, (0, 320 - len(chunk)))
        feed = {inp_name: chunk.reshape(1, -1).astype(np.float32)}
        outs = sess.run(req, feed)
        out_map = {n: v for n, v in zip(req, outs)}
        logits = out_map[logits_name]
        if logits.ndim == 3:
            logits = logits[0]
        hidden = out_map.get(hidden_name) if hidden_name else None
        if hidden is not None and hidden.ndim == 3:
            hidden = hidden[0]
        frame_total = logits.shape[0]
        samples_per_frame = len(chunk) / max(1, frame_total)
        left_ctx_frames = int(round((pos - s) / samples_per_frame)) if pos > 0 else 0
        right_end_sample = min(total, pos + step)
        keep_frames = max(1, min(int(round((right_end_sample - pos) / samples_per_frame)), frame_total - left_ctx_frames))
        seg = logits[left_ctx_frames : left_ctx_frames + keep_frames]
        if seg.shape[0] > 0:
            emissions.append(_remap_logits_to_mms(seg))
            if want_hidden and hidden is not None:
                hseg = hidden[left_ctx_frames : left_ctx_frames + keep_frames]
                hiddens.append(torch.from_numpy(np.asarray(hseg, dtype=np.float32)))
        pos += step
        if pos >= total:
            break
    em = torch.cat(emissions, dim=0) if emissions else torch.zeros((0, _N_MMS), dtype=torch.float32)
    if not want_hidden:
        return em
    hid = torch.cat(hiddens, dim=0) if hiddens else None
    return em, hid


def extract_emissions(audio_16k: np.ndarray, sr: int = 16000, device: str = "cpu") -> torch.Tensor:
    """Extract MMS_FA-compatible emissions [T,29] from 16kHz mono audio."""
    return _run_aligner(audio_16k, sr=sr, want_hidden=False)


def boundary_peaks(audio_16k: np.ndarray, sr: int = 16000, threshold: float = 0.3) -> List[float]:
    """Syllable-onset peak times (sec) from TuneStudio boundary.onnx. Secondary evidence only."""
    try:
        from huggingface_hub import hf_hub_download
        import onnxruntime as ort
        em, hidden = _run_aligner(audio_16k, sr=sr, want_hidden=True)
        if hidden is None or hidden.shape[0] == 0:
            return []
        bpath = os.environ.get("TW_TUNESTUDIO_BOUNDARY_ONNX") or hf_hub_download(
            repo_id="tuneperfect/tunestudio-aligner", filename="boundary.onnx"
        )
        sess = ort.InferenceSession(bpath, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name
        outn = sess.get_outputs()[0].name
        probs = sess.run([outn], {inp: hidden.unsqueeze(0).numpy().astype(np.float32)})[0]
        probs = np.asarray(probs).reshape(-1)
        peaks = []
        for i in range(1, len(probs) - 1):
            if probs[i] >= threshold and probs[i] >= probs[i - 1] and probs[i] >= probs[i + 1]:
                peaks.append(round(i * 0.02, 4))
        return peaks
    except Exception:
        return []
