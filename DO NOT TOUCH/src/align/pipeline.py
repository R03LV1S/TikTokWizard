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
import unicodedata
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
from .mms_fa import clean_phonetic_token, compute_vocal_activity_mask, extract_emissions_chunked
from .transient_snapper import snap_words_transients_2ms
from .word_cementer import WordCementer

FPS: float = 50.0
TARGET_SR: int = 16000


def select_vocal_stem(outputs: Sequence[str | Path]) -> Optional[str]:
    """Select the isolated vocal stem from audio-separator outputs, rejecting instrumental trap."""
    for o in outputs or []:
        bn = os.path.basename(str(o)).lower()
        if "(vocals)" in bn and "(instrumental)" not in bn:
            return str(o)
    return None


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
    ENGINE_ID: str = "universal_engine_v1"
    VERSION: str = "1.0"
    NAME: str = "Universal Engine"
    DISPLAY_NAME: str = "Universal Engine v1.0 (Whisper Guided)"

    def __init__(
        self,
        tau_song: float = 0.23,
        word_target: float = 0.95,
        whisper_size: str = "small",
        primary_backend: str = "mms_fa",
        secondary_on_uncemented: bool = True,
        lang: Optional[str] = None,
        stem_mode: Optional[str] = None,
    ):
        self.tau_song = float(tau_song)
        self.word_target = float(word_target)
        self.whisper_size = str(whisper_size)
        self.primary_backend = str(primary_backend or "mms_fa")
        self.secondary_on_uncemented = bool(secondary_on_uncemented)
        self._lang = lang
        self.stem_mode = stem_mode
        self.stem_separation_backend = "fallback_midside"
        self.vocal_gate_info: Dict[str, Any] = {}
        self.cementer = WordCementer(
            tau_song=self.tau_song, word_target=self.word_target, backend=self.primary_backend
        )
        self.resolver = DiscrepancyResolver(
            tau_song=self.tau_song, confidence_threshold=self.word_target
        )
        self.verifier = IntegrityVerifier(word_target=self.word_target)

    # ------------------------------------------------------------------ #
    def _build_stems(self, mp3_path: str, stems_dir: str) -> Dict[str, str]:
        """Build 4 synchronized 16 kHz mono stems with stereo-flatten validation."""
        os.makedirs(stems_dir, exist_ok=True)
        mono, left, right, _ = decode_audio_stereo_and_mono(
            mp3_path, target_sr=TARGET_SR
        )
        true_dur = float(mono.size) / float(TARGET_SR)

        paths = {
            "original_mix_vocal_band": os.path.join(
                stems_dir, "original_mix_vocal_band.wav"
            ),
            "acapella_full": os.path.join(stems_dir, "acapella_full.wav"),
            "acapella_lead_dry": os.path.join(stems_dir, "acapella_lead_dry.wav"),
            "acapella_backing_echo": os.path.join(
                stems_dir, "acapella_backing_echo.wav"
            ),
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

        info_file = os.path.join(stems_dir, "backend_info.json")
        if cache_valid and os.path.isfile(info_file):
            try:
                with open(info_file, "r", encoding="utf-8") as f:
                    bdata = json.load(f)
                src_name = bdata.get("source_audio_name")
                src_size = bdata.get("source_audio_size")
                cur_size = os.path.getsize(mp3_path) if os.path.isfile(mp3_path) else None
                if src_name and src_name != os.path.basename(mp3_path):
                    cache_valid = False
                elif src_size is not None and cur_size is not None and src_size != cur_size:
                    cache_valid = False
                else:
                    self.stem_separation_backend = bdata.get("stem_separation_backend", "fallback_midside")
            except Exception:
                pass
        if cache_valid:
            print(f"[Stems] Reusing cached BS-Roformer stems (0.0s)", flush=True)
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
        vocal_stereo_left: Optional[np.ndarray] = None
        vocal_stereo_right: Optional[np.ndarray] = None
        backend_name = "fallback_midside"
        try:
            import sys
            import time
            import traceback
            from audio_separator.separator import Separator

            do_not_touch_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
            model_file_dir = os.path.join(
                do_not_touch_root, "models", "audio_separator"
            )

            raw_model_choice = getattr(self, "stem_mode", None) or os.environ.get("TW_STEM_MODEL", "model_bs_roformer_ep_317_sdr_12.9755.ckpt").strip()
            compute_mode = os.environ.get("TW_COMPUTE_MODE", "").strip().lower()
            if not compute_mode:
                try:
                    from ..core.hardware import resolve_active_profile

                    compute_mode = resolve_active_profile(verify_encoders=False).compute_mode
                except Exception:
                    compute_mode = "cpu"

            ort_providers: List[str] = []
            try:
                import onnxruntime as _ort

                ort_providers = list(_ort.get_available_providers())
            except Exception:
                pass

            if raw_model_choice in ("fast", "Kim_Vocal_2.onnx", "kim_vocal_2_dml", "kim_vocal_2_cuda", "kim_vocal_2_coreml", "kim_vocal_2_cpu"):
                stem_model = "Kim_Vocal_2.onnx"
                if compute_mode != "cpu" and "CUDAExecutionProvider" in ort_providers:
                    use_directml = False
                    target_ort_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    target_backend = "kim_vocal_2_cuda"
                    stem_display_name = "Kim_Vocal_2 (NVIDIA CUDA ONNX)"
                elif compute_mode != "cpu" and "CoreMLExecutionProvider" in ort_providers:
                    use_directml = False
                    target_ort_providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
                    target_backend = "kim_vocal_2_coreml"
                    stem_display_name = "Kim_Vocal_2 (Apple CoreML)"
                elif compute_mode != "cpu" and "DmlExecutionProvider" in ort_providers:
                    use_directml = True
                    target_ort_providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
                    target_backend = "kim_vocal_2_dml"
                    stem_display_name = "Kim_Vocal_2 (DirectML GPU)"
                else:
                    use_directml = False
                    target_ort_providers = ["CPUExecutionProvider"]
                    target_backend = "kim_vocal_2_cpu"
                    stem_display_name = "Kim_Vocal_2 (CPU ONNX)"
            else:
                stem_model = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
                use_directml = False
                target_ort_providers = ["CPUExecutionProvider"]
                if compute_mode != "cpu" and torch.cuda.is_available():
                    is_rocm = bool(getattr(torch.version, "hip", None))
                    target_backend = "bs_roformer_rocm" if is_rocm else "bs_roformer_cuda"
                    stem_display_name = f"BS-Roformer ({'AMD ROCm' if is_rocm else 'NVIDIA CUDA GPU'}, overlap=2)"
                elif compute_mode != "cpu" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    target_backend = "bs_roformer_mps"
                    stem_display_name = "BS-Roformer (Apple MPS GPU, overlap=2)"
                else:
                    target_backend = "bs_roformer_cpu"
                    stem_display_name = "BS-Roformer CPU, overlap=2"

            if not os.path.isfile(os.path.join(model_file_dir, stem_model)):
                candidate_dirs = [
                    r"C:\Users\User\Desktop\TikTokWizard_Arc_B70\DO NOT TOUCH\models\audio_separator",
                    r"C:\Users\User\Desktop\TikTokWizard\DO NOT TOUCH\models\audio_separator",
                    r"C:\Users\User\Desktop\TikTokWizard_Repo\DO NOT TOUCH\models\audio_separator",
                    r"W:\TikTokWizard\DO NOT TOUCH\models\audio_separator",
                    os.path.expanduser("~/.cache/audio-separator-models"),
                ]
                for cdir in candidate_dirs:
                    if os.path.isfile(os.path.join(cdir, stem_model)):
                        print(f"[Stems] Reusing existing local model weights from {cdir}...", flush=True)
                        os.makedirs(model_file_dir, exist_ok=True)
                        import shutil
                        for mf in os.listdir(cdir):
                            src_mf = os.path.join(cdir, mf)
                            dst_mf = os.path.join(model_file_dir, mf)
                            if os.path.isfile(src_mf) and not os.path.isfile(dst_mf):
                                try:
                                    os.link(src_mf, dst_mf)
                                except Exception:
                                    shutil.copy2(src_mf, dst_mf)
                        break

            dur_min = true_dur / 60.0
            thread_count = os.cpu_count() or 1
            print("=" * 64, flush=True)
            print("[Stems] Starting neural stem separation:", flush=True)
            print(f"  • Track Duration: {true_dur:.1f}s ({dur_min:.2f} min)", flush=True)
            print(f"  • Model: {stem_display_name} (threads: {thread_count})", flush=True)
            if target_backend == "bs_roformer_cpu":
                est_min_lo = dur_min * 2.0
                est_min_hi = dur_min * 3.5
                print(f"  • Estimated Time: ~{est_min_lo:.1f}–{est_min_hi:.1f} min on CPU (~2–3.5 min per audio minute)", flush=True)
                print("  • Note: Stems are cached on disk for instant reruns.", flush=True)
                print("  • Tip: Use '--fast-stems' for ~25s GPU separation on Intel/AMD DirectML or '--compute gpu' on CUDA/Mac.", flush=True)
            elif target_backend in ("bs_roformer_cuda", "bs_roformer_rocm", "bs_roformer_mps"):
                print("  • Estimated Time: ~15–35s via PyTorch GPU acceleration (cached on disk afterwards).", flush=True)
            elif target_backend == "kim_vocal_2_cpu":
                print("  • Estimated Time: ~1–2.5 min on CPU (cached on disk afterwards).", flush=True)
            else:
                print(f"  • Estimated Time: ~20–35s via {stem_display_name} acceleration.", flush=True)
            print("=" * 64, flush=True)
            t_sep_start = time.perf_counter()

            sep_kwargs = {
                "model_file_dir": model_file_dir,
                "output_dir": stems_dir,
                "output_format": "wav",
                "output_single_stem": "Vocals",
                "use_directml": use_directml,
            }
            if stem_model == "model_bs_roformer_ep_317_sdr_12.9755.ckpt":
                bsr_overlap = int(os.environ.get("TW_BSR_OVERLAP", "2").strip())
                sep_kwargs["mdxc_params"] = {"overlap": bsr_overlap, "batch_size": 2}
            sep = Separator(**sep_kwargs)
            # Enforce chosen device providers directly in code (no site-packages edits needed)
            if compute_mode == "cpu":
                sep.torch_device = torch.device("cpu")
                sep.onnx_execution_provider = ["CPUExecutionProvider"]
            elif stem_model == "Kim_Vocal_2.onnx" and target_ort_providers:
                sep.onnx_execution_provider = list(target_ort_providers)
            sep.load_model(stem_model)
            outs = sep.separate(mp3_path)
            t_sep_dur = time.perf_counter() - t_sep_start
            print(f"[Stems] Separation completed in {t_sep_dur:.1f}s ({t_sep_dur / 60.0:.1f} min). Stems cached.", flush=True)
            vocal_stem_rel = select_vocal_stem(outs)
            if vocal_stem_rel:
                cand_path = (
                    vocal_stem_rel
                    if os.path.isabs(vocal_stem_rel)
                    else os.path.join(stems_dir, vocal_stem_rel)
                )
                if os.path.isfile(cand_path):
                    v_mono, v_left, v_right, _ = decode_audio_stereo_and_mono(
                        cand_path, target_sr=TARGET_SR
                    )
                    if abs(v_mono.size / float(TARGET_SR) - true_dur) <= 1.0:
                        acapella_16k = v_mono
                        vocal_stereo_left = v_left
                        vocal_stereo_right = v_right
                        backend_name = target_backend
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            print(f"[stems] Neural stem separation failed with exception:\n{tb}", file=sys.stderr)
            acapella_16k = None
            vocal_stereo_left = None
            vocal_stereo_right = None
            backend_name = "fallback_midside"

        self.stem_separation_backend = backend_name

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
        if vocal_stereo_left is not None and vocal_stereo_right is not None:
            lead_raw, echo_raw = stereo_azimuth_decompose(
                vocal_stereo_left, vocal_stereo_right, sr=TARGET_SR
            )
            lead_16k = _zero_phase_bandpass(lead_raw, 150.0, 7600.0, TARGET_SR)
            echo_16k = _zero_phase_bandpass(echo_raw, 180.0, 7600.0, TARGET_SR)
        else:
            lead_raw, echo_raw = stereo_azimuth_decompose(left, right, sr=TARGET_SR)
            lead_16k = _zero_phase_bandpass(
                0.65 * acapella_16k + 0.35 * lead_raw, 150.0, 7600.0, TARGET_SR
            )
            echo_16k = _zero_phase_bandpass(echo_raw, 180.0, 7600.0, TARGET_SR)

        lead_16k = (lead_16k / (float(np.max(np.abs(lead_16k))) + 1e-9) * 0.89).astype(
            np.float32
        )
        echo_16k = (echo_16k / (float(np.max(np.abs(echo_16k))) + 1e-9) * 0.89).astype(
            np.float32
        )

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

        try:
            with open(info_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "stem_separation_backend": self.stem_separation_backend,
                        "source_audio_name": os.path.basename(mp3_path),
                        "source_audio_size": os.path.getsize(mp3_path) if os.path.isfile(mp3_path) else None,
                    },
                    f,
                    indent=2,
                )
        except Exception:
            pass

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
            flat_tokens = [normalize_token(w) or romanize_word(w) for w in flat_words]
            repeating_tokens = set()
            for idx, t in enumerate(flat_tokens):
                if not t:
                    continue
                w_lo = max(0, idx - 12)
                w_hi = min(len(flat_tokens), idx + 13)
                if any(flat_tokens[j] == t for j in range(w_lo, w_hi) if j != idx):
                    repeating_tokens.add(t)

            filtered: List[Tuple[float, float, str]] = []
            last_seen_en: Dict[str, float] = {}
            for st, en, txt in asr_words:
                tok = normalize_token(txt) or romanize_word(txt)
                prev_end = last_seen_en.get(tok, -100.0)
                time_since_last = float(st) - prev_end

                i0 = max(0, min(int(round(float(st) * sr)), lead_sig.size - 1))
                i1 = max(i0 + 1, min(int(round(float(en) * sr)), lead_sig.size))
                r_lead = (
                    float(np.sqrt(np.mean(lead_sig[i0:i1].astype(np.float64) ** 2)))
                    if i1 > i0
                    else 0.0
                )
                r_echo = (
                    float(np.sqrt(np.mean(echo_sig[i0:i1].astype(np.float64) ** 2)))
                    if i1 > i0
                    else 0.0
                )
                ratio = r_echo / max(1e-4, r_lead)

                # Acoustic echo clone: same token appearing within reverb decay time (<= 6.0s)
                # where backing echo energy significantly dominates lead dry vocal
                # Do NOT delete tokens that legitimately repeat within +/-12 words in flat_words
                if 0.0 <= time_since_last <= 6.0 and ratio > 1.40:
                    if tok not in repeating_tokens:
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
        pass_idx = 0
        stagnant_passes = 0
        prev_uncemented = None
        original_mix_recovery_done = False
        max_passes = 20  # safety cap; primary stop conditions are zero or stagnation
        while pass_idx < max_passes:
            pass_idx += 1
            gaps = self.resolver.scan_uncemented_gaps(all_words, total_duration=dur)
            uncemented_count = sum(
                1
                for w in all_words
                if not w.get("is_cemented", False)
                or float(w.get("confidence", 0.0)) < self.word_target
            )
            if not gaps or uncemented_count == 0:
                break
            if prev_uncemented is not None and uncemented_count >= prev_uncemented:
                stagnant_passes += 1
            else:
                stagnant_passes = 0
            if stagnant_passes >= 2 and not original_mix_recovery_done:
                # Final gate before stagnation: original-mix vocal-band recovery sweep.
                # Acapella separation can silence words; original mix is the last evidence gate.
                original_mix_recovery_done = True
                log(f"[6.{pass_idx}] stagnation: running original-mix recovery sweep over {uncemented_count} uncemented words")
                recovered = 0
                for gap in gaps:
                    idx_a = int(gap["start_idx"]); idx_b = int(gap["end_idx"])
                    line_a = int(all_words[idx_a]["line_index"]); line_b = int(all_words[idx_b]["line_index"])
                    t_left = float(gap["gap_start_sec"]); t_right = float(gap["gap_end_sec"])
                    pad = 0.12
                    prev_bound = float(all_words[idx_a - 1]["end_sec"]) if idx_a > 0 else max(0.0, float(windows[line_a]["start_sec"]) - 0.50)
                    next_bound = float(all_words[idx_b + 1]["start_sec"]) if idx_b < n_words - 1 else min(dur, float(windows[line_b]["end_sec"]) + 0.50)
                    search_lo = max(prev_bound, t_left - pad); search_hi = min(next_bound, t_right + pad)
                    gap_words = [all_words[k]["word"] for k in range(idx_a, idx_b + 1)]
                    gap_roman = [flat_roman[k] for k in range(idx_a, idx_b + 1)]
                    gap_paren = [bool(all_words[k].get("is_parenthetical", False)) for k in range(idx_a, idx_b + 1)]
                    cand_vb = self.cementer.cement_gap_fresh_audio(gap_words, gap_roman, gap_paren, search_lo, search_hi, mix_vb_sig, sr=sr)
                    if not cand_vb or len(cand_vb) != len(gap_words):
                        gap_query = dict(gap); gap_query["gap_start_sec"]=search_lo; gap_query["gap_end_sec"]=search_hi
                        cand_vb = self.resolver.resolve_gap_with_original(gap_query, acapella_sig, mix_vb_sig, sr=sr)
                    if not cand_vb or len(cand_vb) != len(gap_words):
                        continue
                    for off, cand in enumerate(cand_vb):
                        k = idx_a + off
                        orig_w = all_words[k]
                        if orig_w.get("contradiction_locked"):
                            continue
                        if orig_w.get("is_cemented", False) and float(orig_w.get("confidence",0))>=self.word_target:
                            continue
                        c_st = max(0.0, float(cand.get("start_sec", orig_w.get("start_sec",0))))
                        c_en = max(c_st+0.03, float(cand.get("end_sec", c_st+0.1)))
                        new_conf = float(cand.get("confidence", 0.0))
                        cand_post = float(cand.get("posterior", 0.0) or 0.0)
                        if cand_post < 0.55: new_conf = min(new_conf, self.word_target - 0.05)
                        elif cand_post < 0.70: new_conf = min(new_conf, self.word_target - 0.01)
                        # last-gate: accept if mix-vb gives better evidence than current
                        if new_conf > float(orig_w.get("confidence",0)) + 0.01 or (new_conf >= self.word_target and not orig_w.get("is_cemented", False)):
                            orig_w["start_sec"]=round(c_st,6); orig_w["end_sec"]=round(c_en,6)
                            orig_w["duration_cs"]=max(3,int(round((c_en-c_st)*100)))
                            orig_w["confidence"]=round(new_conf,6)
                            orig_w["posterior"]=round(cand_post,6)
                            orig_w["fb_jitter_sec"]=float(cand.get("fb_jitter_sec", orig_w.get("fb_jitter_sec", 0.05)) or 0.05)
                            orig_w["is_cemented"]=bool(new_conf>=self.word_target)
                            orig_w["anchor_tier"]="cemented" if orig_w["is_cemented"] else orig_w.get("anchor_tier","unverified")
                            orig_w.setdefault("source_stream","original_mix_vocal_band")
                            orig_w["confidence_source"]="original_mix_recovery"
                            if orig_w["is_cemented"]: recovered += 1
                log(f"[6.{pass_idx}] original-mix recovery recovered={recovered}")
                prev_uncemented = uncemented_count  # re-evaluate stagnation after recovery
                stagnant_passes = 0
                continue
            if stagnant_passes >= 2:
                log(
                    f"[6.{pass_idx}] stagnation detected: uncemented={uncemented_count} "
                    f"(no improvement for {stagnant_passes} passes). "
                    "Invoking final safety net; manual troubleshooting/engine review recommended."
                )
                break
            prev_uncemented = uncemented_count

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
                    use_vb = mean_vb > mean_ac + 0.01
                    best_cand = cand_vb if use_vb else cand_ac
                    active_rms = rms_vb if use_vb else rms_ac
                    active_med = vb_med if use_vb else ac_med

                if (
                    not best_cand or len(best_cand) != len(gap_words)
                ) and pass_idx >= 3:
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

                needs_stratum_c = (
                    not best_cand
                    or len(best_cand) != len(gap_words)
                    or (
                        pass_idx >= 5
                        and any(
                            float(c.get("confidence", 0.0)) < self.word_target
                            for c in best_cand
                        )
                    )
                )
                if needs_stratum_c:
                    # Stratum C: Irreducible Bounded-Island Solver (last-resort guarantee)
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
                    cand_post = float(cand.get("posterior", 0.0) or 0.0)
                    if cand_post < 0.55:
                        new_conf = min(new_conf, self.word_target - 0.05)
                    elif cand_post < 0.70:
                        new_conf = min(new_conf, self.word_target - 0.01)

                    orig_conf = float(orig_w.get("confidence", 0.0))
                    target_line = int(orig_w.get("line_index", line_a))
                    line_start_bound = float(windows[target_line]["start_sec"]) - 0.50
                    line_end_bound = float(windows[target_line]["end_sec"]) + 0.50

                    within_line_bounds = (
                        line_start_bound <= c_st and c_en <= line_end_bound + 1.0
                    )
                    # Physical bounds: already clamped to neighbor anchors above
                    within_physical_bounds = (c_en - c_st) >= 0.03

                    # Monotonic Quality Overwrite Gate
                    can_overwrite = False
                    if orig_w.get("contradiction_locked"):
                        can_overwrite = False
                    elif orig_w.get("is_cemented", False):
                        can_overwrite = within_line_bounds and (
                            new_conf > orig_conf + 0.005
                        )
                    else:
                        quality_ok = (
                            new_conf >= self.word_target
                            or new_conf >= orig_conf + 0.005
                            or (new_conf >= 0.85 and orig_conf < 0.85)
                            or (pass_idx >= 4 and new_conf >= 0.90)
                        )
                        bounds_ok = within_line_bounds
                        can_overwrite = bounds_ok and quality_ok

                    if can_overwrite:
                        orig_w["start_sec"] = round(c_st, 4)
                        orig_w["end_sec"] = round(c_en, 4)
                        orig_w["duration_cs"] = max(3, int(round((c_en - c_st) * 100)))
                        orig_w["confidence"] = new_conf
                        orig_w["posterior"] = round(cand_post, 6)
                        orig_w["fb_jitter_sec"] = float(cand.get("fb_jitter_sec", orig_w.get("fb_jitter_sec", 0.05)) or 0.05)
                        orig_w["is_cemented"] = bool(new_conf >= self.word_target)
                        orig_w["anchor_tier"] = "cemented" if orig_w["is_cemented"] else orig_w.get("anchor_tier", "unverified")
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
                        clean_phonetic_token(w_obj.text)
                        or normalize_token(w_obj.text)
                        or w_obj.romanized_key
                        or "a"
                    )
                    flat_paren.append(bool(w_obj.is_parenthetical))
            else:
                for t in ln.split():
                    flat_words.append(t)
                    flat_roman.append(
                        clean_phonetic_token(t)
                        or normalize_token(t)
                        or romanize_word(t)
                        or "a"
                    )
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
        self._lang = str(asr_info.get('language') or 'en')
        log(
            f"[3] ASR raw_lead={len(asr_lead)} raw_vb={len(asr_vb)} merged={len(merged_asr)} "
            f"echo_filtered={len(asr_words)} lang={self._lang}"
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
            lines,
            line_word_slices,
            mapping,
            asr_words,
            dur,
            vocal_rms=rms_vb,
            rms_hop_sec=0.01,
        )
        log(
            f"[3] anchored {len(windows)} line windows; calibrated tau_song={self.tau_song:.3f}s"
        )

        # --- Step 4: Chunked Emissions & Vocal-Silence Masking ---
        emissions_acapella = extract_emissions_chunked(
            torch.from_numpy(acapella_sig).float(), sr=TARGET_SR
        )
        emissions_mix_vb = extract_emissions_chunked(
            torch.from_numpy(mix_vb_sig).float(), sr=TARGET_SR
        )
        n_frames = min(emissions_acapella.shape[0], emissions_mix_vb.shape[0])
        emissions_acapella = emissions_acapella[:n_frames]
        emissions_mix_vb = emissions_mix_vb[:n_frames]
        fps = float(n_frames) / dur if dur > 0 else FPS

        # Apply vocal-silence masking on acapella_full before CTC alignment:
        # On 20ms frames where RMS < max(0.015, 0.25 * p90_rms), apply -6.0 log-prob penalty to non-blank lexical classes (1:28)
        active_mask, gate_info = compute_vocal_activity_mask(acapella_sig, sr=TARGET_SR, hop_sec=0.02)
        T_gate = min(n_frames, len(active_mask))
        silence_frames = ~active_mask[:T_gate]
        if np.any(silence_frames):
            emissions_acapella[:T_gate, 1:28] -= 6.0 * torch.from_numpy(silence_frames[:, None]).float()
        log(
            f"[4] vocal_gate masked={gate_info['masked_fraction']*100:.1f}% frames={int(silence_frames.sum())}/{T_gate} p90={gate_info['p90']:.5f} thresh={gate_info['thresh']:.5f}"
        )
        self.vocal_gate_info = gate_info

        # Primary emissions use acapella_full only (no 30% original_mix_vocal_band blend)
        emissions_acapella_primary = emissions_acapella
        emissions_mix_vb_primary = emissions_mix_vb
        emissions_primary = emissions_acapella
        if self.primary_backend == "echora":
            try:
                from .echora_backend import extract_emissions as _echora_extract
                ea = _echora_extract(acapella_sig, sr=TARGET_SR, device="cpu")
                ev = _echora_extract(mix_vb_sig, sr=TARGET_SR, device="cpu")
                if ea.dim() == 3: ea = ea[0]
                if ev.dim() == 3: ev = ev[0]
                n_ep = min(ea.shape[0], ev.shape[0])
                ea = torch.log_softmax(ea[:n_ep], dim=-1)
                ev = torch.log_softmax(ev[:n_ep], dim=-1)
                T_ech_gate = min(n_ep, len(silence_frames))
                if np.any(silence_frames[:T_ech_gate]):
                    ea[:T_ech_gate, 1:32] -= 6.0 * torch.from_numpy(silence_frames[:T_ech_gate, None]).float()
                emissions_acapella_primary, emissions_mix_vb_primary, emissions_primary = ea, ev, ea
                n_frames = n_ep
                fps = float(n_frames) / dur if dur > 0 else FPS
                log(f"[4E] echora primary emissions shape={tuple(ea.shape)} fps={fps:.2f}")
            except Exception as e:
                log(f"[4E] echora primary emissions failed, falling back to mms_fa: {e}")
        elif self.primary_backend == "tunestudio":
            try:
                from .tunestudio_backend import extract_emissions as _ts_extract
                ea = _ts_extract(acapella_sig, sr=TARGET_SR, device="cpu")
                ev = _ts_extract(mix_vb_sig, sr=TARGET_SR, device="cpu")
                if ea.dim() == 3: ea = ea[0]
                if ev.dim() == 3: ev = ev[0]
                n_ep = min(ea.shape[0], ev.shape[0])
                ea = ea[:n_ep].float()
                ev = ev[:n_ep].float()
                T_ts_gate = min(n_ep, len(silence_frames))
                if np.any(silence_frames[:T_ts_gate]):
                    ea[:T_ts_gate, 1:28] -= 6.0 * torch.from_numpy(silence_frames[:T_ts_gate, None]).float()
                emissions_acapella_primary, emissions_mix_vb_primary, emissions_primary = ea, ev, ea
                n_frames = n_ep
                fps = float(n_frames) / dur if dur > 0 else FPS
                log(f"[4T] tunestudio primary emissions shape={tuple(ea.shape)} fps={fps:.2f}")
            except Exception as e:
                log(f"[4T] tunestudio primary emissions failed, falling back to mms_fa: {e}")

        # --- Step 5: Independent Per-Line <star>-CTC Word Cementing ---
        all_words: List[Dict[str, Any]] = []

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
                emissions_primary,
                fps,
                acapella_sig,
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
            emissions_acapella=emissions_acapella_primary[:n_frames],
            emissions_mix_vb=emissions_mix_vb_primary[:n_frames],
            fps=fps,
            acapella_sig=acapella_sig,
            mix_vb_sig=mix_vb_sig,
            sr=sr,
            dur=dur,
            log=log,
        )

        # --- Step 6B: 2ms Sub-Frame Consonant/Sibilant Transient Snapper ---
        snapped = snap_words_transients_2ms(all_words, waveform=acapella_sig, sr=sr)
        for idx, sw in enumerate(snapped):
            all_words[idx]["start_sec"] = sw["start_sec"]
            all_words[idx]["end_sec"] = sw["end_sec"]
            all_words[idx]["duration_cs"] = sw["duration_cs"]

        # --- Phase B: secondary evidence on uncemented words only, until stagnation ---
        if self.secondary_on_uncemented:
            prev_unc = sum(1 for w in all_words if not (w.get("is_cemented", False) and float(w.get("confidence",0))>=self.word_target))
            for pb in range(1, 4):
                phase_b_gain = 0
                reprobe_ok = 0
                for i, w in enumerate(all_words):
                    if w.get("is_cemented", False) and float(w.get("confidence",0))>=self.word_target:
                        continue
                    li = int(w.get("line_index", 0))
                    win = windows[li] if li < len(windows) else {"start_sec":0.0,"end_sec":dur}
                    prev_end = float(all_words[i-1]["end_sec"]) if i>0 else 0.0
                    next_start = float(all_words[i+1]["start_sec"]) if i+1 < len(all_words) else dur
                    lo = max(prev_end, float(win.get("start_sec",0.0))-0.4)
                    hi = min(next_start, float(win.get("end_sec",dur))+0.4)
                    if hi - lo < 0.12:
                        continue
                    orig_st = float(w.get("start_sec", lo))
                    orig_en = float(w.get("end_sec", orig_st+0.1))
                    orig_conf = float(w.get("confidence", 0.0) or 0.0)
                    gap_words=[w.get("word","a")]
                    gap_roman=[flat_roman[i] if i < len(flat_roman) else w.get("romanized","a")]
                    gap_paren=[bool(w.get("is_parenthetical", False))]
                    best=None; best_src=None
                    for src_sig, src_name in ((mix_vb_sig,"original_mix_reprobe"),(acapella_sig,"acapella_reprobe")):
                        if src_sig is None or len(src_sig)==0: continue
                        cand = self.cementer.cement_gap_fresh_audio(gap_words,gap_roman,gap_paren,lo,hi,src_sig,sr=sr)
                        if not cand or len(cand)!=1: continue
                        c=cand[0]
                        cpost=float(c.get("posterior",0) or 0)
                        cconf=float(c.get("confidence",0) or 0)
                        if cpost < 0.55: cconf=min(cconf, self.word_target-0.05)
                        elif cpost < 0.70: cconf=min(cconf, self.word_target-0.01)
                        c["confidence"]=cconf; c["confidence_source"]=src_name
                        if best is None or cconf>float(best.get("confidence",0)):
                            best=c; best_src=src_name
                    if best is None: continue
                    c_st=max(lo,float(best.get("start_sec",lo)))
                    c_en=min(hi,max(c_st+0.03,float(best.get("end_sec",c_st+0.1))))
                    new_conf=float(best.get("confidence",0))
                    shift = abs(((c_st+c_en)/2.0) - ((orig_st+orig_en)/2.0))
                    max_shift = 1.5
                    accept = False
                    if new_conf >= self.word_target and not (w.get("is_cemented", False) and orig_conf >= self.word_target):
                        accept = shift <= max_shift or new_conf >= self.word_target + 0.02
                    elif new_conf >= orig_conf + 0.08 and shift <= 0.75:
                        accept = True
                    if not accept:
                        continue
                    w["start_sec"]=round(c_st,6); w["end_sec"]=round(c_en,6)
                    w["duration_cs"]=max(3,int(round((c_en-c_st)*100)))
                    w["confidence"]=round(new_conf,6)
                    w["is_cemented"]=bool(new_conf>=self.word_target)
                    w["posterior"]=float(best.get("posterior", w.get("posterior",0)) or 0)
                    w["source_stream"]= "original_mix_vocal_band" if best_src=="original_mix_reprobe" else "acapella_full"
                    w["confidence_source"]=best_src
                    if w["is_cemented"]:
                        reprobe_ok+=1; phase_b_gain+=1
                log(f"[Phase B.{pb}] secondary re-probe cemented={reprobe_ok}")

                wx_agree_boost = 0
                try:
                    from .whisperx_backend import align_words_window as _wx_align_window
                    wx_lang = getattr(self, '_lang', 'en') or 'en'
                    for li, (ws, we) in enumerate(line_word_slices):
                        idxs = [i for i in range(ws,we) if not (all_words[i].get('is_cemented', False) and float(all_words[i].get('confidence',0))>=self.word_target)]
                        if not idxs:
                            continue
                        win = windows[li] if li < len(windows) else {'start_sec':0.0,'end_sec':dur}
                        lo = max(0.0, float(win.get('start_sec',0.0)) - 0.5)
                        hi = min(dur, float(win.get('end_sec',dur)) + 0.5)
                        line_words = [str(all_words[i].get('word','')) for i in range(ws,we)]
                        wx = _wx_align_window(acapella_sig if acapella_sig is not None and len(acapella_sig)>0 else mix_vb_sig, sr, line_words, lo, hi, lang=wx_lang, device='cpu')
                        if not wx:
                            continue
                        wx_i = 0
                        for i in range(ws,we):
                            w = all_words[i]
                            wt = ''.join(ch for ch in str(w.get('word','')).lower() if ch.isalnum())
                            match = None
                            for k in range(wx_i, min(len(wx), wx_i+6)):
                                xt = ''.join(ch for ch in str(wx[k].get('word','')).lower() if ch.isalnum())
                                if wt and xt and wt == xt:
                                    match = wx[k]; wx_i = k+1; break
                            if not match:
                                if wx_i < len(wx): wx_i += 1
                                continue
                            if w.get('is_cemented', False) and float(w.get('confidence',0))>=self.word_target:
                                continue
                            if w.get('contradiction_locked'):
                                continue
                            st = float(match.get('start_sec', w.get('start_sec',0.0)))
                            en = float(match.get('end_sec', w.get('end_sec',st+0.1)))
                            cur_st = float(w.get('start_sec',0.0)); cur_en = float(w.get('end_sec',cur_st+0.1))
                            if abs(st-cur_st) <= 0.25 and abs(en-cur_en) <= 0.25:
                                post = float(w.get('posterior',0.0) or 0.0)
                                conf = float(w.get('confidence',0.0) or 0.0)
                                wx_sc = float(match.get('score',0.0) or 0.0)
                                if wx_sc >= 0.5 and post >= 0.55:
                                    new_conf = max(conf, conf + 0.03)
                                    if post < 0.55:
                                        new_conf = min(new_conf, self.word_target - 0.05)
                                    elif post < 0.70:
                                        new_conf = min(new_conf, self.word_target - 0.01)
                                    else:
                                        new_conf = min(new_conf, 0.998)
                                    if new_conf > conf + 1e-6:
                                        w['confidence'] = round(new_conf,6)
                                        w['is_cemented'] = bool(new_conf >= self.word_target)
                                        w['confidence_source'] = (w.get('confidence_source') or 'mms_fa_star_ctc') + '+whisperx_agree'
                                        wx_agree_boost += 1
                                        if w['is_cemented']: phase_b_gain += 1
                except Exception as e:
                    log(f"[Phase B.{pb}] whisperx skipped: {e}")
                if wx_agree_boost:
                    log(f"[Phase B.{pb}] whisperx agreement boosts={wx_agree_boost}")
                unc = sum(1 for w in all_words if not (w.get("is_cemented", False) and float(w.get("confidence",0))>=self.word_target))
                log(f"[Phase B.{pb}] uncemented={unc} gain={phase_b_gain}")
                if phase_b_gain == 0 and unc >= prev_unc:
                    log(f"[Phase B] stagnation after pass {pb}")
                    break
                prev_unc = unc

        # --- B4: TuneStudio boundary.onnx secondary onset corroboration (uncemented only) ---
        try:
            from .tunestudio_backend import boundary_peaks as _ts_boundary_peaks
            b_peaks = _ts_boundary_peaks(mix_vb_sig if mix_vb_sig is not None and len(mix_vb_sig) > 0 else acapella_sig, sr=sr, threshold=0.3)
            if b_peaks:
                b_boost = 0
                for i, w in enumerate(all_words):
                    if w.get("is_cemented", False) and float(w.get("confidence", 0)) >= self.word_target:
                        continue
                    if w.get("contradiction_locked"):
                        continue
                    post = float(w.get("posterior", 0) or 0)
                    if post < 0.55:
                        continue
                    mid = 0.5 * (float(w.get("start_sec", 0)) + float(w.get("end_sec", 0)))
                    near = min((abs(p - mid) for p in b_peaks), default=9e9)
                    if near <= 0.30:
                        conf = float(w.get("confidence", 0) or 0)
                        new_conf = conf + 0.02
                        if post < 0.55:
                            new_conf = min(new_conf, self.word_target - 0.05)
                        elif post < 0.70:
                            new_conf = min(new_conf, self.word_target - 0.01)
                        else:
                            new_conf = min(new_conf, 0.998)
                        if new_conf > conf + 1e-6:
                            w["confidence"] = round(new_conf, 6)
                            # B4 corroboration-only (never flips is_cemented=True)
                            w["confidence_source"] = (w.get("confidence_source") or "unknown") + "+boundary_agree"
                            b_boost += 1
                if b_boost:
                    log(f"[B4] boundary corroboration boosts={b_boost}")
        except Exception as e:
            log(f"[B4] boundary skipped: {e}")

        # --- 6X: Contradiction guard - do not keep high-conf cement if independent evidence disagrees ---
        try:
            from .whisperx_backend import align_words_window as _wx_align_window
            wx_lang = getattr(self, '_lang', 'en') or 'en'
            contradictions = 0
            for li, (ws, we) in enumerate(line_word_slices):
                win = windows[li] if li < len(windows) else {'start_sec':0.0,'end_sec':dur}
                lo = max(0.0, float(win.get('start_sec',0.0)) - 1.5)
                hi = min(dur, float(win.get('end_sec',dur)) + 1.5)
                line_words = [str(all_words[i].get('word','')) for i in range(ws,we)]
                wx = _wx_align_window(acapella_sig if acapella_sig is not None and len(acapella_sig)>0 else mix_vb_sig, sr, line_words, lo, hi, lang=wx_lang, device='cpu')
                if not wx:
                    continue
                wx_i = 0
                for i in range(ws, we):
                    w = all_words[i]
                    wt = ''.join(ch for ch in str(w.get('word','')).lower() if ch.isalnum())
                    match = None
                    for k in range(wx_i, min(len(wx), wx_i+6)):
                        xt = ''.join(ch for ch in str(wx[k].get('word','')).lower() if ch.isalnum())
                        if wt and xt and wt == xt:
                            match = wx[k]; wx_i = k+1; break
                    if not match:
                        # advance cursor conservatively to avoid permanent desync
                        if wx_i < len(wx):
                            wx_i += 1
                        continue
                    if not (w.get('is_cemented', False) and float(w.get('confidence',0)) >= self.word_target):
                        continue
                    if w.get('contradiction_locked'):
                        continue
                    wx_st = float(match.get('start_sec', 0)); wx_en = float(match.get('end_sec', 0))
                    cur_st = float(w.get('start_sec',0)); cur_en = float(w.get('end_sec',0))
                    shift = abs(((wx_st+wx_en)/2) - ((cur_st+cur_en)/2))
                    wx_sc = float(match.get('score',0) or 0)
                    # If independent aligner strongly disagrees on timing, do not keep cemented.
                    if shift >= 0.60 and wx_sc >= 0.65:
                        w['is_cemented'] = False
                        w['confidence'] = round(min(float(w.get('confidence',0)), self.word_target - 0.01), 6)
                        w['anchor_tier'] = 'contradiction_uncemented'
                        w['confidence_source'] = (w.get('confidence_source') or 'unknown') + '+contradiction_guard'
                        w['reject_reason'] = 'evidence_contradicted_or_capped'
                        w['contradiction_locked'] = True
                        contradictions += 1
            if contradictions:
                log(f"[6X] contradiction guard uncemented={contradictions}")
        except Exception as e:
            log(f"[6X] contradiction guard skipped: {e}")

        # --- Step 6C-pre: Intra-line coherence check (un-cement poisoned/outlier anchors) ---
        for li, (ws, we) in enumerate(line_word_slices):
            n_lw = we - ws
            if n_lw < 1:
                continue
            lws = all_words[ws:we]
            times = [0.5 * (float(w.get("start_sec",0)) + float(w.get("end_sec",0))) for w in lws]
            if not times:
                continue
            med_t = float(np.median(times))
            expected_span = max(4.0, 1.2 * n_lw)
            # precompute distance to nearest cemented word outside this line
            outside_times = []
            for k, ow in enumerate(all_words):
                if ws <= k < we:
                    continue
                if ow.get("is_cemented", False):
                    outside_times.append(0.5 * (float(ow.get("start_sec",0)) + float(ow.get("end_sec",0))))
            for j, w in enumerate(lws):
                st = float(w.get("start_sec", 0)); en = float(w.get("end_sec", 0))
                ct = 0.5 * (st + en)
                far_from_median = abs(ct - med_t) > max(8.0, expected_span)
                gap_prev = (st - float(lws[j-1].get("end_sec", st))) if j > 0 else 0.0
                gap_next = (float(lws[j+1].get("start_sec", en)) - en) if j < n_lw-1 else 0.0
                large_gap_prev = gap_prev > max(5.0, expected_span)
                large_gap_next = gap_next > max(5.0, expected_span)
                # Short lines: an absurd internal gap is impossible regardless of outside context
                absurd_short_line_gap = (n_lw <= 3 and (gap_prev > 4.0 or gap_next > 4.0))
                if not ((far_from_median and (large_gap_prev or large_gap_next or n_lw <= 2)) or absurd_short_line_gap):
                    continue
                nearest_outside = min((abs(ct - ot) for ot in outside_times), default=9999.0)
                # Allow outside-context keep only for non-absurd gaps in longer lines
                if nearest_outside < 4.0 and n_lw > 1 and not absurd_short_line_gap:
                    continue
                if w.get("is_cemented", False):
                    log(f"[6C-pre] un-cementing line-incoherent word '{w.get('word')}' at {st:.2f}-{en:.2f}s (line {li+1})")
                    w["is_cemented"] = False
                    w["confidence"] = min(float(w.get("confidence", 0.0)), 0.50)
                    w["anchor_tier"] = "decemented_line_incoherent"

        # For short lines fully decemented by absurd-gap check, cluster at line window center
        for li, (ws, we) in enumerate(line_word_slices):
            lws = all_words[ws:we]
            if not lws or (we-ws) > 4:
                continue
            if not all((not w.get("is_cemented", False) or w.get("anchor_tier")=="decemented_line_incoherent") for w in lws):
                continue
            # only if there was an absurd gap
            had_absurd = any(
                (j>0 and float(lws[j]["start_sec"])-float(lws[j-1]["end_sec"]) > 4.0) or
                (j < len(lws)-1 and float(lws[j+1]["start_sec"])-float(lws[j]["end_sec"]) > 4.0)
                for j in range(len(lws))
            )
            if not had_absurd:
                continue
            win = windows[li] if li < len(windows) else {}
            wc = 0.5 * (float(win.get("start_sec", 0.0)) + float(win.get("end_sec", 0.0)))
            total_d = max(0.3, 0.28 * len(lws))
            t0 = max(0.0, wc - total_d / 2)
            cur = t0
            for w in lws:
                durw = max(0.18, min(0.8, self.cementer.compute_merit_bounds(w.get("word","a"), is_parenthetical=bool(w.get("is_parenthetical",False)))[1]))
                w["start_sec"] = round(cur, 4)
                w["end_sec"] = round(cur + durw, 4)
                w["is_cemented"] = False
                w["confidence"] = min(float(w.get("confidence",0.0)), 0.45)
                w["anchor_tier"] = "decemented_line_incoherent"
                cur += durw

        # --- Step 6C: Final timing placement (renderable, honesty-preserving; no fake cementation) ---
        n_aw = len(all_words)
        # 6C.1: any remaining uncemented words get bounded placement for rendering, but stay uncemented
        for i, w in enumerate(all_words):
            if w.get("contradiction_locked"):
                continue
            if w.get("is_cemented", False) and float(w.get("confidence", 0.0)) >= self.word_target:
                continue
            # find cemented neighbors
            prev_end = 0.0
            for j in range(i - 1, -1, -1):
                if all_words[j].get("is_cemented", False):
                    prev_end = float(all_words[j].get("end_sec", prev_end))
                    break
                prev_end = min(prev_end, float(all_words[j].get("start_sec", prev_end)))
            next_start = dur
            for j in range(i + 1, n_aw):
                if all_words[j].get("is_cemented", False):
                    next_start = float(all_words[j].get("start_sec", next_start))
                    break
                next_start = max(next_start, float(all_words[j].get("end_sec", next_start)))
            if next_start <= prev_end:
                next_start = min(dur, prev_end + 0.20)
            span = max(0.08, next_start - prev_end)
            orig_st = float(w.get("start_sec", prev_end))
            orig_en = float(w.get("end_sec", orig_st + 0.12))
            use_interp = (
                w.get("anchor_tier") == "decemented_line_incoherent"
                or orig_en <= prev_end + 0.01
                or orig_st >= next_start - 0.01
                or (span > 5.0 and (orig_st < prev_end - 0.5 or orig_en > next_start + 0.5))
            )
            if use_interp:
                # Prefer anchoring near coherent same-line words
                li_w = int(w.get("line_index", -1))
                same_line = [k for k in range(n_aw) if int(all_words[k].get("line_index",-2))==li_w and k!=i and all_words[k].get("is_cemented",False) and all_words[k].get("anchor_tier")!="decemented_line_incoherent"]
                placed = False
                if same_line:
                    same_line.sort()
                    # find nearest same-line cemented neighbor
                    before = [k for k in same_line if k < i]
                    after = [k for k in same_line if k > i]
                    anchor_k = after[0] if after else before[-1]
                    anchor_w = all_words[anchor_k]
                    a_st = float(anchor_w.get("start_sec", prev_end))
                    a_en = float(anchor_w.get("end_sec", a_st+0.2))
                    want_d = max(0.18, self.cementer.compute_merit_bounds(w.get("word","a"), is_parenthetical=bool(w.get("is_parenthetical",False)))[1])
                    if anchor_k > i:
                        st = max(prev_end, a_st - want_d - 0.02)
                        en = min(a_st - 0.01, st + want_d)
                    else:
                        st = max(a_en + 0.01, a_en + 0.02)
                        en = min(next_start, st + want_d)
                    if en - st >= 0.08:
                        w["start_sec"] = round(st,4)
                        w["end_sec"] = round(en,4)
                        w["duration_cs"] = max(3,int(round((en-st)*100)))
                        if float(w.get("confidence",0.0)) < self.word_target:
                            w["confidence"] = round(min(float(w.get("confidence",0.0)), self.word_target - 0.01),6)
                        w["is_cemented"] = False
                        w["anchor_tier"] = "placement_unverified"
                        placed = True
                if not placed:
                    # distribute proportionally among contiguous uncemented run in this gap
                    run = [i]
                    j2 = i - 1
                    while j2 >= 0 and not (all_words[j2].get("is_cemented", False) and float(all_words[j2].get("confidence",0))>=self.word_target):
                        run.insert(0,j2); j2 -= 1
                    j2 = i + 1
                    while j2 < n_aw and not (all_words[j2].get("is_cemented", False) and float(all_words[j2].get("confidence",0))>=self.word_target):
                        run.append(j2); j2 += 1
                    rprev = prev_end
                    rnext = next_start
                    total_w = max(1e-6, sum(max(0.08, self.cementer.compute_merit_bounds(all_words[k].get("word","a"))[1]) for k in run))
                    # Don't stretch unverified placements across huge gaps
                    cluster_cap = sum(min(1.5, max(0.18, self.cementer.compute_merit_bounds(all_words[k].get("word","a"))[2]*1.2)) for k in run)
                    effective_span = min(span, max(cluster_cap, 0.5 * len(run)))
                    # center cluster in gap if gap is much larger
                    if span > effective_span + 1.0:
                        rprev_c = rprev + (span - effective_span) * 0.35
                        rnext_c = min(rnext, rprev_c + effective_span)
                    else:
                        rprev_c, rnext_c = rprev, rprev + effective_span
                    cursor = rprev_c
                    for k in run:
                        mw = max(0.08, self.cementer.compute_merit_bounds(all_words[k].get("word","a"))[1])
                        wdur = min(max(0.08, effective_span * (mw / total_w)), 1.8)
                        st = cursor
                        en = min(rnext_c, st + wdur)
                        if k == run[-1]:
                            en = min(rnext_c, max(en, st+0.08))
                        all_words[k]["start_sec"] = round(st,4)
                        all_words[k]["end_sec"] = round(max(st+0.06,en),4)
                        all_words[k]["duration_cs"] = max(3,int(round((float(all_words[k]["end_sec"])-float(all_words[k]["start_sec"]))*100)))
                        if float(all_words[k].get("confidence",0.0)) < self.word_target:
                            all_words[k]["confidence"] = round(min(float(all_words[k].get("confidence",0.0)), self.word_target - 0.01),6)
                        all_words[k]["is_cemented"] = False
                        all_words[k].setdefault("anchor_tier","placement_unverified")
                        cursor = float(all_words[k]["end_sec"])
                continue
            st = orig_st
            en = orig_en
            st = max(prev_end, min(st, next_start - 0.06))
            en = max(st + 0.06, min(next_start, en))
            if en - st < 0.08:
                en = min(next_start, st + 0.12)
            # Cap unverified placements to reasonable duration
            if float(w.get("confidence", 0.0)) < self.word_target:
                max_d = min(1.8, max(0.4, self.cementer.compute_merit_bounds(w.get("word","a"), is_parenthetical=bool(w.get("is_parenthetical",False)))[2] * 1.2))
                if en - st > max_d:
                    en = min(next_start, st + max_d)
            w["start_sec"] = round(st, 4)
            w["end_sec"] = round(en, 4)
            w["duration_cs"] = max(3, int(round((en - st) * 100)))
            if float(w.get("confidence", 0.0)) >= self.word_target:
                w["confidence"] = round(self.word_target - 0.01, 6)
            w["is_cemented"] = False
            w.setdefault("anchor_tier", "placement_unverified")

        # 6C.2: enforce minimum readable durations (monotonic, neighbor-aware)
        def _min_dur_for_word(wd: Dict[str, Any]) -> float:
            txt = str(wd.get("word", "") or "")
            t = unicodedata.normalize("NFKD", txt).lower()
            clean = "".join(ch for ch in t if not unicodedata.combining(ch) and ch.isalnum())
            nchar = max(1, len(clean))
            nsyl = max(1, len([g for g in __import__('re').findall(r'[aeiouy]+', clean)]))
            base = 0.14 if nchar <= 2 else (0.16 if nsyl == 1 else 0.20)
            if bool(wd.get("is_parenthetical", False)):
                base = max(base, 0.28)
            return min(base, 0.60)

        for li, (ws, we) in enumerate(line_word_slices):
            if we <= ws:
                continue
            # forward pass
            prev_end = float(all_words[ws - 1]["end_sec"]) if ws > 0 else 0.0
            line_end_bound = float(all_words[we]["start_sec"]) if we < n_aw else dur
            for i in range(ws, we):
                w = all_words[i]
                if w.get("is_cemented", False) and float(w.get("confidence", 0)) >= self.word_target:
                    prev_end = float(w.get("end_sec", prev_end))
                    continue
                st = max(prev_end, float(w.get("start_sec", prev_end)))
                en = max(st + 0.03, float(w.get("end_sec", st + 0.03)))
                min_d = _min_dur_for_word(w)
                next_start = float(all_words[i + 1]["start_sec"]) if i + 1 < n_aw else dur
                room = max(0.03, next_start - st - (0.02 if i + 1 < we else 0.0))
                en = max(en, st + min(min_d, room))
                en = min(en, next_start if i + 1 < n_aw else dur)
                if en - st < min_d - 1e-6 and i + 1 < we:
                    need = min_d - (en - st)
                    tail_room = max(0.0, line_end_bound - float(all_words[we - 1]["end_sec"]) - 0.02)
                    push = min(need, tail_room, 0.25)
                    if push > 1e-6:
                        for k in range(i + 1, we):
                            all_words[k]["start_sec"] = round(float(all_words[k]["start_sec"]) + push, 4)
                            all_words[k]["end_sec"] = round(min(line_end_bound - 0.01, float(all_words[k]["end_sec"]) + push), 4)
                        next_start = float(all_words[i + 1]["start_sec"]) if i + 1 < n_aw else dur
                        en = min(next_start, st + min_d)
                if en <= st:
                    en = min(dur, st + 0.03)
                w["start_sec"] = round(st, 4)
                w["end_sec"] = round(en, 4)
                prev_end = float(w["end_sec"])
            # backward pass to fix overlaps + enforce minimum readable duration
            next_start = float(all_words[we]["start_sec"]) if we < n_aw else dur
            for i in range(we - 1, ws - 1, -1):
                w = all_words[i]
                if w.get("is_cemented", False) and float(w.get("confidence", 0)) >= self.word_target:
                    next_start = float(w.get("start_sec", next_start))
                    continue
                st = float(w["start_sec"])
                en = min(float(w["end_sec"]), next_start)
                min_d = min(_min_dur_for_word(w), 0.30)
                prev_bound = float(all_words[i-1]["end_sec"]) if i > ws else (float(all_words[ws-1]["end_sec"]) if ws > 0 else 0.0)
                if en - st < min_d:
                    # try extend backward into previous word's slack
                    need = min_d - (en - st)
                    prev_slack = max(0.0, st - prev_bound - 0.06)
                    take = min(need, prev_slack)
                    st = max(prev_bound, st - take)
                if en - st < 0.03:
                    st = max(prev_bound, en - 0.03)
                w["start_sec"] = round(max(0.0, st), 4)
                w["end_sec"] = round(max(w["start_sec"] + 0.03, en), 4)
                next_start = w["start_sec"]

        # 6C.3: intra-line coherence - re-cluster isolated unverified words causing >2s gaps
        for li, (ws, we) in enumerate(line_word_slices):
            if we - ws < 2:
                continue
            lw = list(range(ws, we))
            # build clusters separated by >2s
            clusters=[]; cur=[lw[0]]
            for a,b in zip(lw,lw[1:]):
                gap = float(all_words[b]["start_sec"]) - float(all_words[a]["end_sec"])
                if gap > 2.0:
                    clusters.append(cur); cur=[b]
                else:
                    cur.append(b)
            clusters.append(cur)
            if len(clusters) <= 1:
                continue
            # keep largest cluster as anchor; move small unverified-only clusters toward it
            anchor = max(clusters, key=lambda c: (sum(1 for i in c if all_words[i].get('is_cemented') and float(all_words[i].get('confidence',0))>=self.word_target), len(c)))
            anchor_mid = (float(all_words[anchor[0]]["start_sec"]) + float(all_words[anchor[-1]]["end_sec"])) / 2.0
            for c in clusters:
                if c is anchor:
                    continue
                if any(all_words[i].get("is_cemented") and float(all_words[i].get("confidence",0))>=self.word_target for i in c):
                    continue
                # move this unverified cluster near anchor edge
                c_dur = sum(max(0.08, float(all_words[i]["end_sec"]) - float(all_words[i]["start_sec"])) for i in c)
                if float(all_words[c[-1]]["end_sec"]) < anchor_mid:
                    # place before anchor
                    target_end = max(float(all_words[anchor[0]]["start_sec"]) - 0.05, ws and float(all_words[ws-1]["end_sec"])+0.02 or 0.0)
                    target_start = max((float(all_words[ws-1]["end_sec"])+0.02) if ws>0 else 0.0, target_end - c_dur)
                else:
                    # place after anchor
                    target_start = min(float(all_words[anchor[-1]]["end_sec"]) + 0.05, (float(all_words[we]["start_sec"])-0.02-c_dur) if we<n_aw else dur-c_dur)
                    target_start = max(target_start, float(all_words[anchor[-1]]["end_sec"]) + 0.02)
                cur_t = target_start
                for i in c:
                    w = all_words[i]
                    wd = max(0.08, float(w["end_sec"]) - float(w["start_sec"]))
                    st = cur_t; en = st + wd
                    hi_bound = float(all_words[we]["start_sec"]) - 0.02 if we < n_aw else dur
                    lo_bound = float(all_words[ws-1]["end_sec"]) + 0.02 if ws > 0 else 0.0
                    st = max(lo_bound, min(st, hi_bound - wd))
                    en = min(hi_bound, st + wd)
                    w["start_sec"] = round(st,4); w["end_sec"] = round(max(st+0.03,en),4)
                    w["duration_cs"] = max(3, int(round((float(w["end_sec"])-float(w["start_sec"]))*100)))
                    w["is_cemented"] = False
                    w["confidence"] = round(min(float(w.get("confidence",0.0)), self.word_target - 0.01),6)
                    w.setdefault("anchor_tier","placement_unverified")
                    w["confidence_source"] = w.get("confidence_source") or "line_coherence_recluster"
                    cur_t = float(w["end_sec"]) + 0.01

        # 6C.4: final minimum readable duration enforcement (monotonic, bounded)
        for li, (ws, we) in enumerate(line_word_slices):
            if we <= ws: continue
            prev_end = float(all_words[ws-1]["end_sec"]) if ws > 0 else 0.0
            for i in range(ws, we):
                w = all_words[i]
                if w.get("is_cemented", False) and float(w.get("confidence", 0)) >= self.word_target:
                    prev_end = float(w.get("end_sec", prev_end))
                    continue
                st = max(prev_end, float(w.get("start_sec", prev_end)))
                en = max(st + 0.03, float(w.get("end_sec", st + 0.03)))
                min_d = _min_dur_for_word(w)
                clean_w = "".join(ch for ch in unicodedata.normalize("NFKD", str(w.get("word","")).lower()) if not unicodedata.combining(ch) and ch.isalnum())
                nsy = len(__import__("re").findall(r"[aeiouy]+", clean_w))
                if nsy >= 2: min_d = max(min_d, 0.28)
                if nsy >= 3: min_d = max(min_d, 0.38)
                next_bound = float(all_words[i+1]["start_sec"]) if i+1 < len(all_words) else dur
                room = max(0.03, next_bound - st - (0.02 if i+1 < we else 0.0))
                en = max(en, min(st + min_d, st + room))
                en = min(en, next_bound)
                w["start_sec"] = round(st,4); w["end_sec"] = round(max(st+0.03,en),4)
                w["duration_cs"] = max(3, int(round((w["end_sec"]-w["start_sec"])*100)))
                prev_end = w["end_sec"]
            next_start = float(all_words[we]["start_sec"]) if we < len(all_words) else dur
            for i in range(we-1, ws-1, -1):
                w = all_words[i]
                if w.get("is_cemented", False) and float(w.get("confidence", 0)) >= self.word_target:
                    next_start = float(w.get("start_sec", next_start))
                    continue
                st = float(w["start_sec"]); en = min(float(w["end_sec"]), next_start)
                if en - st < 0.03:
                    prev_bound = float(all_words[i-1]["end_sec"]) if i > ws else (float(all_words[ws-1]["end_sec"]) if ws>0 else 0.0)
                    st = max(prev_bound, en - 0.03)
                w["start_sec"] = round(max(0.0,st),4); w["end_sec"] = round(max(w["start_sec"]+0.03,en),4)
                w["duration_cs"] = max(3, int(round((w["end_sec"]-w["start_sec"])*100)))
                next_start = w["start_sec"]

        # Invariant: posterior evidence floor cannot be bypassed by any upstream path
        for w in all_words:
            post = float(w.get("posterior", 0.0) or 0.0)
            conf = float(w.get("confidence", 0.0) or 0.0)
            orig_conf = conf
            if post < 0.55:
                conf = min(conf, self.word_target - 0.05)
            elif post < 0.70:
                conf = min(conf, self.word_target - 0.01)
            if conf != orig_conf:
                log(f"[floor] capped '{w.get('word')}' post={post:.3f} {orig_conf:.3f}->{conf:.3f} tier={w.get('anchor_tier')}")
                w["confidence_source"]="floor_invariant"
            w["confidence"] = round(conf, 6)
            w["is_cemented"] = bool(conf >= self.word_target)

        # Acoustic snap: word-level for unverified words; line-cluster for absurd-gap lines
        try:
            from .onset import best_peak_time as _onset_best_peak
        except Exception:
            _onset_best_peak = None

        def _rms_peak(sig, sr, t0, t1, hop=0.02, win=0.16):
            if sig is None or len(sig)==0 or t1<=t0: return None, -1.0
            if _onset_best_peak is not None:
                try:
                    bt, bv, _meta = _onset_best_peak(sig, sr, t0, t1, min_dur=max(0.08, float(win)))
                    return bt, bv
                except Exception:
                    pass
            best_t, best_v = None, -1.0
            t=t0
            while t+win <= t1+1e-6:
                i0=max(0,int(t*sr)); i1=min(len(sig),int((t+win)*sr))
                if i1>i0:
                    seg=sig[i0:i1].astype(np.float64)
                    rms=float(np.sqrt(np.mean(seg*seg)))
                    onset=float(np.sqrt(np.mean(np.diff(seg)**2))) if seg.size>1 else 0.0
                    v=0.7*rms + 0.3*onset
                    if v>best_v: best_v,best_t=v,t
                t+=hop
            return best_t, best_v
        ref_sig = mix_vb_sig if mix_vb_sig is not None and len(mix_vb_sig)>0 else acapella_sig
        for li, (ws, we) in enumerate(line_word_slices):
            lws = all_words[ws:we]
            if not lws: continue
            unverified = [w for w in lws if not (w.get("is_cemented", False) and float(w.get("confidence",0))>=self.word_target)]
            has_absurd_gap = any((float(lws[j+1]["start_sec"])-float(lws[j]["end_sec"])>3.0) for j in range(len(lws)-1))
            if not unverified and not has_absurd_gap:
                continue
            win = windows[li] if li < len(windows) else {}
            w0 = max(0.0, float(win.get("start_sec", 0.0)) - 0.75)
            w1 = min(dur, float(win.get("end_sec", w0+1.0)) + 0.75)
            prev_bound = float(all_words[ws-1]["end_sec"]) if ws>0 else 0.0
            next_bound = float(all_words[we]["start_sec"]) if we < len(all_words) else dur
            if has_absurd_gap or len(unverified)==len(lws):
                peak_t, _ = _rms_peak(ref_sig, sr, w0, w1)
                if peak_t is not None:
                    total_d = sum(max(0.12, min(0.8, self.cementer.compute_merit_bounds(w.get("word","a"), is_parenthetical=bool(w.get("is_parenthetical",False)))[1])) for w in lws)
                    total_d = min(total_d, max(0.3, next_bound-prev_bound-0.05))
                    t0 = max(prev_bound, min(peak_t, next_bound - total_d))
                    cur = t0
                    for w in lws:
                        if w.get("is_cemented", False) and float(w.get("confidence", 0)) >= self.word_target:
                            cur = float(w.get("end_sec", cur))
                            continue
                        dw = max(0.12, min(0.8, self.cementer.compute_merit_bounds(w.get("word","a"), is_parenthetical=bool(w.get("is_parenthetical",False)))[1]))
                        dw = min(dw, max(0.08, next_bound-cur-0.02))
                        w["start_sec"]=round(cur,4); w["end_sec"]=round(cur+dw,4)
                        if not (w.get("is_cemented", False) and float(w.get("confidence",0))>=self.word_target):
                            w["is_cemented"]=False
                            w["confidence"]=round(min(float(w.get("confidence",0.0)), self.word_target-0.01),6)
                            w.setdefault("anchor_tier","placement_unverified")
                            w["snap_reason"]="acoustic_rms_peak_line"
                        cur += dw
            else:
                # word-level snap: only move unverified words within local bounds
                for j, w in enumerate(lws):
                    if w.get("is_cemented", False) and float(w.get("confidence",0))>=self.word_target:
                        continue
                    lo = max(prev_bound, float(lws[j-1]["end_sec"]) if j>0 else prev_bound, w0)
                    hi = min(next_bound, float(lws[j+1]["start_sec"]) if j < len(lws)-1 else next_bound, w1)
                    dw = max(0.12, min(0.8, self.cementer.compute_merit_bounds(w.get("word","a"), is_parenthetical=bool(w.get("is_parenthetical",False)))[1]))
                    if hi - lo < dw + 0.02:
                        continue
                    peak_t, pv = _rms_peak(ref_sig, sr, lo, max(lo+dw, hi-dw))
                    if peak_t is None:
                        continue
                    st = max(lo, min(peak_t, hi-dw))
                    w["start_sec"]=round(st,4); w["end_sec"]=round(min(hi, st+dw),4)
                    w["is_cemented"]=False
                    w["confidence"]=round(min(float(w.get("confidence",0.0)), self.word_target-0.01),6)
                    w.setdefault("anchor_tier","placement_unverified")
                    w["snap_reason"]="acoustic_rms_peak_word"

        # recompute windows from final word timings
        for li, (ws, we) in enumerate(line_word_slices):
            if we > ws:
                l_words = all_words[ws:we]
                windows[li]["start_sec"] = round(float(l_words[0]["start_sec"]), 4)
                windows[li]["end_sec"] = round(float(l_words[-1]["end_sec"]), 4)


        # 6C.5: post-snap min-duration guard (prevents acoustic snap from creating unreadable words)
        for li, (ws, we) in enumerate(line_word_slices):
            if we <= ws: continue
            for _pass in range(2):
                prev_end = float(all_words[ws-1]["end_sec"]) if ws > 0 else 0.0
                for i in range(ws, we):
                    w = all_words[i]
                    if w.get("is_cemented", False) and float(w.get("confidence", 0)) >= self.word_target:
                        prev_end = float(w.get("end_sec", prev_end))
                        continue
                    st = max(prev_end, float(w.get("start_sec", prev_end)))
                    en = max(st + 0.03, float(w.get("end_sec", st + 0.03)))
                    min_d = _min_dur_for_word(w)
                    clean_w = "".join(ch for ch in unicodedata.normalize("NFKD", str(w.get("word","")).lower()) if not unicodedata.combining(ch) and ch.isalnum())
                    nsy = len(__import__("re").findall(r"[aeiouy]+", clean_w))
                    if nsy >= 2: min_d = max(min_d, 0.28)
                    next_bound = float(all_words[i+1]["start_sec"]) if i+1 < len(all_words) else dur
                    room = max(0.03, next_bound - st - 0.01)
                    if en - st < min_d - 1e-6 and room >= min_d:
                        en = min(next_bound, st + min_d)
                    w["start_sec"]=round(st,4); w["end_sec"]=round(max(st+0.03,en),4)
                    w["duration_cs"]=max(3,int(round((w["end_sec"]-w["start_sec"])*100)))
                    prev_end = w["end_sec"]
                next_start = float(all_words[we]["start_sec"]) if we < len(all_words) else dur
                for i in range(we-1, ws-1, -1):
                    w = all_words[i]
                    if w.get("is_cemented", False) and float(w.get("confidence", 0)) >= self.word_target:
                        next_start = float(w.get("start_sec", next_start))
                        continue
                    st = float(w["start_sec"]); en = min(float(w["end_sec"]), next_start)
                    min_d = _min_dur_for_word(w)
                    clean_w = "".join(ch for ch in unicodedata.normalize("NFKD", str(w.get("word","")).lower()) if not unicodedata.combining(ch) and ch.isalnum())
                    nsy = len(__import__("re").findall(r"[aeiouy]+", clean_w))
                    if nsy >= 2: min_d = max(min_d, 0.28)
                    prev_bound = float(all_words[i-1]["end_sec"]) if i > ws else (float(all_words[ws-1]["end_sec"]) if ws>0 else 0.0)
                    if en - st < min_d - 1e-6 and (en - prev_bound - 0.01) >= min_d:
                        st = max(prev_bound, en - min_d)
                    if en - st < 0.03:
                        st = max(prev_bound, en - 0.03)
                    w["start_sec"]=round(max(0.0,st),4); w["end_sec"]=round(max(w["start_sec"]+0.03,en),4)
                    w["duration_cs"]=max(3,int(round((w["end_sec"]-w["start_sec"])*100)))
                    next_start = w["start_sec"]

        # Enforce global monotonicity & recompute line windows from word spans
        for i in range(len(all_words)):
            st_i = float(all_words[i]["start_sec"])
            en_i = float(all_words[i]["end_sec"])
            if en_i <= st_i:
                en_i = st_i + 0.03
                all_words[i]["end_sec"] = round(en_i, 4)
                all_words[i]["duration_cs"] = max(3, int(round((en_i - st_i) * 100)))
            if i > 0:
                prev_en = float(all_words[i - 1]["end_sec"])
                if st_i < prev_en:
                    st_i = prev_en
                    all_words[i]["start_sec"] = round(st_i, 4)
                    if float(all_words[i]["end_sec"]) <= st_i:
                        all_words[i]["end_sec"] = round(st_i + 0.03, 4)
                    all_words[i]["duration_cs"] = max(3, int(round((float(all_words[i]["end_sec"]) - st_i) * 100)))

        for li, (ws, we) in enumerate(line_word_slices):
            if we > ws:
                windows[li]["start_sec"] = round(float(all_words[ws]["start_sec"]), 4)
                windows[li]["end_sec"] = round(float(all_words[we - 1]["end_sec"]), 4)
                if windows[li]["end_sec"] <= windows[li]["start_sec"]:
                    windows[li]["end_sec"] = round(windows[li]["start_sec"] + 0.05, 4)
                if li > 0 and windows[li]["start_sec"] < windows[li - 1]["end_sec"]:
                    windows[li]["start_sec"] = windows[li - 1]["end_sec"]
                    if windows[li]["end_sec"] <= windows[li]["start_sec"]:
                        windows[li]["end_sec"] = round(windows[li]["start_sec"] + 0.05, 4)

        # B2 instrumentation: additive audit fields per word
        for _w in all_words:
            post = float(_w.get("posterior", 0) or 0)
            conf = float(_w.get("confidence", 0) or 0)
            cem = bool(_w.get("is_cemented", False)) and conf >= self.word_target
            _w["is_cemented"] = cem
            if cem:
                _w["resolution_status"] = "cemented_evidence_backed"
                _w.pop("reject_reason", None)
            else:
                tier = str(_w.get("anchor_tier") or "")
                src = str(_w.get("confidence_source") or "")
                if "incoherent" in tier:
                    reason = "decemented_line_incoherent"
                elif "placement_unverified" in tier or "placement" in src:
                    reason = "placement_only"
                elif post < 0.55:
                    reason = "evidence_insufficient"
                elif post < 0.70:
                    reason = "below_target_after_floor"
                else:
                    reason = "evidence_contradicted_or_capped"
                _w["resolution_status"] = "uncemented"
                _w["reject_reason"] = reason

        wcount = len(all_words)
        cemented_end = sum(1 for w in all_words if w.get("is_cemented", False))
        uncemented_end = wcount - cemented_end
        log(f"[6] final cemented={cemented_end}/{wcount} uncemented={uncemented_end}")

        # --- Step 7: 1:1 Export & Anti-Fabrication Verification Gate ---
        self._export(lines, all_words, windows, out_dir, dur)

        project_name = Path(mp3_path).parent.name or "Project"
        manifest: Dict[str, Any] = {
            "project": project_name,
            "engine": "greenfield_v4_4_adaptive_mms_fa",
            "stem_separation_backend": getattr(self, "stem_separation_backend", "fallback_midside"),
            "vocal_gate": getattr(self, "vocal_gate_info", {}),
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
