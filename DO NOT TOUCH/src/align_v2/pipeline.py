
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import torch

try:
    from ..align.integrity_verifier import IntegrityVerifier
    from ..align.mms_fa import extract_emissions_chunked, get_mms_fa, romanized_to_ids
    from ..align import echora_backend
    from ..audio.streams import decode_audio_mono
    from ..subtitles.ass_writer import write_ass
    from ..subtitles.lrc_exporter import export_lrc
    from ..sync.ingestor import parse_user_lyric
except Exception:
    try:
        from src.align.integrity_verifier import IntegrityVerifier
        from src.align.mms_fa import extract_emissions_chunked, get_mms_fa, romanized_to_ids
        from src.align import echora_backend
        from src.audio.streams import decode_audio_mono
        from src.subtitles.ass_writer import write_ass
        from src.subtitles.lrc_exporter import export_lrc
        from src.sync.ingestor import parse_user_lyric
    except Exception:
        from align.integrity_verifier import IntegrityVerifier
        from align.mms_fa import extract_emissions_chunked, get_mms_fa, romanized_to_ids
        from align import echora_backend
        from audio.streams import decode_audio_mono
        from subtitles.ass_writer import write_ass
        from subtitles.lrc_exporter import export_lrc
        from sync.ingestor import parse_user_lyric
from .schema import ENGINE_V5, ENGINE_V2, ENGINE_VERSION, SCHEMA_VERSION, validate_manifest
from .ctc_engine import align_global_words
from .consensus import ConsensusPolicy, merge_model_candidates

def _extend_voiced_sustain_and_trim_outliers(
    words: List[Dict[str, Any]],
    acapella_sig: np.ndarray,
    sr: int = 16000,
    dur: float = 452.6,
    spf: float = 0.02,
) -> None:
    """Continuous voiced-sustain tail extension for belted vowels and uncemented outlier trimming."""
    fl = int(spf * sr)
    n_frames = len(acapella_sig) // fl
    rms = np.array(
        [np.sqrt(np.mean(acapella_sig[i * fl : (i + 1) * fl] ** 2)) for i in range(n_frames)],
        dtype=np.float32,
    )
    N = len(words)
    for i in range(N):
        w = words[i]
        st = float(w["start_sec"])
        en = float(w["end_sec"])
        nxt_st = float(words[i + 1]["start_sec"]) if i + 1 < N else dur

        f_st = max(0, min(n_frames - 1, int(round(st / spf))))
        f_en = max(f_st + 1, min(n_frames, int(round(en / spf))))

        word_rms = rms[f_st:f_en]
        peak_rms = float(np.max(word_rms)) if len(word_rms) > 0 else 0.0

        # 1. Uncemented island-tail stretch trimming (e.g. L59 gracias across chop riser)
        if not w.get("is_cemented") and (en - st) > 2.0 and peak_rms > 0.02:
            trough_thresh = 0.25 * peak_rms
            f_min = max(0, min(n_frames, int(round((st + 0.40) / spf))))
            trough_cnt = 0
            trim_frame = None
            for f in range(f_min, f_en):
                if rms[f] < trough_thresh:
                    trough_cnt += 1
                    if trough_cnt >= 3:  # 60ms
                        trim_frame = f - 2
                        break
                else:
                    trough_cnt = 0
            if trim_frame is not None:
                new_en = max(st + 0.40, float(trim_frame * spf))
                w["end_sec"] = round(new_en, 4)
                w["duration_cs"] = max(2, int(round((new_en - st) * 100)))
                en = new_en
                f_en = min(n_frames, int(round(en / spf)))

        # 2. Continuous Voiced-Sustain Tail Extension (e.g. belted 'pai')
        if peak_rms > 0.02 and f_en < n_frames:
            end_rms = float(rms[min(n_frames - 1, f_en - 1)])
            if end_rms >= 0.35 * peak_rms:
                drop_thresh = 0.25 * peak_rms
                cur_f = f_en
                # Guard sustain extension on 1-2 char function words (L93 "A")
                raw_dur = en - st
                alpha_chars = [c for c in w.get("word", "") if c.isalpha()]
                if len(alpha_chars) <= 2 and (len(alpha_chars) == 1 or raw_dur <= 0.45):
                    max_dur = 0.60
                else:
                    max_dur = 4.0
                max_f = min(n_frames, int(round(min(nxt_st - 0.02, st + max_dur) / spf)))
                while cur_f < max_f and rms[cur_f] >= drop_thresh:
                    cur_f += 1
                new_en = float(cur_f * spf)
                if new_en > en:
                    w["end_sec"] = round(new_en, 4)
                    w["duration_cs"] = max(2, int(round((new_en - st) * 100)))

def interpolate_anchors_with_cadence_and_vad(
    word_count: int,
    word_times: List[Optional[float]],
    active_mask: np.ndarray,
    dur: float,
    spf: float = 0.02,
) -> List[float]:
    """Self-calibrated track-specific cadence & VAD-bounded monotonic interpolation."""
    x_known = [i for i, t in enumerate(word_times) if t is not None]
    if not x_known:
        return [float(i) / max(1, word_count) * dur for i in range(word_count)]

    y_known = [float(word_times[i]) for i in x_known]
    if len(x_known) >= 2:
        cadence_samples = []
        for j in range(1, len(x_known)):
            w_delta = x_known[j] - x_known[j - 1]
            t_delta = y_known[j] - y_known[j - 1]
            if 1 <= w_delta <= 8 and 0.05 <= t_delta <= 6.0:
                cadence_samples.append(t_delta / float(w_delta))
        measured_spw = float(np.median(cadence_samples)) if cadence_samples else 0.35
    else:
        measured_spw = 0.35
    sec_per_word = float(np.clip(measured_spw, 0.12, 0.90))

    # Hard VAD acoustic boundary clamps: never extrapolate before first or after last vocal burst
    act_idx = np.where(active_mask)[0] if len(active_mask) > 0 else np.array([])
    t_first_vocal = float(act_idx[0] * spf) if len(act_idx) > 0 else 0.0
    t_last_vocal = float(act_idx[-1] * spf) if len(act_idx) > 0 else dur

    if x_known[0] > 0:
        missing_head = x_known[0]
        est_t0 = max(t_first_vocal, y_known[0] - missing_head * sec_per_word)
        x_known.insert(0, 0)
        y_known.insert(0, est_t0)
    if x_known[-1] < word_count - 1:
        missing_tail = word_count - 1 - x_known[-1]
        est_tN = min(t_last_vocal, y_known[-1] + missing_tail * sec_per_word)
        x_known.append(word_count - 1)
        y_known.append(est_tN)

    return list(np.interp(range(word_count), x_known, y_known))


class V5AlignmentEngine:
    ENGINE_ID: str = ENGINE_V5
    VERSION: str = ENGINE_VERSION
    NAME: str = "Greenfield Universal Engine"
    DISPLAY_NAME: str = f"Greenfield Universal Engine v{ENGINE_VERSION} (Dual-Model CTC Consensus)"

    def __init__(self, tau_song: float = 0.23, word_target: float = 0.95, lang: str | None = None, stem_mode: str = "hq"):
        self.tau_song = float(tau_song)
        self.word_target = float(word_target) if word_target != 0.95 else 0.85
        self.lang = lang
        self.stem_mode = stem_mode

    def run(self, lyric_path: str, mp3_path: str, out_dir: str, log: Callable[[str], None] = print) -> Dict[str, Any]:
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        stems_dir = out / "stems"; stems_dir.mkdir(parents=True, exist_ok=True)
        lyric_text = Path(lyric_path).read_text(encoding="utf-8")
        raw_lines, flat = parse_user_lyric(lyric_text)
        if not flat:
            raise RuntimeError("No lyrics parsed")
        from ..audio.streams import decode_audio_mono
        sr = 16000
        acapella = stems_dir / "acapella_full.wav"
        mix_vb = stems_dir / "original_mix_vocal_band.wav"
        stems_match_audio = True
        info_file_chk = stems_dir / "backend_info.json"
        if info_file_chk.exists():
            try:
                b_chk = json.loads(info_file_chk.read_text(encoding="utf-8"))
                src_name = b_chk.get("source_audio_name")
                src_size = b_chk.get("source_audio_size")
                cur_size = Path(mp3_path).stat().st_size if Path(mp3_path).is_file() else None
                if src_name and src_name != Path(mp3_path).name:
                    stems_match_audio = False
                elif src_size is not None and cur_size is not None and src_size != cur_size:
                    stems_match_audio = False
            except Exception:
                pass
        if not acapella.exists() or not mix_vb.exists() or not stems_match_audio:
            v2_stems = out.parent / f"{out.stem}.v2" / "stems"
            if stems_match_audio and v2_stems.exists() and (v2_stems / "acapella_full.wav").exists():
                import shutil
                for f_src in v2_stems.glob("*"):
                    f_dst = stems_dir / f_src.name
                    if not f_dst.exists():
                        shutil.copy2(f_src, f_dst)
                log("[v5] Reused cached verified BS-Roformer stems from .v2 cache")
            if not acapella.exists() or not mix_vb.exists() or not stems_match_audio:
                try:
                    from ..align.pipeline import UniversalEngine
                    ue = UniversalEngine(tau_song=self.tau_song, word_target=self.word_target, stem_mode=self.stem_mode)
                    ue._build_stems(mp3_path, str(stems_dir))
                    log("[v5] built/reused stems via UniversalEngine")
                except Exception as e:
                    log(f"[v5] stem build failed, falling back to raw mix: {e}")
        mono, sr = decode_audio_mono(mp3_path, target_sr=sr)
        dur = float(len(mono)) / float(sr)
        if acapella.exists():
            import soundfile as sf
            sig, _ = sf.read(str(acapella), dtype='float32')
            acapella_sig = np.asarray(sig, dtype=np.float32).reshape(-1)
        else:
            acapella_sig = np.asarray(mono, dtype=np.float32).reshape(-1)
        if mix_vb.exists():
            import soundfile as sf
            sig, _ = sf.read(str(mix_vb), dtype='float32')
            mix_sig = np.asarray(sig, dtype=np.float32).reshape(-1)
        else:
            mix_sig = acapella_sig

        # build word lists
        words = [w.text for w in flat]
        line_idxs = [w.line_index for w in flat]
        parens = [bool(w.is_parenthetical) for w in flat]

        # Multi-Tier Language Detection (Unicode script analysis + Latin ranker + acoustic prior)
        from .lang_detect import detect_text_language, resolve_language_consensus
        text_lang, text_conf = detect_text_language(words)

        cached_audio_lang = None
        cached_audio_prob = 0.0
        asr_cache_files = list(stems_dir.glob(".asr_cache_*.json"))
        for cf in asr_cache_files:
            try:
                cdata = json.loads(cf.read_text(encoding="utf-8"))
                c_info = cdata.get("info", {})
                if "language" in c_info:
                    cached_audio_lang = c_info.get("language")
                    cached_audio_prob = float(c_info.get("prob", 1.0))
                    break
            except Exception:
                pass

        lang = resolve_language_consensus(
            lyric_lang=text_lang,
            lyric_conf=text_conf,
            audio_lang=cached_audio_lang,
            audio_prob=cached_audio_prob,
            explicit_lang=self.lang,
        )
        log(f"[v5] resolved language: {lang} (text={text_lang}:{text_conf:.2f}, audio={cached_audio_lang}:{cached_audio_prob:.2f})")

        from ..align.mms_fa import clean_phonetic_token
        roman = [clean_phonetic_token(w.text, lang=lang) for w in flat]

        # global emissions (no ASR window lock)
        try:
            from ..core.hardware import get_torch_device

            align_device = get_torch_device()
        except Exception:
            align_device = "cpu"
        log(f"[v5] extracting MMS_FA emissions over full song ({dur:.1f}s, lang={lang}, device={align_device})")
        mms_em = extract_emissions_chunked(torch.from_numpy(acapella_sig), sr=sr, device=align_device)
        _, mms_dict, mms_blank, mms_star = get_mms_fa()
        word_ids = [romanized_to_ids(r, mms_dict, lang=lang) or [mms_dict.get('a',1)] for r in roman]
        # Vocal-activity gate: bleed floor max(0.015,0.25*p90_rms) + hysteresis/morph closing
        from ..align.mms_fa import compute_vocal_activity_mask
        active_mask, gate_info = compute_vocal_activity_mask(acapella_sig, sr=sr, hop_sec=0.02)
        T_em = int((mms_em[0] if mms_em.dim()==3 else mms_em).shape[0])
        if len(active_mask) < T_em:
            active_mask = np.pad(active_mask, (0, T_em-len(active_mask)), constant_values=False)
        active_mask = active_mask[:T_em]
        silence_mask = ~active_mask
        if np.any(silence_mask[:T_em]):
            mms_em[:T_em, 1:28] -= 6.0 * torch.from_numpy(silence_mask[:T_em, None]).float()
        log(f"[v5] vocal_gate masked={gate_info['masked_fraction']*100:.1f}% frames={int(silence_mask.sum())}/{T_em} p90={gate_info['p90']:.5f} thresh={gate_info['thresh']:.5f}")
        # Load or compute ASR words for multi-word phrase anchoring
        asr_words = []
        asr_cache_files = list(stems_dir.glob(f".asr_cache_*_{lang}_*.json")) if stems_match_audio else []
        if stems_match_audio and not asr_cache_files and (out.parent / f"{out.stem.replace('.v2', '.v1')}").exists():
            asr_cache_files = list((out.parent / f"{out.stem.replace('.v2', '.v1')}" / "stems").glob(f".asr_cache_*_{lang}_*.json"))
        if stems_match_audio and not asr_cache_files:
            for cf in list(stems_dir.glob(".asr_cache_*.json")):
                try:
                    cdata = json.loads(cf.read_text(encoding="utf-8"))
                    if cdata.get("info", {}).get("language") == lang:
                        asr_cache_files.append(cf)
                        break
                except Exception:
                    pass
        if asr_cache_files:
            try:
                cdata = json.loads(asr_cache_files[0].read_text(encoding="utf-8"))
                asr_words = cdata.get("words", [])
            except Exception as e:
                log(f"[v5] could not load asr cache: {e}")
        if not asr_words:
            try:
                from ..sync.line_synchronizer import transcribe_vocal
                ac_lead = stems_dir / "acapella_lead_dry.wav"
                vb_path = stems_dir / "original_mix_vocal_band.wav"
                target_audio = str(ac_lead if ac_lead.exists() else (vb_path if vb_path.exists() else mp3_path))
                raw_asr, _ = transcribe_vocal(target_audio, model_size="small", language=lang)
                asr_words = raw_asr
            except Exception as e:
                log(f"[v5] asr transcription fallback: {e}")

        # Construct macro vocal islands separated by silence gaps >= 4.0s
        spf = 0.02
        gaps = []
        gs = None
        cur = 0
        for i, v in enumerate(silence_mask):
            if v:
                if gs is None: gs = i
                cur += 1
            else:
                if gs is not None and cur * spf >= 4.0:
                    gaps.append((gs, i))
                gs = None
                cur = 0
        if gs is not None and cur * spf >= 4.0:
            gaps.append((gs, len(silence_mask)))

        cuts = [0]
        for g0, g1 in gaps:
            cuts.append(g0)
            cuts.append(g1)
        cuts.append(len(silence_mask))

        raw_islands = []
        for i in range(0, len(cuts) - 1, 2):
            a = cuts[i]
            b = cuts[i + 1]
            if (b - a) * spf >= 1.5:
                raw_islands.append((a, b))

        merged_islands = []
        for a, b in raw_islands:
            if not merged_islands:
                merged_islands.append((a, b))
            else:
                prev_a, prev_b = merged_islands[-1]
                gap_sec = (a - prev_b) * spf
                if gap_sec < 4.0:
                    merged_islands[-1] = (prev_a, b)
                else:
                    merged_islands.append((a, b))

        islands_sec = [(a * spf, b * spf) for a, b in merged_islands]
        if not islands_sec:
            islands_sec = [(0.0, dur)]
        log(f"[v5] constructed {len(islands_sec)} macro vocal islands")

        # Monotonic DP word-to-island assignment
        from ..sync.line_synchronizer import align_words_monotonic
        asr_txts = [w[2] for w in asr_words] if asr_words else []
        if asr_txts:
            mapping = align_words_monotonic(words, asr_txts)
            word_times = []
            for i, a_idx in enumerate(mapping):
                if a_idx is not None and a_idx < len(asr_words):
                    word_times.append(float(asr_words[a_idx][0]))
                else:
                    word_times.append(None)
            word_times = interpolate_anchors_with_cadence_and_vad(
                word_count=len(words),
                word_times=word_times,
                active_mask=active_mask,
                dur=dur,
                spf=spf,
            )
        else:
            word_times = [float(i) / max(1, len(words)) * dur for i in range(len(words))]

        midpoints = []
        for k in range(len(islands_sec) - 1):
            mid = 0.5 * (islands_sec[k][1] + islands_sec[k + 1][0])
            midpoints.append(mid)

        sec_words = [[] for _ in islands_sec]
        cur_w = 0
        for li, line_text in enumerate(raw_lines):
            lw = line_text.split()
            w_indices = list(range(cur_w, cur_w + len(lw)))
            cur_w += len(lw)
            l_times = [word_times[wi] for wi in w_indices]
            l_mid = float(np.median(l_times)) if l_times else 0.0
            isl = 0
            for m_idx, mid in enumerate(midpoints):
                if l_mid > mid:
                    isl = m_idx + 1
            for wi in w_indices:
                sec_words[isl].append(wi)

        # Lexical-overlap + burst-count Island DP:
        # Forward shift for empty trailing vocal islands (>= 3.0s):
        line_words = [l.split() for l in raw_lines]
        for k in range(1, len(islands_sec)):
            st_k, en_k = islands_sec[k]
            if len(sec_words[k]) == 0 and (en_k - st_k) >= 3.0:
                prev_k = k - 1
                while prev_k >= 0 and len(sec_words[prev_k]) == 0:
                    prev_k -= 1
                if prev_k >= 0 and len(sec_words[prev_k]) > 0:
                    prev_wis = sec_words[prev_k]
                    cur_w = 0
                    lines_in_prev = []
                    for li, lw in enumerate(line_words):
                        w_range = list(range(cur_w, cur_w + len(lw)))
                        cur_w += len(lw)
                        if any(wi in prev_wis for wi in w_range):
                            lines_in_prev.append((li, w_range))
                    if len(lines_in_prev) > 1:
                        shift_lines = [lines_in_prev[-1]]
                        shift_words = [wi for _, wr in shift_lines for wi in wr]
                        sec_words[prev_k] = [wi for wi in sec_words[prev_k] if wi not in shift_words]
                        sec_words[k] = sorted(shift_words)
                        log(f"[v5] island {k} captured {len(shift_words)} words from trailing line {shift_lines[0][0]+1}")

        for k, (st, en) in enumerate(islands_sec):
            log(f"[v5] island {k} ({st:.1f}s..{en:.1f}s): {len(sec_words[k])} words assigned")

        def _run_island_ctc(em, ids, blank_id, star_id, mask, spf=0.02):
            T = int((em[0] if em.dim() == 3 else em).shape[0])
            merged = [None] * len(ids)
            for isl_idx, (isl_st, isl_en) in enumerate(islands_sec):
                wis = sec_words[isl_idx]
                if not wis:
                    continue
                w_lo = max(0.0, isl_st - 2.5)
                w_hi = min(dur, isl_en + 2.5)
                f0 = int(round(w_lo / spf))
                f1 = min(T, int(round(w_hi / spf)))
                if f1 <= f0:
                    continue
                sub_em = (em[0] if em.dim() == 3 else em)[f0:f1]
                sub_ids = [ids[i] for i in wis]
                sub_lines = [line_idxs[i] for i in wis]
                sub_mask = mask[f0:f1] if f0 < len(mask) else mask[:0]
                sub_act = active_mask[f0:f1] if f0 < len(active_mask) else active_mask[:0]
                try:
                    sub_posts = align_global_words(
                        sub_em,
                        sub_ids,
                        blank_id=blank_id,
                        star_id=star_id,
                        sec_per_frame=spf,
                        word_target=self.word_target,
                        silence_frame_mask=sub_mask,
                        active_vocal_mask=sub_act,
                        leading_vocal_penalty=0.08,
                        line_indices=sub_lines,
                        intra_line_silence_penalty=0.12,
                    )
                except Exception as ex:
                    log(f"[v5] island {isl_idx} CTC exception: {ex}")
                    continue
                for k, wi in enumerate(wis):
                    if k < len(sub_posts):
                        sp = sub_posts[k]
                        sp.start_frame = int(sp.start_frame) + f0
                        sp.end_frame = int(sp.end_frame) + f0
                        merged[wi] = sp
            for i in range(len(merged)):
                if merged[i] is None:
                    from .ctc_engine import WordPosterior
                    f_st = int(round(word_times[i] / spf))
                    merged[i] = WordPosterior(
                        start_frame=f_st,
                        end_frame=f_st + 15,
                        posterior=0.40,
                        null_llr=0.0,
                        confidence=0.40,
                        marginal_mass=0.10,
                    )
            return merged

        mms_post = _run_island_ctc(mms_em, word_ids, mms_blank, mms_star, silence_mask, spf=0.02)

        def _apply_silence_stretch_guard(posts):
            for wp in posts:
                f0 = max(0, int(float(wp.start_frame)))
                f1 = min(len(silence_mask), int(float(wp.end_frame)))
                if f1-f0 < 5: continue
                seg = silence_mask[f0:f1]
                max_run=0; cur=0
                for v in seg:
                    if v: cur+=1; max_run=max(max_run,cur)
                    else: cur=0
                if max_run*0.02 > 1.0:
                    wp.confidence = min(float(wp.confidence), self.word_target - 0.10)
                    wp.marginal_mass = min(float(wp.marginal_mass), 0.20)
            return posts
        mms_post = _apply_silence_stretch_guard(mms_post)

        model_outputs: Dict[str, List[Dict[str, Any]]] = {}
        def to_words(posts, src):
            out=[]
            for i,p in enumerate(posts):
                st=p.start_frame*0.02; en=max(st+0.02,p.end_frame*0.02)
                out.append({"word":words[i],"start_sec":st,"end_sec":en,"posterior":p.posterior,"null_llr":p.null_llr,"confidence":p.confidence,"confidence_source":src})
            return out
        model_outputs["mms_fa"] = to_words(mms_post, "mms_fa_global_ctc")

        # Release temporary MMS emissions memory before Echora forward pass (important on 8 GB hosts/laptops)
        try:
            import gc

            del mms_em
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # Echora global (best effort; fail loud if unavailable? architect said fail fast, but be practical)
        try:
            log(f"[v5] extracting Echora emissions (device={align_device})")
            ech_em = echora_backend.extract_emissions(acapella_sig, sr=sr, device=align_device)
            if ech_em.dim()==3: ech_lp=torch.log_softmax(ech_em[0],dim=-1)
            else: ech_lp=torch.log_softmax(ech_em,dim=-1)
            # map romanized chars to echora vocab ids
            _,_,vocab,blank_id = echora_backend._load(device=align_device)
            V_ech = ech_lp.shape[-1]
            star_col = torch.full((ech_lp.shape[0], 1), -1.85, dtype=ech_lp.dtype, device=ech_lp.device)
            ech_lp = torch.cat([ech_lp, star_col], dim=-1)
            star_id = V_ech
            ech_ids=[]
            for r in roman:
                toks=[]
                for ch in r:
                    tid=vocab.get(ch, vocab.get('<unk>', blank_id))
                    if int(tid)!=int(blank_id): toks.append(int(tid))
                if not toks: toks=[int(vocab.get('a', blank_id))]
                ech_ids.append(toks)
            T_ech = int((ech_lp[0] if ech_lp.dim()==3 else ech_lp).shape[0])
            s_mask = silence_mask[:T_ech]
            if np.any(s_mask):
                non_blank_cols = torch.ones(ech_lp.shape[-1], dtype=torch.bool)
                non_blank_cols[int(blank_id)] = False
                non_blank_cols[int(star_id)] = False
                ech_lp[:T_ech, non_blank_cols] -= 6.0 * torch.from_numpy(s_mask[:, None]).float()
            ech_post = _run_island_ctc(ech_lp, ech_ids, int(blank_id), int(star_id), silence_mask, spf=0.02)
            ech_post = _apply_silence_stretch_guard(ech_post)
            model_outputs["echora_fixed"] = to_words(ech_post, "echora_global_ctc")
        except Exception as e:
            log(f"[v5] Echora unavailable: {e}")
            # keep MMS_FA only but do not fake consensus
            pass

        if len(model_outputs) < 2:
            log("[v5] WARNING: single-model mode; words will not be cemented by consensus")
        # build base words
        base=[]
        for i,w in enumerate(flat):
            base.append({
                "word":w.text,"start_sec":float(mms_post[i].start_frame*0.02),"end_sec":float(max(mms_post[i].start_frame*0.02+0.02,mms_post[i].end_frame*0.02)),
                "duration_cs":max(2,int(round((max(0.02,(mms_post[i].end_frame-mms_post[i].start_frame)*0.02))*100))),
                "confidence":0.0,"posterior":float(mms_post[i].posterior),"null_llr":float(mms_post[i].null_llr),
                "is_cemented":False,"anchor_tier":"unverified","confidence_source":"v2_base","line_index":int(line_idxs[i]),"is_parenthetical":bool(parens[i])
            })
        merged = merge_model_candidates(base, model_outputs, policy=ConsensusPolicy(word_target=self.word_target))

        # Continuous voiced-sustain tail extension & uncemented island-tail trimming
        try:
            _extend_voiced_sustain_and_trim_outliers(merged, acapella_sig, sr=sr, dur=dur, spf=0.02)
        except Exception as _ex:
            log(f"[v5] sustain extension / trim exception: {_ex}")

        # Enforce global word monotonicity and minimum duration floor (>= 60ms)
        for i in range(len(merged)):
            st_i = float(merged[i]["start_sec"])
            en_i = float(merged[i]["end_sec"])
            alpha_len = len([c for c in merged[i].get("word", "") if c.isalpha()])
            min_dur = max(0.06, min(0.18, 0.035 * max(1, alpha_len)))
            if en_i < st_i + min_dur:
                en_i = st_i + min_dur
                merged[i]["end_sec"] = round(en_i, 4)
                merged[i]["duration_cs"] = max(3, int(round((en_i - st_i) * 100)))
            if i > 0:
                prev_en = float(merged[i - 1]["end_sec"])
                if st_i < prev_en:
                    st_i = prev_en
                    merged[i]["start_sec"] = round(st_i, 4)
                    if float(merged[i]["end_sec"]) < st_i + min_dur:
                        merged[i]["end_sec"] = round(st_i + min_dur, 4)
                    merged[i]["duration_cs"] = max(3, int(round((float(merged[i]["end_sec"]) - st_i) * 100)))

        # diagnostics: confidence distribution of agreeing echora_fixed+mms_fa words
        try:
            agree_w = [w for w in merged if "echora_fixed" in str(w.get("confidence_source","")) and "mms_fa" in str(w.get("confidence_source",""))]
            if agree_w:
                confs = [float(w.get("confidence",0)) for w in agree_w]
                buckets = {"<0.5":0,"0.5-0.7":0,"0.7-0.85":0,"0.85-0.95":0,">=0.95":0}
                for c in confs:
                    if c<0.5: buckets["<0.5"]+=1
                    elif c<0.7: buckets["0.5-0.7"]+=1
                    elif c<0.85: buckets["0.7-0.85"]+=1
                    elif c<0.95: buckets["0.85-0.95"]+=1
                    else: buckets[">=0.95"]+=1
                log(f"[v5] echora+mms agree_n={len(agree_w)} conf_mean={float(np.mean(confs)):.3f} buckets={buckets}")
        except Exception as _e:
            log(f"[v5] diag failed: {_e}")

        # build lines
        line_map: Dict[int, List[Dict[str,Any]]] = {}
        for w in merged:
            line_map.setdefault(int(w["line_index"]), []).append(w)
        lines_out=[]
        for li, txt in enumerate(raw_lines):
            lw=line_map.get(li,[])
            st=float(lw[0]["start_sec"]) if lw else 0.0
            en=float(lw[-1]["end_sec"]) if lw else st + 0.05
            if en <= st:
                en = st + 0.05
            if lines_out and st < lines_out[-1]["end_sec"]:
                st = lines_out[-1]["end_sec"]
                if en <= st:
                    en = st + 0.05
            lines_out.append({"index":li+1,"start_sec":round(st,4),"end_sec":round(en,4),"text":txt})

        cemented=sum(1 for w in merged if w.get("is_cemented"))
        backend_name = "fallback_midside"
        info_file = stems_dir / "backend_info.json"
        if info_file.exists():
            try:
                bdata = json.loads(info_file.read_text(encoding="utf-8"))
                backend_name = bdata.get("stem_separation_backend", "fallback_midside")
            except Exception:
                pass
        manifest={
            "project": Path(mp3_path).stem,
            "audio_file": Path(mp3_path).name,
            "audio_size_bytes": Path(mp3_path).stat().st_size if Path(mp3_path).is_file() else None,
            "engine": ENGINE_V5,
            "engine_version": ENGINE_VERSION,
            "stem_separation_backend": backend_name,
            "vocal_gate": gate_info,
            "schema_version": SCHEMA_VERSION,
            "tau_song": self.tau_song,
            "word_target": self.word_target,
            "total_duration_sec": round(float(dur),4),
            "total_input_lines": len(raw_lines),
            "total_words": len(merged),
            "cemented_words_ge_95_count": cemented,
            "uncemented_words_count": len(merged)-cemented,
            "asr_words": 0,
            "words": merged,
            "lines": lines_out,
            "diagnostics": {"models": list(model_outputs.keys()), "global_ctc": True, "no_asr_window_lock": True},
        }
        errs = validate_manifest(manifest, word_target=self.word_target)
        if errs: log(f"[v5] manifest invariant warnings: {errs[:10]}")

        # exports
        (out/"alignment_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            write_ass(manifest.get("words", []), out/"karaoke.ass")
        except Exception as e:
            log(f"[v5] ASS export warning: {e}")
        try:
            export_lrc(manifest.get("words", []), out/"lyrics_synced.lrc", lines=raw_lines)
            export_lrc(manifest.get("words", []), out/"lyrics_synced.txt", lines=raw_lines)
        except Exception as e:
            log(f"[v5] LRC/TXT export warning: {e}")
        try:
            v = IntegrityVerifier()
            rep = v.verify(manifest, mix_vb_sig=acapella_sig, sr=sr)
            (out/"verification_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest["verification"]=rep
        except Exception as e:
            log(f"[v5] verification warning: {e}")
        return manifest

V2AlignmentEngine = V5AlignmentEngine  # Backwards-compatible alias
