import math, sys
from pathlib import Path
import numpy as np
import torch
# add DO NOT TOUCH and DO NOT TOUCH/src to path
dnt = Path(__file__).resolve().parents[1] / "DO NOT TOUCH"
src = dnt / "src"
sys.path.insert(0, str(dnt))
sys.path.insert(0, str(src))

try:
    from src.align_v2.ctc_engine import _alpha_beta_marginals
    from src.align_v2.consensus import ConsensusPolicy, merge_model_candidates
except ImportError:
    from align_v2.ctc_engine import _alpha_beta_marginals
    from align_v2.consensus import ConsensusPolicy, merge_model_candidates


def test_stride320_conv_output_length_invariant():
    # Wav2Vec2 conv stack: (10,5),(3,2),(3,2),(3,2),(3,2),(2,2),(2,2)
    layers = [(10,5),(3,2),(3,2),(3,2),(3,2),(2,2),(2,2)]
    for t_expected in (1,2,50,1000,11315):
        n = 320*t_expected + 80
        cur=n
        for k,s in layers:
            cur = (cur-k)//s + 1
        assert cur == t_expected


def test_alpha_beta_exposes_ambiguous_alignment():
    n_frames,vocab=80,5
    blank,tok_a,star=0,1,4
    lp=np.full((n_frames,vocab), -8.0, dtype=np.float32)
    lp[:,blank]=-0.05; lp[:,star]=-2.2
    lp[10:14,tok_a]=-0.05; lp[10:14,blank]=-5.0
    lp[60:64,tok_a]=-0.05; lp[60:64,blank]=-5.0
    targets=[star, tok_a, star]
    gamma,_ = _alpha_beta_marginals(lp, targets, blank_id=blank)
    assert np.allclose(gamma.sum(axis=1), 1.0, atol=1e-3)
    state_a=3
    mass_peak1=float(gamma[8:16,state_a].sum())
    mass_total=float(gamma[:,state_a].sum())
    ratio=mass_peak1/max(mass_total,1e-6)
    assert 0.25 <= ratio <= 0.75, ratio


def test_contradiction_blocks_cementation_and_anchors_immutable():
    base=[{"word":"Que","start_sec":1.0,"end_sec":1.3,"posterior":0.8,"null_llr":1.0,"confidence":0.8,"is_cemented":False,"line_index":0,"is_parenthetical":False},
          {"word":"no","start_sec":1.35,"end_sec":1.55,"posterior":0.8,"null_llr":1.0,"confidence":0.8,"is_cemented":False,"line_index":0,"is_parenthetical":False},
          {"word":"salgo","start_sec":2.4,"end_sec":2.8,"posterior":0.8,"null_llr":1.0,"confidence":0.8,"is_cemented":False,"line_index":0,"is_parenthetical":False}]
    model_outputs={
        "mms_fa":[{"start_sec":1.0,"end_sec":1.3,"posterior":0.9,"null_llr":2.0,"confidence":0.96},
                  {"start_sec":1.35,"end_sec":1.55,"posterior":0.85,"null_llr":1.5,"confidence":0.90},
                  {"start_sec":2.4,"end_sec":2.85,"posterior":0.9,"null_llr":2.0,"confidence":0.96}],
        "echora_fixed":[{"start_sec":1.02,"end_sec":1.32,"posterior":0.9,"null_llr":2.0,"confidence":0.96},
                        {"start_sec":2.0,"end_sec":2.25,"posterior":0.85,"null_llr":1.8,"confidence":0.90},
                        {"start_sec":2.42,"end_sec":2.88,"posterior":0.9,"null_llr":2.0,"confidence":0.96}],
    }
    merged = merge_model_candidates(base, model_outputs, policy=ConsensusPolicy(word_target=0.95))
    assert merged[0]["is_cemented"] is True
    assert merged[1]["is_cemented"] is False
    assert merged[2]["is_cemented"] is True
    # immutable anchors not moved by placement
    s0=(merged[0]["start_sec"], merged[0]["end_sec"])
    s2=(merged[2]["start_sec"], merged[2]["end_sec"])
    merged2 = merge_model_candidates(base, model_outputs, policy=ConsensusPolicy(word_target=0.95))
    assert (merged2[0]["start_sec"], merged2[0]["end_sec"]) == s0
    assert (merged2[2]["start_sec"], merged2[2]["end_sec"]) == s2


def test_clean_phonetic_token_nfk_diacritics():
    try:
        from align.mms_fa import clean_phonetic_token
    except Exception:
        from src.align.mms_fa import clean_phonetic_token
    assert clean_phonetic_token("¿qué?") == "que"
    assert clean_phonetic_token("enseñarme") == "ensenarme"
    assert clean_phonetic_token("días") == "dias"
    assert clean_phonetic_token("Después") == "despues"


def test_render_start_time_pts_shift():
    from pipeline.render import build_ffmpeg_command
    from pathlib import Path
    import tempfile, json
    # minimal config
    config = {"ffmpeg_path":"ffmpeg","video_width":1080,"video_height":1920,"fps":30,"codec":"libx264","crf":23,"audio_codec":"aac"}
    effect_cfg = {"background":{"style":"Solid Black"},"lyrics":{},"visualizer":{"style":"none","handle":""}}
    with tempfile.TemporaryDirectory() as td:
        td=Path(td)
        ass=td/"k.ass"; ass.write_text("[Script Info]\n[Events]\nDialogue: 0,0:04:25.00,0:04:26.00,Default,,0,0,0,,test\n",encoding='utf-8')
        audio=td/"a.mp3"; audio.write_bytes(b'')
        out=td/"out.mp4"
        cmd = build_ffmpeg_command(config,effect_cfg,audio,None,td,ass,out,duration=30.0,start_time=265.0)
        fc = " ".join(cmd)
        assert "setpts=PTS-STARTPTS+265.000/TB" in fc
        assert "ass='" in fc
        assert "setpts=PTS-STARTPTS[final]" in fc


def test_v2_smoke_imports_and_schema():
    # smoke: imports and basic schema objects don't crash
    from align_v2.ctc_engine import WordPosterior, align_global_words
    import numpy as np, torch
    wp = WordPosterior(start_frame=0,end_frame=5,posterior=0.9,null_llr=1.0,confidence=0.9,marginal_mass=0.5)
    assert wp.marginal_mass == 0.5
    lp = torch.log_softmax(torch.randn(20,5), dim=-1)
    # trivial single word
    posts = align_global_words(lp, [[1,2]], blank_id=0, star_id=4, sec_per_frame=0.02, word_target=0.85)
    assert len(posts)==1
    assert hasattr(posts[0],'marginal_mass')


def test_stem_selector_rejects_instrumental_trap():
    from src.align.pipeline import select_vocal_stem
    outputs = [
        "foo_(Instrumental)_Kim_Vocal_2.wav",
        "foo_(Vocals)_Kim_Vocal_2.wav",
    ]
    selected = select_vocal_stem(outputs)
    assert selected == "foo_(Vocals)_Kim_Vocal_2.wav"
    assert select_vocal_stem(["foo_(Instrumental)_Kim_Vocal_2.wav"]) is None


def test_anti_fabrication_gate_rejects_inverted_or_overlapping_timestamps():
    from src.align.integrity_verifier import IntegrityVerifier
    v = IntegrityVerifier(word_target=0.95)
    sr = 16000
    waveform = np.zeros(16000 * 10, dtype=np.float32)

    # 1. Inverted word: start >= end
    inverted_words = [
        {"word": "a", "start_sec": 1.0, "end_sec": 0.5, "confidence": 0.96, "is_cemented": True},
        {"word": "b", "start_sec": 1.5, "end_sec": 2.0, "confidence": 0.96, "is_cemented": True},
    ]
    windows = [{"index": 1, "start_sec": 0.5, "end_sec": 2.0, "text": "a b"}]
    res = v.anti_fabrication_gate(inverted_words, 10.0, waveform, sr, windows, "dummy")
    assert res["checks"]["temporal_monotonicity"]["pass"] is False
    assert res["overall_pass"] is False

    # 2. Overlapping words: prev.end > curr.start
    overlap_words = [
        {"word": "a", "start_sec": 1.0, "end_sec": 1.8, "confidence": 0.96, "is_cemented": True},
        {"word": "b", "start_sec": 1.5, "end_sec": 2.2, "confidence": 0.96, "is_cemented": True},
    ]
    res2 = v.anti_fabrication_gate(overlap_words, 10.0, waveform, sr, windows, "dummy")
    assert res2["checks"]["temporal_monotonicity"]["pass"] is False
    assert res2["overall_pass"] is False

    # 3. Inverted line: start >= end
    valid_words = [
        {"word": "a", "start_sec": 1.0, "end_sec": 1.4, "confidence": 0.96, "is_cemented": True},
        {"word": "b", "start_sec": 1.5, "end_sec": 2.0, "confidence": 0.96, "is_cemented": True},
    ]
    inverted_windows = [{"index": 1, "start_sec": 2.0, "end_sec": 1.0, "text": "a b"}]
    res3 = v.anti_fabrication_gate(valid_words, 10.0, waveform, sr, inverted_windows, "dummy")
    assert res3["checks"]["temporal_monotonicity"]["pass"] is False
    assert res3["overall_pass"] is False

    # 4. Overlapping lines: line 0 end > line 1 start
    overlap_windows = [
        {"index": 1, "start_sec": 1.0, "end_sec": 3.0, "text": "a"},
        {"index": 2, "start_sec": 2.5, "end_sec": 4.0, "text": "b"},
    ]
    res4 = v.anti_fabrication_gate(valid_words, 10.0, waveform, sr, overlap_windows, "dummy")
    assert res4["checks"]["temporal_monotonicity"]["pass"] is False
    assert res4["overall_pass"] is False


def test_intra_word_blank_penalty_preserves_compactness():
    from align_v2.ctc_engine import _viterbi_path
    T, V = 12, 5
    blank, c1, c2, star = 0, 1, 2, 4
    lp = np.full((T, V), -6.0, dtype=np.float32)
    lp[:, blank] = -0.05
    lp[:, star] = -1.85
    # Hypothesis A (spread out): c1 at t=2, c2 at t=8
    lp[2, c1] = 0.0
    # Hypothesis B (compact): c1 at t=7, c2 at t=8
    lp[7, c1] = -0.1
    lp[8, c2] = 0.0

    targets = [star, c1, c2, star]
    is_intra = np.zeros(9, dtype=bool)
    is_intra[4] = True  # blank between c1 and c2

    st_unpenalized, _, _ = _viterbi_path(lp, targets, blank, is_intra_word_blank=is_intra, intra_blank_penalty=0.0)
    st_penalized, _, _ = _viterbi_path(lp, targets, blank, is_intra_word_blank=is_intra, intra_blank_penalty=0.30)

    assert st_unpenalized[2] == 3  # enters word early at t=2
    assert st_penalized[2] <= 2    # stays in pre-word blank
    assert st_penalized[7] == 3    # enters word compactly at t=7
    assert st_penalized[8] == 5    # finishes word at t=8


def test_voiced_sustain_extension_and_trim():
    try:
        from src.align_v2.pipeline import _extend_voiced_sustain_and_trim_outliers
    except ImportError:
        from align_v2.pipeline import _extend_voiced_sustain_and_trim_outliers
    sr = 16000
    sig = np.zeros(sr * 10, dtype=np.float32)
    # word 0: 'pai' 1.0s to 1.1s, sustain to 1.8s
    sig[int(1.0 * sr) : int(1.8 * sr)] = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.8, int(0.8 * sr)))
    # word 1: 'gracias' 4.0s to 7.0s, vocal 4.0 to 4.5s, trough 4.5 to 5.0s
    sig[int(4.0 * sr) : int(4.5 * sr)] = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(0.5 * sr)))
    sig[int(5.5 * sr) : int(7.0 * sr)] = 0.4 * np.sin(2 * np.pi * 880 * np.linspace(0, 1.5, int(1.5 * sr)))

    words = [
        {"word": "pai", "start_sec": 1.0, "end_sec": 1.1, "duration_cs": 10, "is_cemented": True},
        {"word": "gracias", "start_sec": 4.0, "end_sec": 7.0, "duration_cs": 300, "is_cemented": False},
    ]

    _extend_voiced_sustain_and_trim_outliers(words, sig, sr=sr, dur=10.0, spf=0.02)
    assert words[0]["end_sec"] >= 1.75
    assert 4.4 <= words[1]["end_sec"] <= 4.6


def test_dual_model_consensus_calibration():
    base = [{"word": "test", "start_sec": 1.0, "end_sec": 1.3, "posterior": 0.1, "null_llr": 0.1, "confidence": 0.1, "is_cemented": False, "line_index": 0, "is_parenthetical": False}]

    # Case 1: MMS has low post (0.10) but Echora has strong post (0.85) and timing agrees tightly:
    model_outputs_agree = {
        "mms_fa": [{"start_sec": 1.0, "end_sec": 1.3, "posterior": 0.10, "null_llr": 0.2, "confidence": 0.5, "vocal_vad_ratio": 0.8}],
        "echora_fixed": [{"start_sec": 1.02, "end_sec": 1.32, "posterior": 0.85, "null_llr": 2.0, "confidence": 0.90, "vocal_vad_ratio": 0.8}],
    }
    merged = merge_model_candidates(base, model_outputs_agree, policy=ConsensusPolicy(word_target=0.85))
    assert merged[0]["is_cemented"] is True

    # Case 2: Timing agrees, but vad_ratio < 0.30 (silence):
    model_outputs_silent = {
        "mms_fa": [{"start_sec": 1.0, "end_sec": 1.3, "posterior": 0.70, "null_llr": 1.5, "confidence": 0.85, "vocal_vad_ratio": 0.1}],
        "echora_fixed": [{"start_sec": 1.02, "end_sec": 1.32, "posterior": 0.85, "null_llr": 2.0, "confidence": 0.90, "vocal_vad_ratio": 0.1}],
    }
    merged_silent = merge_model_candidates(base, model_outputs_silent, policy=ConsensusPolicy(word_target=0.85))
    assert merged_silent[0]["is_cemented"] is False
    assert merged_silent[0]["reject_reason"] == "low_vocal_vad_ratio"


def test_leading_active_vocal_penalty_anchors_first_repeat():
    from align_v2.ctc_engine import _viterbi_path
    T = 80
    V = 5
    blank, c1, c2, star = 0, 1, 2, 4

    lp = np.full((T, V), -6.0, dtype=np.float32)
    lp[:, blank] = -0.05
    lp[:, star] = -1.85

    # Phrase 1: t=10..25 (15 frames active vocal)
    lp[10:17, c1] = -0.25
    lp[17:25, c2] = -0.25

    # Phrase 2 (repeat, slightly higher log-prob by 0.05): t=50..65
    lp[50:57, c1] = -0.20
    lp[57:65, c2] = -0.20

    active_mask = np.zeros(T, dtype=bool)
    active_mask[10:25] = True
    active_mask[50:65] = True

    targets = [star, c1, c2, star]
    S = 9
    is_intra = np.zeros(S, dtype=bool)
    is_intra[4] = True

    is_leading = np.zeros(S, dtype=bool)
    is_leading[:3] = True

    st_unpenalized, _, _ = _viterbi_path(
        lp, targets, blank,
        is_intra_word_blank=is_intra, intra_blank_penalty=0.30,
        is_leading_blank_or_star=is_leading, leading_vocal_penalty=0.0,
        active_vocal_mask=active_mask,
    )
    st_penalized, _, _ = _viterbi_path(
        lp, targets, blank,
        is_intra_word_blank=is_intra, intra_blank_penalty=0.30,
        is_leading_blank_or_star=is_leading, leading_vocal_penalty=0.08,
        active_vocal_mask=active_mask,
    )

    w_start_unpen = np.where(st_unpenalized >= 3)[0][0]
    w_start_pen = np.where(st_penalized >= 3)[0][0]

    assert w_start_unpen >= 50, f"Unpenalized skipped to phrase 2 (frame {w_start_unpen})"
    assert w_start_pen < 25, f"Penalized anchored to phrase 1 (frame {w_start_pen})"


def test_function_word_sustain_cap():
    try:
        from src.align_v2.pipeline import _extend_voiced_sustain_and_trim_outliers
    except ImportError:
        from align_v2.pipeline import _extend_voiced_sustain_and_trim_outliers

    sr = 16000
    spf = 0.02
    dur = 10.0
    sig = np.zeros(int(dur * sr), dtype=np.float32)
    sig[0:int(2.0 * sr)] = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 2.0, int(2.0 * sr)))
    sig[int(3.0 * sr):int(5.0 * sr)] = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 2.0, int(2.0 * sr)))

    words = [
        {"word": "A", "start_sec": 0.0, "end_sec": 0.10, "duration_cs": 10, "is_cemented": False},
        {"word": "mi", "start_sec": 1.5, "end_sec": 1.7, "duration_cs": 20, "is_cemented": True},
        {"word": "pai", "start_sec": 3.0, "end_sec": 3.1, "duration_cs": 10, "is_cemented": True},
    ]

    _extend_voiced_sustain_and_trim_outliers(words, sig, sr=sr, dur=dur, spf=spf)

    # 1-char word 'A' is capped to <= 0.60s
    assert words[0]["end_sec"] <= 0.60, f"Expected 'A' end <= 0.60, got {words[0]['end_sec']}"
    # 3-char word 'pai' extends cleanly across sustained energy to >= 4.50s
    assert words[2]["end_sec"] >= 4.50, f"Expected 'pai' end >= 4.50, got {words[2]['end_sec']}"


def test_zero_hardcoded_song_timestamps_in_v2_pipeline():
    v2_pipeline_path = Path(__file__).resolve().parents[1] / "DO NOT TOUCH" / "src" / "align_v2" / "pipeline.py"
    text = v2_pipeline_path.read_text(encoding="utf-8")
    assert "425" not in text, "Found hardcoded '425' timestamp in align_v2/pipeline.py!"
    assert "max_gap = 10.0" not in text, "Found hardcoded 'max_gap = 10.0' in align_v2/pipeline.py!"


def test_ass_writer_clean_dialogue_invariants():
    import tempfile, re
    try:
        from src.subtitles.ass_writer import write_ass
    except ImportError:
        from subtitles.ass_writer import write_ass

    words = [
        {"word": "Llevo", "start_sec": 1.0, "end_sec": 1.5, "duration_cs": 50, "line_index": 0},
        {"word": "días", "start_sec": 1.58, "end_sec": 2.0, "duration_cs": 42, "line_index": 0},
        {"word": "que", "start_sec": 2.15, "end_sec": 2.4, "duration_cs": 25, "line_index": 0},
        {"word": "no", "start_sec": 2.4, "end_sec": 2.6, "duration_cs": 20, "line_index": 0},
        {"word": "salgo", "start_sec": 3.0, "end_sec": 3.5, "duration_cs": 50, "line_index": 1},
        {"word": "de", "start_sec": 3.55, "end_sec": 3.7, "duration_cs": 15, "line_index": 1},
        {"word": "casa", "start_sec": 3.75, "end_sec": 4.2, "duration_cs": 45, "line_index": 1},
    ]

    with tempfile.TemporaryDirectory() as td:
        ass_file = Path(td) / "test.ass"
        write_ass(words, ass_file)

        lines = ass_file.read_text(encoding="utf-8").splitlines()
        dialogues = [l for l in lines if l.startswith("Dialogue:")]
        assert len(dialogues) == 2, f"Expected 2 lines, got {len(dialogues)}"

        prev_end_sec = -1.0
        for d in dialogues:
            parts = d.split(",", 9)
            st_str, en_str, payload = parts[1], parts[2], parts[9]

            # 1. Zero consecutive double spaces
            clean = re.sub(r"\{[^}]*\}", "", payload)
            assert "  " not in clean, f"Found double space in: '{clean}'"

            def parse_t(s):
                h, m, sec_cs = s.split(":")
                sec, cs = sec_cs.split(".")
                return int(h) * 3600 + int(m) * 60 + int(sec) + int(cs) / 100.0

            st_sec = parse_t(st_str)
            en_sec = parse_t(en_str)

            # 2. Zero same-slot overlaps
            assert st_sec >= prev_end_sec - 0.001, f"Overlap detected: {st_sec} < {prev_end_sec}"
            prev_end_sec = en_sec

            # 3. Zero last-word truncations (>20ms)
            k_durations = [int(k) for k in re.findall(r"\\k(\d+)", payload)]
            total_k_sec = sum(k_durations) / 100.0
            line_dur_sec = en_sec - st_sec
            assert total_k_sec <= line_dur_sec + 0.02, f"Truncation: k={total_k_sec} > line={line_dur_sec}"


def test_number_expansion_preserves_single_display_token():
    """Verify numeric tokens expand to spoken phonetics while preserving single display token byte-for-byte."""
    from src.align.number_words import expand_token_numbers
    from src.align.mms_fa import clean_phonetic_token, romanized_to_ids, get_mms_fa

    # Test Spanish expansions
    assert expand_token_numbers("48", "es") == "cuarenta y ocho"
    assert expand_token_numbers("38", "es") == "treinta y ocho"
    assert expand_token_numbers("4", "es") == "cuatro"
    assert expand_token_numbers("2", "es") == "dos"
    assert expand_token_numbers("9C", "es") == "nueve ce"

    # Test English expansions
    assert expand_token_numbers("48", "en") == "forty eight"
    assert expand_token_numbers("38", "en") == "thirty eight"
    assert expand_token_numbers("9C", "en") == "nine c"

    # Verify MMS_FA clean phonetic token and dictionary mapping
    _, mms_dict, _, _ = get_mms_fa()
    cl_es = clean_phonetic_token("48", lang="es")
    assert cl_es == "cuarenta y ocho"
    ids_es = romanized_to_ids(cl_es, mms_dict)
    # Check that all decoded characters are alphabetical Spanish phonetics
    decoded = "".join([k for i in ids_es for k, v in mms_dict.items() if v == i])
    assert decoded == "cuarentayocho"

    # Verify single display token invariant in flat word list
    flat_word_tokens = ["Sector", "libertad", "9C", "y", "48", "4", "2", "Josefa", "y", "38"]
    roman = [clean_phonetic_token(tok, lang="es") for tok in flat_word_tokens]
    assert len(flat_word_tokens) == len(roman) == 10
    assert flat_word_tokens[4] == "48"  # Byte-for-byte exact display token
    assert roman[4] == "cuarenta y ocho"  # Expanded phonetic alignment key


def test_post_line_readability_hold_and_rapid_clamp():
    """Verify post-line readability hold extends before open silence and clamps with zero overlap on rapid lines."""
    import tempfile, re
    from src.subtitles.ass_writer import write_ass

    # Scenario:
    # Line 0: 1.00s .. 3.00s, followed by open silence (next line at 8.00s) -> holds +0.25s to 3.25s
    # Line 1: 8.00s .. 10.00s, followed by rapid speech (next line at 10.08s) -> clamped to 10.04s
    # Line 2: 10.08s .. 12.00s (last line) -> holds +0.25s to 12.25s
    words = [
        # Line 0
        {"line_index": 0, "word": "first", "start_sec": 1.00, "end_sec": 1.80, "duration_cs": 80},
        {"line_index": 0, "word": "line", "start_sec": 2.00, "end_sec": 3.00, "duration_cs": 100},
        # Line 1
        {"line_index": 1, "word": "second", "start_sec": 8.00, "end_sec": 8.80, "duration_cs": 80},
        {"line_index": 1, "word": "line", "start_sec": 9.00, "end_sec": 10.00, "duration_cs": 100},
        # Line 2 (rapid successor starting at 10.08s)
        {"line_index": 2, "word": "rapid", "start_sec": 10.08, "end_sec": 11.00, "duration_cs": 92},
        {"line_index": 2, "word": "end", "start_sec": 11.20, "end_sec": 12.00, "duration_cs": 80},
    ]

    def parse_t(s):
        h, m, sec_cs = s.split(":")
        sec, cs = sec_cs.split(".")
        return int(h) * 3600 + int(m) * 60 + int(sec) + int(cs) / 100.0

    with tempfile.TemporaryDirectory() as td:
        ass_file = Path(td) / "test_hold.ass"
        write_ass(words, ass_file)

        dialogues = [l for l in ass_file.read_text(encoding="utf-8").splitlines() if l.startswith("Dialogue:")]
        assert len(dialogues) == 3

        # Parse timings
        t0_st, t0_en = parse_t(dialogues[0].split(",", 9)[1]), parse_t(dialogues[0].split(",", 9)[2])
        t1_st, t1_en = parse_t(dialogues[1].split(",", 9)[1]), parse_t(dialogues[1].split(",", 9)[2])
        t2_st, t2_en = parse_t(dialogues[2].split(",", 9)[1]), parse_t(dialogues[2].split(",", 9)[2])

        # Line 0: open space -> extended +0.25s
        assert abs(t0_en - 3.25) <= 0.02, f"Line 0 expected ~3.25s, got {t0_en}"

        # Line 1: rapid next line at 10.08 -> strictly clamped to <= 10.04s
        assert t1_en <= 10.04 + 0.01, f"Line 1 exceeded clamp: {t1_en} > 10.04"
        assert t1_en < t2_st, f"Overlap detected: line 1 end {t1_en} >= line 2 start {t2_st}"

        # Line 2: last line -> extended +0.25s
        assert abs(t2_en - 12.25) <= 0.02, f"Line 2 expected ~12.25s, got {t2_en}"


def test_intra_line_silence_penalty_vs_active_vocal_stutter():
    """Verify intra-line silence penalty anchors same-line words across silence while preserving active-vocal stutter absorption."""
    from src.align_v2.ctc_engine import align_global_words

    # Setup: 2 words on the SAME lyric line (line_indices=[0, 0])
    # T = 150 frames (3.0s).
    # Word 0: [1] at frames 10-20.
    # Word 1: [2] has vocal burst at frames 25-35 (-1.0 lp), but a noise spike at frames 100-110 (-0.8 lp).
    # Frames 40-95 are silent: active_vocal_mask = False.
    T = 150
    vocab_size = 5
    blank_id = 0
    star_id = 4
    lp = np.full((T, vocab_size), -4.0, dtype=np.float32)
    lp[:, blank_id] = -0.5
    lp[:, star_id] = -1.85

    # Word 0 at frames 10-20
    lp[10:20, 1] = -0.2
    # Word 1 true vocal at frames 25-35
    lp[25:35, 2] = -0.8
    # Noise burst at frames 100-110 (slightly higher lp if unpenalized)
    lp[100:110, 2] = -0.6

    active_mask = np.zeros(T, dtype=bool)
    active_mask[8:38] = True  # Vocal active during phrase 1
    active_mask[98:115] = True  # Vocal/noise active during phrase 2

    # Case A: WITH intra-line silence penalty (silence across frames 40-95 costs ~6.6)
    posts_penalized = align_global_words(
        torch.from_numpy(lp),
        [[1], [2]],
        blank_id=blank_id,
        star_id=star_id,
        active_vocal_mask=active_mask,
        line_indices=[0, 0],
        intra_line_silence_penalty=0.12,
    )
    # Word 1 MUST stay attached to phrase 1 at frames 25-35 (start_frame < 40)
    assert posts_penalized[1].start_frame < 40, f"Word 1 detached across silence: start_frame={posts_penalized[1].start_frame}"

    # Case B: ACTIVE VOCAL STUTTER (active_vocal_mask is True throughout frames 40-95)
    # Stutter on same line (e.g. dar-dar-dar -> gracias) must NOT be penalized
    active_stutter_mask = np.ones(T, dtype=bool)
    posts_stutter = align_global_words(
        torch.from_numpy(lp),
        [[1], [2]],
        blank_id=blank_id,
        star_id=star_id,
        active_vocal_mask=active_stutter_mask,
        line_indices=[0, 0],
        intra_line_silence_penalty=0.12,
    )
    # Under active stutter without silence, Word 1 can freely reach the louder target at frame 100
    assert posts_stutter[1].start_frame >= 95, f"Word 1 failed to absorb active stutter: start_frame={posts_stutter[1].start_frame}"


def test_spanish_ll_mapping_language_boundary():
    """Verify Spanish ll -> y triggers ONLY when lang=='es' and NEVER alters English words."""
    from src.align.mms_fa import clean_phonetic_token

    # Spanish: ll -> y
    assert clean_phonetic_token("llena", lang="es") == "yena"
    assert clean_phonetic_token("lleno", lang="es") == "yeno"
    assert clean_phonetic_token("calle", lang="es") == "caye"
    assert clean_phonetic_token("llegar", lang="es") == "yegar"

    # English: ll stays intact
    assert clean_phonetic_token("still", lang="en") == "still"
    assert clean_phonetic_token("hello", lang="en") == "hello"
    assert clean_phonetic_token("all", lang="en") == "all"
    assert clean_phonetic_token("yellow", lang="en") == "yellow"

    # None / Default: ll stays intact
    assert clean_phonetic_token("still", lang=None) == "still"
    assert clean_phonetic_token("hello", lang=None) == "hello"
    assert clean_phonetic_token("all", lang=None) == "all"


def test_cli_engine_routing_and_defaults():
    """Verify CLI engine routing: default resolves to v5, --engine v2 aliases to v5, --engine v1 routes to v1."""
    import argparse
    from src.align_v2.schema import ENGINE_V5, ENGINE_VERSION

    assert ENGINE_V5 == "greenfield_v5_dual_ctc_consensus"
    assert ENGINE_VERSION == "5.0"

    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["v5", "v2", "v1"], default="v5")

    # Case 1: Default (no --engine flag passed)
    args_default = ap.parse_args([])
    assert args_default.engine == "v5"
    engine_resolved_default = "v5" if args_default.engine in ("v5", "v2") else "v1"
    assert engine_resolved_default == "v5"

    # Case 2: Explicit --engine v2 (backwards-compatible alias)
    args_v2 = ap.parse_args(["--engine", "v2"])
    assert args_v2.engine == "v2"
    engine_resolved_v2 = "v5" if args_v2.engine in ("v5", "v2") else "v1"
    assert engine_resolved_v2 == "v5"

    # Case 3: Legacy --engine v1
    args_v1 = ap.parse_args(["--engine", "v1"])
    assert args_v1.engine == "v1"
    engine_resolved_v1 = "v5" if args_v1.engine in ("v5", "v2") else "v1"
    assert engine_resolved_v1 == "v1"


def test_cli_stem_flags_resolution():
    """Verify --hq-stems (default) resolves to BS-Roformer CPU overlap=2 and --fast-stems resolves to Kim_Vocal_2 DirectML."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--hq-stems", action="store_true", default=False)
    ap.add_argument("--fast-stems", action="store_true", default=False)

    def resolve_stem_config(argv):
        args = ap.parse_args(argv)
        if args.fast_stems:
            stem_model = "Kim_Vocal_2.onnx"
            use_directml = True
            backend = "kim_vocal_2_dml"
        else:
            stem_model = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
            use_directml = False
            backend = "bs_roformer_cpu"
        return stem_model, use_directml, backend

    # Case 1: Default (no flags)
    m, dml, b = resolve_stem_config([])
    assert m == "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    assert dml is False
    assert b == "bs_roformer_cpu"

    # Case 2: Explicit --hq-stems
    m, dml, b = resolve_stem_config(["--hq-stems"])
    assert m == "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    assert dml is False
    assert b == "bs_roformer_cpu"

    # Case 3: Explicit --fast-stems
    m, dml, b = resolve_stem_config(["--fast-stems"])
    assert m == "Kim_Vocal_2.onnx"
    assert dml is True
    assert b == "kim_vocal_2_dml"


def test_hardware_vendor_classification_and_profiles():
    """Verify multi-vendor GPU classification and CPU vs GPU ComputeProfile selection."""
    from src.core.hardware import GpuDevice, HardwareReport, build_compute_profile, classify_gpu_vendor

    assert classify_gpu_vendor("NVIDIA GeForce RTX 4090") == "nvidia"
    assert classify_gpu_vendor("AMD Radeon RX 7900 XTX") == "amd"
    assert classify_gpu_vendor("Intel(R) Arc(TM) Pro B70 Graphics") == "intel"
    assert classify_gpu_vendor("Apple M3 Max GPU") == "apple"

    # 1. NVIDIA CUDA host
    nvidia_rep = HardwareReport(
        os_name="windows", os_label="Windows 11", arch="AMD64", hostname="test",
        cpu_name="Ryzen", cpu_cores=8, cpu_threads=16, ram_gb=32.0,
        gpus=[GpuDevice(vendor="nvidia", name="NVIDIA GeForce RTX 4080", vram_gb=16.0)],
        torch_cuda=True, ort_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        ctranslate2_cuda=True, working_hw_encoders=["libx264", "h264_nvenc"],
    )
    p_nv = build_compute_profile(nvidia_rep, mode="gpu")
    assert p_nv.profile_id == "nvidia_cuda"
    assert p_nv.bs_roformer_on_gpu is True
    assert p_nv.align_device == "cuda"
    assert p_nv.whisper_device == "cuda"
    assert p_nv.video_codec == "h264_nvenc"

    # CPU-only mode on same NVIDIA host must force CPU everywhere
    p_nv_cpu = build_compute_profile(nvidia_rep, mode="cpu")
    assert p_nv_cpu.profile_id == "cpu_only"
    assert p_nv_cpu.align_device == "cpu"
    assert p_nv_cpu.video_codec == "libx264"
    assert p_nv_cpu.bs_roformer_on_gpu is False

    # 2. Windows AMD Radeon DirectML host
    amd_win_rep = HardwareReport(
        os_name="windows", os_label="Windows 11", arch="AMD64", hostname="test",
        cpu_name="Ryzen", cpu_cores=8, cpu_threads=16, ram_gb=32.0,
        gpus=[GpuDevice(vendor="amd", name="AMD Radeon RX 7800 XT", vram_gb=16.0)],
        ort_providers=["DmlExecutionProvider", "CPUExecutionProvider"],
        working_hw_encoders=["libx264", "h264_amf"],
    )
    p_amd = build_compute_profile(amd_win_rep, mode="gpu")
    assert p_amd.profile_id == "amd_directml"
    assert p_amd.stem_mode == "fast"
    assert p_amd.stem_device == "dml"
    assert p_amd.video_codec == "h264_amf"

    # 3. Intel Arc Windows DirectML host (filtering virtual monitor first)
    intel_rep = HardwareReport(
        os_name="windows", os_label="Windows 11", arch="AMD64", hostname="test",
        cpu_name="Ryzen", cpu_cores=16, cpu_threads=32, ram_gb=31.4,
        gpus=[
            GpuDevice(vendor="other", name="Virtual Desktop Monitor", is_virtual=True),
            GpuDevice(vendor="intel", name="Intel(R) Arc(TM) Pro B70 Graphics", vram_gb=32.0),
        ],
        ort_providers=["DmlExecutionProvider", "CPUExecutionProvider"],
        working_hw_encoders=["libx264", "h264_qsv"],
    )
    p_int = build_compute_profile(intel_rep, mode="gpu")
    assert p_int.profile_id == "intel_arc_dml"
    assert p_int.stem_device == "dml"
    assert p_int.video_codec == "h264_qsv"

    # 4. Apple Silicon MPS host
    mac_rep = HardwareReport(
        os_name="macos", os_label="macOS 15", arch="arm64", hostname="mac",
        cpu_name="Apple M3 Max", cpu_cores=14, cpu_threads=14, ram_gb=36.0,
        gpus=[GpuDevice(vendor="apple", name="Apple Silicon Unified GPU")],
        torch_mps=True, ort_providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        working_hw_encoders=["libx264", "h264_videotoolbox"],
    )
    p_mac = build_compute_profile(mac_rep, mode="gpu")
    assert p_mac.profile_id == "apple_mps"
    assert p_mac.bs_roformer_on_gpu is True
    assert p_mac.align_device == "mps"
    assert p_mac.video_codec == "h264_videotoolbox"


def test_vendor_encoder_args_per_codec():
    """Verify each FFmpeg hardware encoder receives only valid codec-specific flags."""
    from src.pipeline.render import _build_video_encoder_args

    qsv = _build_video_encoder_args("h264_qsv", 22)
    assert "-global_quality" in qsv and "-crf" not in qsv

    nv = _build_video_encoder_args("h264_nvenc", 22)
    assert "-cq" in nv and "-crf" not in nv

    amf = _build_video_encoder_args("h264_amf", 22)
    assert "-qp_i" in amf and "-preset" not in amf and "-crf" not in amf

    vt = _build_video_encoder_args("h264_videotoolbox", 22)
    assert "-q:v" in vt and "-crf" not in vt and "-preset" not in vt

    sw = _build_video_encoder_args("libx264", 22)
    assert "-crf" in sw and "-preset" in sw


def test_requirements_and_config_universality():
    """Verify requirements.txt has all core packages and config.json uses universal defaults."""
    import json

    root = Path(__file__).resolve().parents[1] / "DO NOT TOUCH"
    req_txt = (root / "requirements.txt").read_text(encoding="utf-8")
    for pkg in ("torch", "torchaudio", "faster-whisper", "audio-separator", "transformers", "librosa", "scipy", "soundfile"):
        assert pkg in req_txt, f"Missing core requirement: {pkg}"

    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    assert cfg.get("codec") == "libx264"
    assert cfg.get("lyric_sync_engine") == "greenfield_v5_dual_ctc_consensus"


def test_dynamic_cps_readability_hold_model():
    """Verify Characters-Per-Second (CPS) model dynamically scales hold time by character count."""
    import tempfile
    from src.subtitles.ass_writer import write_ass

    def parse_t(s):
        h, m, sec_cs = s.split(":")
        sec, cs = sec_cs.split(".")
        return int(h) * 3600 + int(m) * 60 + int(sec) + int(cs) / 100.0

    # Short line (5 chars): sung in 0.2s -> floor 0.90s
    # Long line (40 chars): sung in 0.5s -> scaled ~2.75s
    # Rapid successor line: clamps without collision
    words = [
        {"line_index": 0, "word": "If", "start_sec": 1.00, "end_sec": 1.10, "duration_cs": 10},
        {"line_index": 0, "word": "you", "start_sec": 1.10, "end_sec": 1.20, "duration_cs": 10},

        {"line_index": 1, "word": "What", "start_sec": 5.00, "end_sec": 5.10, "duration_cs": 10},
        {"line_index": 1, "word": "you", "start_sec": 5.10, "end_sec": 5.20, "duration_cs": 10},
        {"line_index": 1, "word": "give", "start_sec": 5.20, "end_sec": 5.30, "duration_cs": 10},
        {"line_index": 1, "word": "what", "start_sec": 5.30, "end_sec": 5.40, "duration_cs": 10},
        {"line_index": 1, "word": "you", "start_sec": 5.40, "end_sec": 5.50, "duration_cs": 10},
        {"line_index": 1, "word": "give", "start_sec": 5.50, "end_sec": 5.60, "duration_cs": 10},
        {"line_index": 1, "word": "comes", "start_sec": 5.60, "end_sec": 5.70, "duration_cs": 10},
        {"line_index": 1, "word": "back", "start_sec": 5.70, "end_sec": 5.80, "duration_cs": 10},
        {"line_index": 1, "word": "fast", "start_sec": 5.80, "end_sec": 5.90, "duration_cs": 10},

        # Successor line close behind (at 9.00s)
        {"line_index": 2, "word": "End", "start_sec": 9.00, "end_sec": 9.50, "duration_cs": 50},
    ]

    with tempfile.TemporaryDirectory() as td:
        ass_file = Path(td) / "test_cps.ass"
        write_ass(words, ass_file)
        dialogues = [l for l in ass_file.read_text(encoding="utf-8").splitlines() if l.startswith("Dialogue:")]
        assert len(dialogues) == 3

        t0_st, t0_en = parse_t(dialogues[0].split(",", 9)[1]), parse_t(dialogues[0].split(",", 9)[2])
        t1_st, t1_en = parse_t(dialogues[1].split(",", 9)[1]), parse_t(dialogues[1].split(",", 9)[2])
        t2_st, t2_en = parse_t(dialogues[2].split(",", 9)[1]), parse_t(dialogues[2].split(",", 9)[2])

        # Short line: 5 chars / 14.5 = 0.34s -> clamped to floor 0.90s -> end ~1.90s
        assert abs((t0_en - t0_st) - 0.90) <= 0.05, f"Short line hold expected ~0.90s, got {t0_en - t0_st}"

        # Long line: 36 chars / 14.5 = 2.48s -> end >= 5.0 + 2.48 = 7.48s, clamped before 9.00s
        assert (t1_en - t1_st) >= 2.0, f"Long line hold expected >= 2.0s, got {t1_en - t1_st}"
        assert t1_en <= 8.96 + 0.01, f"Long line collided with successor: {t1_en} > 8.96"


def test_multilingual_script_and_latin_ranker():
    """Verify universal language detection across CJK, Cyrillic, Arabic, and major Latin languages."""
    from src.align_v2.lang_detect import detect_text_language, resolve_language_consensus

    # 1. Non-Latin Unicode script detection
    ja_lang, ja_conf = detect_text_language(["こんにちは", "世界", "さようなら"])
    assert ja_lang == "ja" and ja_conf >= 0.90

    ru_lang, ru_conf = detect_text_language(["Привет", "мир", "как", "дела"])
    assert ru_lang == "ru" and ru_conf >= 0.90

    ar_lang, ar_conf = detect_text_language(["مرحبا", "بالعالم", "كيف", "حالك"])
    assert ar_lang == "ar" and ar_conf >= 0.90

    # 2. Latin multilingual profiler
    en_lang, en_conf = detect_text_language(["if", "you", "want", "love", "come", "in", "with", "your", "guard", "down"])
    assert en_lang == "en" and en_conf >= 0.70

    es_lang, es_conf = detect_text_language(["sector", "libertad", "y", "la", "bolsa", "siempre", "bien", "llena"])
    assert es_lang == "es" and es_conf >= 0.70

    fr_lang, fr_conf = detect_text_language(["la", "vie", "en", "rose", "avec", "toi", "mon", "amour"])
    assert fr_lang == "fr" and fr_conf >= 0.70

    de_lang, de_conf = detect_text_language(["ich", "liebe", "dich", "und", "du", "bist", "mein", "leben"])
    assert de_lang == "de" and de_conf >= 0.70

    pt_lang, pt_conf = detect_text_language(["você", "é", "meu", "amor", "e", "minha", "vida", "onde", "está"])
    assert pt_lang == "pt" and pt_conf >= 0.70

    # 3. Bayesian consensus resolution
    # Confident text overrides ambiguous audio
    assert resolve_language_consensus("fr", 0.85, "en", 0.30) == "fr"
    # Unanimous agreement
    assert resolve_language_consensus("es", 0.90, "es", 0.95) == "es"
    # Ambiguous text yields to confident audio
    assert resolve_language_consensus("en", 0.20, "pt", 0.88) == "pt"
    # Explicit override always wins
    assert resolve_language_consensus("en", 0.99, "en", 0.99, explicit_lang="de") == "de"


def test_otsu_bimodal_vad_energy_thresholding():
    """Verify Otsu bimodal thresholding cleanly separates speech harmonics from background bleed floor."""
    from src.align.mms_fa import compute_vocal_activity_mask

    sr = 16000
    dur_sec = 4.0
    t = np.linspace(0, dur_sec, int(sr * dur_sec), endpoint=False)

    # Synthetic signal: 1s silence/bleed (RMS ~0.005) + 2s vocal burst (RMS ~0.15) + 1s bleed
    signal = np.zeros_like(t, dtype=np.float32)
    # Bleed floor (low amplitude noise)
    signal += 0.005 * np.sin(2 * np.pi * 100 * t).astype(np.float32)
    # Vocal burst between 1.0s and 3.0s (high amplitude harmonics)
    vocal_window = (t >= 1.0) & (t <= 3.0)
    signal[vocal_window] += 0.25 * np.sin(2 * np.pi * 440 * t[vocal_window]).astype(np.float32)

    active_mask, info = compute_vocal_activity_mask(signal, sr=sr, hop_sec=0.02)
    assert len(active_mask) > 0
    # Vocal section should be active
    hop_sec = 0.02
    vocal_frames = active_mask[int(1.2 / hop_sec) : int(2.8 / hop_sec)]
    assert np.mean(vocal_frames) >= 0.90, "Vocal burst frames should be >= 90% active"
    # Silent margins should be mostly inactive
    head_frames = active_mask[: int(0.8 / hop_sec)]
    assert np.mean(head_frames) <= 0.15, "Head bleed frames should be <= 15% active"


def test_self_calibrated_cadence_and_vad_clamps():
    """Verify interpolation measures empirical words-per-second and clamps to VAD vocal bounds."""
    from src.align_v2.pipeline import interpolate_anchors_with_cadence_and_vad

    # Setup: 10 words. Audio duration 10.0s.
    # Active vocal energy exists strictly between frame 50 (1.0s) and frame 450 (9.0s).
    spf = 0.02
    total_frames = 500
    active_mask = np.zeros(total_frames, dtype=bool)
    active_mask[50:450] = True  # 1.0s .. 9.0s

    # Anchors at index 3 (2.0s) and index 5 (2.4s) -> Measured cadence: (2.4 - 2.0) / 2 = 0.20s/word (fast speech)
    word_times = [None] * 10
    word_times[3] = 2.0
    word_times[5] = 2.4

    interp = interpolate_anchors_with_cadence_and_vad(
        word_count=10,
        word_times=word_times,
        active_mask=active_mask,
        dur=10.0,
        spf=spf,
    )

    assert len(interp) == 10
    # 1. Monotonicity
    for k in range(len(interp) - 1):
        assert interp[k] <= interp[k + 1], f"Non-monotonic interpolation at index {k}: {interp[k]} > {interp[k+1]}"

    # 2. Hard VAD clamp on head: word 0 must not start before first active vocal frame (1.0s)
    # Head extrapolation: 2.0s - 3 * 0.20s = 1.4s (well within vocal boundary)
    assert interp[0] >= 1.0, f"Head leaked before first vocal burst: {interp[0]} < 1.0"
    assert abs(interp[0] - 1.4) <= 0.05, f"Expected cadence-interpolated head ~1.4s, got {interp[0]}"

    # 3. Known anchors preserved
    assert abs(interp[3] - 2.0) < 1e-4
    assert abs(interp[5] - 2.4) < 1e-4

    # 4. Tail clamped <= t_last_vocal (9.0s)
    assert interp[9] <= 9.0 + 1e-4, f"Tail exceeded last vocal burst: {interp[9]} > 9.0"









