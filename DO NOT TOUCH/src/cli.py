"""CLI entry point for the greenfield universal alignment engine (v5.3).

Usage:
    python -m src.cli --project "START HERE/_Projects/PaoPao - Grax" --align-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Ensure DO NOT TOUCH root is in sys.path
_dnt_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _dnt_root not in sys.path:
    sys.path.insert(0, _dnt_root)

import numpy as np

from src.align.integrity_verifier import IntegrityVerifier
from src.align.pipeline import UniversalEngine
from src.audio.streams import decode_audio_mono

# Banned generated artifacts that must never be used as the raw lyric source.
_LYRIC_EXCLUDE = {"lyrics_easy.txt", "lyrics_synced.txt", "readme.txt"}
_LYRIC_PREFER = ("paopao_grax.txt",)
_AUDIO_EXTS = (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac")


def _find_sources(project_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """Locate the canonical read-only lyric source and audio inside `project_dir`."""
    lyric: Optional[str] = None
    audio: Optional[str] = None
    if not os.path.isdir(project_dir):
        return None, None

    entries = sorted(os.listdir(project_dir))
    for pref in _LYRIC_PREFER:
        for fn in entries:
            if fn.lower() == pref:
                cand = os.path.join(project_dir, fn)
                if os.path.isfile(cand):
                    lyric = cand
                    break
        if lyric:
            break

    if lyric is None:
        for fn in entries:
            low = fn.lower()
            if low.endswith(".txt") and low not in _LYRIC_EXCLUDE:
                cand = os.path.join(project_dir, fn)
                if os.path.isfile(cand):
                    lyric = cand
                    break

    for fn in entries:
        low = fn.lower()
        if any(low.endswith(ext) for ext in _AUDIO_EXTS):
            cand = os.path.join(project_dir, fn)
            if os.path.isfile(cand):
                audio = cand
                break

    return lyric, audio


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        from src.cli_interactive import main as interactive_main

        return interactive_main()

    ap = argparse.ArgumentParser(
        description="TikTok Wizard - greenfield universal engine v5.3"
    )
    ap.add_argument("--project", required=True, help="Project dir (read-only source)")
    ap.add_argument("--align-only", action="store_true", help="Run alignment only")
    ap.add_argument("--tau-song", type=float, default=0.23)
    ap.add_argument("--word-target", type=float, default=0.95)
    ap.add_argument("--whisper-size", default="small")
    ap.add_argument(
        "--out",
        default=None,
        help="Output dir (default: DO NOT TOUCH/data/<project name>)",
    )
    ap.add_argument("--log", default=None, help="Append a UTF-8 execution log here")
    args = ap.parse_args(argv)

    project_dir = os.path.abspath(args.project)
    project_name = os.path.basename(project_dir.rstrip("\\/"))

    here = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # DO NOT TOUCH
    root = os.path.abspath(os.path.join(here, ".."))  # repo root
    out_dir = os.path.abspath(args.out) if args.out else os.path.join(here, "data", project_name)

    log_path = args.log
    logf = None
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        logf = open(log_path, "a", encoding="utf-8")

    def log(msg: object) -> None:
        line = str(msg)
        print(line, flush=True)
        if logf:
            logf.write(line + "\n")
            logf.flush()

    log("=" * 64)
    log("TikTok Wizard - greenfield universal engine v5.3 (real MMS_FA)")
    log(f"project={project_dir}")
    log(f"out={out_dir}")

    lyric, audio = _find_sources(project_dir)
    log(f"lyric={lyric}")
    log(f"audio={audio}")
    if not lyric or not audio:
        log("ERROR: lyric or audio not found")
        if logf:
            logf.close()
        return 2

    engine = UniversalEngine(
        tau_song=args.tau_song,
        word_target=args.word_target,
        whisper_size=args.whisper_size,
    )
    manifest = engine.run(lyric, audio, out_dir, log=log)

    mono, sr = decode_audio_mono(
        os.path.join(out_dir, "stems", "original_mix_vocal_band.wav"), 16000
    )
    verifier = IntegrityVerifier(word_target=args.word_target)
    gate = verifier.anti_fabrication_gate(
        manifest["words"],
        manifest["total_duration_sec"],
        mono,
        sr,
        manifest["lines"],
        root,
    )

    with open(os.path.join(out_dir, "lyrics_synced.txt"), "r", encoding="utf-8") as f:
        synced = [ln for ln in f.read().splitlines() if ln.strip() and not ln.strip().startswith("#")]
    with open(os.path.join(out_dir, "karaoke.ass"), "r", encoding="utf-8") as f:
        ass_lines = [l for l in f.read().splitlines() if l.startswith("Dialogue:")]

    sw = sorted(manifest["words"], key=lambda x: float(x["start_sec"]))
    idur = [float(w["end_sec"]) - float(w["start_sec"]) for w in sw]
    gaps = [
        float(sw[k + 1]["start_sec"]) - float(sw[k]["end_sec"])
        for k in range(len(sw) - 1)
    ]
    confs = [float(w["confidence"]) for w in manifest["words"]]

    def line_at(n: int) -> Optional[str]:
        return synced[n - 1] if 0 < n <= len(synced) else None

    w31 = next((w for w in manifest["lines"] if int(w["index"]) == 31), None)
    w33 = next((w for w in manifest["lines"] if int(w["index"]) == 33), None)

    line_31_txt = line_at(31) or ""
    utf8_ok = ("enseñarme" in line_31_txt) and ("enseÃ±arme" not in line_31_txt)

    report = {
        "run_steps_3_4_stub_deleted": bool(gate["checks"]["no_stub_script"]["pass"]),
        "alignment_v5_py_deleted_and_absent": not os.path.exists(
            os.path.join(here, "src", "pipeline", "alignment_v5.py")
        ),
        "venv_python_executable_used": sys.executable,
        "utf8_ensenarme_preserved_exactly": bool(utf8_ok),
        "true_audio_duration_sec": round(float(manifest["total_duration_sec"]), 3),
        "inter_word_gap_std_sec": round(float(np.std(gaps)) if gaps else 0.0, 5),
        "confidence_std": round(float(np.std(confs)) if confs else 0.0, 6),
        "vocal_vs_silence_rms_ratio": gate["checks"]["acoustic_snr"]["ratio"],
        "line_31_start_end_sec": [round(float(w31["start_sec"]), 3), round(float(w31["end_sec"]), 3)]
        if w31
        else None,
        "line_33_start_end_sec": [round(float(w33["start_sec"]), 3), round(float(w33["end_sec"]), 3)]
        if w33
        else None,
        "pytest_status": "n/a (run separately)",
        "total_input_lines": int(manifest["total_input_lines"]),
        "total_output_ass_lines": len(ass_lines),
        "total_output_synced_txt_lines": len(synced),
        "total_words": int(manifest["total_words"]),
        "cemented_words_ge_95_count": int(manifest["cemented_words_ge_95_count"]),
        "uncemented_words_count": int(manifest["uncemented_words_count"]),
        "unique_word_durations": len(set(round(d, 2) for d in idur)),
        "gate_overall_pass": bool(gate["overall_pass"]),
        "gate_checks": gate["checks"],
        "lines_17_22_and_28_to_36_from_lyrics_synced_txt": {
            str(n): line_at(n) for n in [17, 22, 28, 29, 30, 31, 32, 33, 34, 35, 36]
        },
    }
    rpath = os.path.join(out_dir, "verification_report.json")
    with open(rpath, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log(f"wrote {rpath}")
    log(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "true_audio_duration_sec",
                    "inter_word_gap_std_sec",
                    "confidence_std",
                    "vocal_vs_silence_rms_ratio",
                    "utf8_ensenarme_preserved_exactly",
                    "line_31_start_end_sec",
                    "line_33_start_end_sec",
                    "cemented_words_ge_95_count",
                    "uncemented_words_count",
                    "gate_overall_pass",
                ]
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if logf:
        logf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
