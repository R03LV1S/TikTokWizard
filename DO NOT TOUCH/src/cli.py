"""CLI entry point for the greenfield universal alignment engine (v5.7).

Usage:
    python -m src.cli --project "START HERE/_Projects/PaoPao - Grax" --align-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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
from src.version import get_app_version, get_engine_info, get_app_header

# Banned generated artifacts that must never be used as the raw lyric source.
_LYRIC_EXCLUDE = {"lyrics_easy.txt", "lyrics_synced.txt", "readme.txt"}
_LYRIC_PREFER = ("paopao_grax.txt",)
_AUDIO_EXTS = (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac")


def _is_manifest_or_rendered_companion(filepath: str) -> bool:
    """Return True if filepath is a video render manifest or companion log."""
    base, _ = os.path.splitext(filepath)
    if os.path.isfile(base + ".mp4") or os.path.isfile(base + ".mov") or os.path.isfile(base + ".mkv"):
        return True
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(512)
            if "TIKTOK WIZARD VIDEO RENDER" in head or "SETTINGS MANIFEST" in head:
                return True
    except Exception:
        pass
    return False


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
                if os.path.isfile(cand) and not _is_manifest_or_rendered_companion(cand):
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
        description=f"TikTok Wizard v{get_app_version()} - {get_engine_info()[2]}"
    )
    ap.add_argument("--project", required=True, help="Project dir (read-only source)")
    ap.add_argument("--lyric", "--lyrics", default=None, help="Explicit lyric file path")
    ap.add_argument("--align-only", action="store_true", help="Run alignment only")
    ap.add_argument("--tau-song", type=float, default=0.23)
    ap.add_argument("--word-target", type=float, default=0.95)
    ap.add_argument("--whisper-size", default="small")
    ap.add_argument(
        "--multipass",
        type=int,
        default=1,
        help="Clean-scratch passes for consensus (robust mode)",
    )
    ap.add_argument("--robust", action="store_true", help="Alias for --multipass 3")
    ap.add_argument(
        "--out",
        default=None,
        help="Output dir (default: DO NOT TOUCH/data/<project name>)",
    )
    ap.add_argument(
        "--lang", default=None, help="ASR language hint (default: auto-detect)"
    )
    ap.add_argument(
        "--engine",
        choices=["v5", "v2", "v1"],
        default="v5",
        help="Alignment engine: v5 (default, dual-model consensus), v2 (alias to v5), or v1 (legacy)",
    )
    ap.add_argument(
        "--hq-stems",
        action="store_true",
        default=False,
        help="Use high-quality BS-Roformer CPU stem separation (overlap=2, default)",
    )
    ap.add_argument(
        "--fast-stems",
        action="store_true",
        default=False,
        help="Use fast Kim_Vocal_2 DirectML GPU stem separation (~25s on Intel Arc)",
    )
    ap.add_argument(
        "--compute",
        choices=["auto", "gpu", "cpu"],
        default=None,
        help="Compute backend mode override (auto, gpu, or cpu)",
    )
    ap.add_argument("--log", default=None, help="Append a UTF-8 execution log here")
    args = ap.parse_args(argv)
    if args.compute:
        os.environ["TW_COMPUTE_MODE"] = args.compute
    if args.robust and args.multipass <= 1:
        args.multipass = 3

    project_dir = os.path.abspath(args.project)
    project_name = os.path.basename(project_dir.rstrip("\\/"))

    here = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )  # DO NOT TOUCH
    root = os.path.abspath(os.path.join(here, ".."))  # repo root
    out_dir = (
        os.path.abspath(args.out)
        if args.out
        else os.path.join(here, "data", project_name)
    )

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

    # Resolve stems model
    if args.fast_stems:
        stem_model = "Kim_Vocal_2.onnx"
        stem_mode = "fast"
    else:
        # Default is HQ BS-Roformer CPU
        stem_model = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
        stem_mode = "hq"
    os.environ["TW_STEM_MODEL"] = stem_model

    # Engine routing: default v5 (v2 aliases cleanly to v5), v1 is legacy fallback
    engine_arg = (args.engine or "v5").lower()
    if engine_arg in ("v5", "v2"):
        active_engine = "v5"
        engine_suffix = ""  # canonical project directory DO NOT TOUCH\data\<Project>\
    else:
        active_engine = "v1"
        engine_suffix = ".v1"

    log("=" * 64)
    log(f"TikTok Wizard v{get_app_version()} - {get_engine_info()[2]}")
    log(f"project={project_dir}")
    # Engine-specific output subdir: v5 writes directly to canonical dir, v1 writes to .v1
    out_dir_engine = out_dir + engine_suffix
    if not args.out:
        out_dir_engine = os.path.join(
            os.path.dirname(out_dir), os.path.basename(out_dir) + engine_suffix
        )
    log(f"out={out_dir_engine}")
    log(f"engine={active_engine} (stems={stem_mode})")

    if args.lyric:
        lyric = os.path.abspath(args.lyric)
        _, audio = _find_sources(project_dir)
    else:
        lyric, audio = _find_sources(project_dir)
    log(f"lyric={lyric}")
    log(f"audio={audio}")
    if not lyric or not audio:
        log("ERROR: lyric or audio not found")
        if logf:
            logf.close()
        return 2

    if getattr(args, "multipass", 1) and int(args.multipass) > 1:
        if active_engine == "v5":
            log("ERROR: --multipass is only supported with --engine v1 in this slice")
            if logf:
                logf.close()
            return 2
        from src.align.multipass import run_multipass

        manifest = run_multipass(
            project=Path(project_dir),
            cli_cwd=Path(here),
            data_dir=Path(out_dir),
            python_exe=sys.executable,
            passes=int(args.multipass),
            log=log,
        )
        # write merged manifest and regenerate derived files
        os.makedirs(out_dir, exist_ok=True)
        with open(
            os.path.join(out_dir, "alignment_manifest.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        # regenerate ASS and lyrics_synced from merged manifest
        try:
            from src.pipeline.ass_writer import write_ass_karaoke
            from src.pipeline.schema import KaraokeEntry, KaraokeWord

            words = manifest.get("words", [])
            by = {}
            for w in words:
                li = int(w.get("line_index", 0))
                by.setdefault(li, []).append(w)
            entries = []
            for li in sorted(by.keys()):
                wl = sorted(by[li], key=lambda x: float(x.get("start_sec", 0)))
                kwords = []
                for w in wl:
                    st = float(w.get("start_sec", 0))
                    en = float(w.get("end_sec", st + 0.03))
                    dur_cs = max(1, int(round((en - st) * 100)))
                    kwords.append(
                        KaraokeWord(
                            text=str(w.get("word", "")),
                            duration_cs=dur_cs,
                            start=st,
                            end=en,
                        )
                    )
                if not kwords:
                    continue
                raw = " ".join(str(w.get("word", "")) for w in wl)
                entries.append(
                    KaraokeEntry(
                        line_index=li,
                        words=kwords,
                        start=float(kwords[0].start),
                        end=float(kwords[-1].end),
                        raw_text=raw,
                    )
                )
            prof_path = Path(here) / "profiles" / "default.json"
            prof = (
                json.loads(prof_path.read_text(encoding="utf-8"))
                if prof_path.exists()
                else {}
            )
            config = {
                "profiles": {"default": prof},
                "active_profile": "default",
                "karaoke": prof.get("karaoke", {}),
                "defaults_dir": str(
                    Path(here) / "START HERE" / "_Projects" / "_Assets"
                ),
            }
            effect_cfg = prof.get("effects", prof)
            write_ass_karaoke(
                entries,
                Path(out_dir) / "karaoke.ass",
                config,
                effect_cfg,
                1080,
                1920,
                defaults_dir=Path(here) / "START HERE" / "_Projects" / "_Assets",
            )
            try:
                from src.subtitles.lrc_exporter import export_lrc

                export_lrc(
                    manifest.get("words", []),
                    os.path.join(out_dir, "lyrics_synced.lrc"),
                    lines=[ln.get("text", "") for ln in manifest.get("lines", [])],
                )
                # also keep lyrics_synced.txt as LRC-compatible text for tooling
                lrc_path = Path(out_dir) / "lyrics_synced.lrc"
                txt_path = Path(out_dir) / "lyrics_synced.txt"
                if lrc_path.exists():
                    txt_path.write_text(
                        lrc_path.read_text(encoding="utf-8"), encoding="utf-8"
                    )
                else:
                    txt_path.write_text(
                        "# lyrics_synced.txt\n"
                        + "\n".join(e.raw_text for e in entries)
                        + "\n",
                        encoding="utf-8",
                    )
            except Exception:
                with open(
                    os.path.join(out_dir, "lyrics_synced.txt"), "w", encoding="utf-8"
                ) as f:
                    f.write("# lyrics_synced.txt\n")
                    for e in entries:
                        f.write(e.raw_text + "\n")
        except Exception as e:
            log(f"[multipass] warn: failed to regenerate ASS/lyrics_synced: {e}")
    else:
        if active_engine == "v5":
            from src.align_v2.pipeline import V5AlignmentEngine

            engine = V5AlignmentEngine(
                tau_song=args.tau_song,
                word_target=args.word_target,
                lang=getattr(args, "lang", None),
                stem_mode=stem_mode,
            )
            manifest = engine.run(lyric, audio, out_dir_engine, log=log)
        else:
            engine = UniversalEngine(
                tau_song=args.tau_song,
                word_target=args.word_target,
                whisper_size=args.whisper_size,
                primary_backend=os.environ.get("TW_PRIMARY_BACKEND", "mms_fa"),
                secondary_on_uncemented=os.environ.get("TW_SECONDARY_UNCEMENTED", "1")
                != "0",
                lang=getattr(args, "lang", None),
            )
            manifest = engine.run(lyric, audio, out_dir_engine, log=log)

    vb_path = os.path.join(out_dir_engine, "stems", "original_mix_vocal_band.wav")
    ac_path = os.path.join(out_dir_engine, "stems", "acapella_full.wav")
    ref_path = (
        vb_path
        if os.path.exists(vb_path)
        else (ac_path if os.path.exists(ac_path) else audio)
    )
    try:
        mono, sr = decode_audio_mono(ref_path, 16000)
    except Exception:
        mono, sr = np.zeros(16000, dtype=np.float32), 16000
    verifier = IntegrityVerifier(word_target=args.word_target)
    gate = verifier.anti_fabrication_gate(
        manifest.get("words", []),
        float(manifest.get("total_duration_sec", len(mono) / max(1, sr))),
        mono,
        sr,
        manifest.get("lines", []),
        root,
    )

    synced_path = os.path.join(out_dir_engine, "lyrics_synced.txt")
    if os.path.exists(synced_path):
        with open(synced_path, "r", encoding="utf-8") as f:
            synced = [
                ln
                for ln in f.read().splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            ]
    else:
        synced = [str(l.get("text", "")) for l in manifest.get("lines", [])]
    with open(os.path.join(out_dir_engine, "karaoke.ass"), "r", encoding="utf-8") as f:
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
        "line_31_start_end_sec": [
            round(float(w31["start_sec"]), 3),
            round(float(w31["end_sec"]), 3),
        ]
        if w31
        else None,
        "line_33_start_end_sec": [
            round(float(w33["start_sec"]), 3),
            round(float(w33["end_sec"]), 3),
        ]
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
    rpath = os.path.join(out_dir_engine, "verification_report.json")
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
