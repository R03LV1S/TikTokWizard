# align/multipass.py
"""Multipass clean-scratch alignment consensus.

Runs align-only N times (fresh data dir each time), merges best evidence.
Base = highest-cemented run; replace only unverified words with cemented
candidates from other passes when they fit local monotonic bounds.
No fake cementation; posterior floors enforced.
"""

from __future__ import annotations

import json, shutil, subprocess, sys, datetime
from pathlib import Path
from typing import Any, Dict, List


def _load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _save(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _cemented_count(m, target=0.95):
    return sum(
        1
        for w in m.get("words", [])
        if bool(w.get("is_cemented")) and float(w.get("confidence", 0)) >= target
    )


def _enforce_floors(w, target=0.95):
    post = float(w.get("posterior", 0) or 0)
    conf = float(w.get("confidence", 0) or 0)
    if post < 0.55:
        conf = min(conf, target - 0.05)
    elif post < 0.70:
        conf = min(conf, target - 0.01)
    w["confidence"] = round(conf, 6)
    w["is_cemented"] = bool(conf >= target)
    return w


def merge_manifests(
    manifests: List[Dict[str, Any]], target: float = 0.95
) -> Dict[str, Any]:
    manifests = sorted(
        manifests, key=lambda m: _cemented_count(m, target), reverse=True
    )
    base = json.loads(json.dumps(manifests[0]))
    words = base.get("words", [])
    n = len(words)
    for i in range(n):
        bw = words[i]
        if bw.get("contradiction_locked"):
            continue
        if bool(bw.get("is_cemented")) and float(bw.get("confidence", 0)) >= target:
            continue
        prev_end = float(words[i - 1]["end_sec"]) if i > 0 else 0.0
        next_start = (
            float(words[i + 1]["start_sec"])
            if i + 1 < n
            else float(base.get("total_duration_sec", prev_end + 10))
        )
        best = None
        for m in manifests[1:]:
            if i >= len(m.get("words", [])):
                continue
            cw = json.loads(json.dumps(m["words"][i]))
            cw = _enforce_floors(cw, target)
            if not (
                bool(cw.get("is_cemented")) and float(cw.get("confidence", 0)) >= target
            ):
                continue
            st = float(cw.get("start_sec", 0))
            en = float(cw.get("end_sec", st + 0.03))
            if st < prev_end - 0.05 or en > next_start + 0.05:
                continue
            st = max(prev_end, st)
            en = min(next_start, max(st + 0.03, en))
            cw["start_sec"] = round(st, 4)
            cw["end_sec"] = round(en, 4)
            cw["duration_cs"] = max(3, int(round((en - st) * 100)))
            score = (
                float(cw.get("posterior", 0) or 0),
                float(cw.get("confidence", 0) or 0),
            )
            if best is None or score > best[0]:
                best = (score, cw)
        if best is not None:
            cw = best[1]
            cw["confidence_source"] = (
                cw.get("confidence_source") or "unknown"
            ) + "+multipass_consensus"
            words[i] = cw
    prev_end = 0.0
    dur = float(base.get("total_duration_sec", 99999))
    for i, w in enumerate(words):
        st = max(prev_end, float(w.get("start_sec", prev_end)))
        en = max(st + 0.03, float(w.get("end_sec", st + 0.03)))
        if i + 1 < len(words):
            ns = float(words[i + 1].get("start_sec", dur))
            if en > ns:
                en = max(st + 0.03, ns)
        w["start_sec"] = round(st, 4)
        w["end_sec"] = round(min(dur, en), 4)
        w["duration_cs"] = max(
            3, int(round((float(w["end_sec"]) - float(w["start_sec"])) * 100))
        )
        w = _enforce_floors(w, target)
        prev_end = float(w["end_sec"])
    base["words"] = words
    # recompute lines from merged words
    lines = base.get("lines", [])
    by_line = {}
    for w in words:
        by_line.setdefault(int(w.get("line_index", 0)), []).append(w)
    for ln in lines:
        li = int(ln.get("index", 0)) - 1
        lw = by_line.get(li, [])
        if lw:
            ln["start_sec"] = round(float(lw[0]["start_sec"]), 4)
            ln["end_sec"] = round(float(lw[-1]["end_sec"]), 4)
    base["lines"] = lines
    base["cemented_words_ge_95_count"] = sum(1 for w in words if w.get("is_cemented"))
    base["uncemented_words_count"] = len(words) - base["cemented_words_ge_95_count"]
    base["engine"] = str(base.get("engine", "")) + "+multipass_consensus"
    return base


def run_multipass(
    *,
    project: Path,
    cli_cwd: Path,
    data_dir: Path,
    python_exe: str,
    passes: int = 3,
    log=None,
) -> Dict[str, Any]:
    manifests = []
    report = {"passes": [], "target": 0.95}
    for i in range(1, passes + 1):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if data_dir.exists():
            arch = data_dir.parent / f"{data_dir.name}.archived_multipass_p{i}_{ts}"
            shutil.move(str(data_dir), str(arch))
        cmd = [
            python_exe,
            "-u",
            "-m",
            "src.cli",
            "--project",
            str(project.resolve()),
            "--align-only",
        ]
        if log:
            log(f"[multipass] pass {i}/{passes}: {' '.join(cmd)}")
        proc = subprocess.run(
            cmd,
            cwd=str(cli_cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if log and proc.stdout:
            for line in proc.stdout.splitlines()[-20:]:
                log(f"[multipass.p{i}] {line}")
        if proc.returncode != 0:
            if log:
                log(f"[multipass] pass {i} failed rc={proc.returncode}")
            report["passes"].append({"pass": i, "rc": proc.returncode, "ok": False})
            continue
        mp = data_dir / "alignment_manifest.json"
        if mp.exists():
            m = _load(mp)
            manifests.append(m)
            c = _cemented_count(m)
            report["passes"].append(
                {
                    "pass": i,
                    "rc": 0,
                    "ok": True,
                    "cemented": c,
                    "uncemented": len(m.get("words", [])) - c,
                }
            )
            if log:
                log(f"[multipass] pass {i} cemented={c}")
    if not manifests:
        raise RuntimeError("multipass produced no manifests")
    merged = merge_manifests(manifests)
    merged["_multipass_report"] = report
    return merged
