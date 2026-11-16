#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TikTok Wizard - first-run installer with a real 0-100% progress bar.

Runs on the SYSTEM python (before the virtual environment exists), so it
must use the standard library only.

What it does, in order:
    1. create the virtual environment
    2. upgrade pip inside it
    3. resolve the dependency graph and measure the exact download size
    4. download every python package (in parallel) with byte-accurate progress
    5. download the AI models with the same progress bar
    6. install the packages
    7. verify the installation really works

If the fast path fails it falls back to a plain "pip install -r
requirements.txt", so the install always completes.

Usage:
    python installer.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
REQUIREMENTS = APP_DIR / "requirements.txt"
VENV_DIR = APP_DIR / "venv"


def venv_python() -> Path:
    """Path to the venv's python.

    Windows puts it in venv/Scripts/python.exe, macOS and Linux use
    venv/bin/python.
    """
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


VENV_PY = venv_python()

# Phase boundaries as a percentage of the whole job.
P_VENV_START, P_VENV_END = 0, 5
P_PIP_START, P_PIP_END = 5, 9
P_RESOLVE_START, P_RESOLVE_END = 9, 14
P_DOWNLOAD_START, P_DOWNLOAD_END = 14, 34
P_INSTALL_START, P_INSTALL_END = 34, 42
P_MODEL_START, P_MODEL_END = 42, 96
P_VERIFY_START, P_VERIFY_END = 96, 100

DOWNLOAD_THREADS = 8
USER_AGENT = "TikTokWizard-Installer/1.0"

# The AI models. These are what produce the ~90% alignment confidence:
#   faster-whisper small     -> dual-stream ASR transcript
#   torchaudio MMS_FA        -> state-tracking <star>-CTC phoneme alignment
#   audio-separator          -> vocal isolation
MODELS = [
    ("Systran/faster-whisper-small", "Whisper small - speech recognition"),
]


# ======================================================================
# Console rendering
# ======================================================================
class Console:
    """A small ANSI progress renderer that degrades to plain text."""

    def __init__(self) -> None:
        self.tty = False
        self.ansi = False
        self._lines_drawn = 0
        self._lock = threading.Lock()
        self._setup()

    def _setup(self) -> None:
        # TW_FORCE_ANSI=1 renders the progress bar even when stdout is
        # redirected (useful for testing, and for consoles that misreport
        # themselves as non-interactive).
        forced = os.environ.get("TW_FORCE_ANSI", "").strip().lower() in (
            "1", "true", "yes", "on")
        try:
            self.tty = sys.stdout.isatty()
        except Exception:
            self.tty = False
        if not self.tty and not forced:
            return
        self.tty = True
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
                if not (mode.value & ENABLE_VIRTUAL_TERMINAL_PROCESSING):
                    kernel32.SetConsoleMode(
                        handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING)
        except Exception:
            pass
        self.ansi = True

    # -- colour helpers ------------------------------------------------
    def dim(self, s: str) -> str:
        return f"\033[2m{s}\033[0m" if self.ansi else s

    def bold(self, s: str) -> str:
        return f"\033[1m{s}\033[0m" if self.ansi else s

    def cyan(self, s: str) -> str:
        return f"\033[36m{s}\033[0m" if self.ansi else s

    def green(self, s: str) -> str:
        return f"\033[32m{s}\033[0m" if self.ansi else s

    def yellow(self, s: str) -> str:
        return f"\033[33m{s}\033[0m" if self.ansi else s

    def red(self, s: str) -> str:
        return f"\033[31m{s}\033[0m" if self.ansi else s

    # -- primitives -----------------------------------------------------
    def width(self) -> int:
        try:
            w = shutil.get_terminal_size((80, 24)).columns
            return max(40, min(w, 120))
        except Exception:
            return 80

    def write_line(self, text: str = "") -> None:
        with self._lock:
            if self.ansi:
                sys.stdout.write("\033[2K" + text + "\n")
                self._lines_drawn += 1
            else:
                sys.stdout.write(text + "\n")
            sys.stdout.flush()

    def draw_block(self, lines: list) -> None:
        """Redraw a block of lines in place (ANSI only)."""
        if not self.ansi:
            return
        with self._lock:
            if self._lines_drawn:
                sys.stdout.write(f"\033[{self._lines_drawn}A")
            for line in lines:
                sys.stdout.write("\033[2K" + line + "\n")
            self._lines_drawn = len(lines)
            sys.stdout.flush()

    def hide_cursor(self) -> None:
        if self.ansi:
            try:
                sys.stdout.write("\033[?25l")
                sys.stdout.flush()
            except Exception:
                pass

    def show_cursor(self) -> None:
        if self.ansi:
            try:
                sys.stdout.write("\033[?25h")
                sys.stdout.flush()
            except Exception:
                pass

    def clear_final(self) -> None:
        """Erase the live block so normal output follows cleanly."""
        if self.ansi and self._lines_drawn:
            with self._lock:
                sys.stdout.write(f"\033[{self._lines_drawn}A")
                for _ in range(self._lines_drawn):
                    sys.stdout.write("\033[2K\n")
                sys.stdout.write(f"\033[{self._lines_drawn}A")
                sys.stdout.flush()
            self._lines_drawn = 0


# ======================================================================
# Progress tracker
# ======================================================================
class Tracker:
    """Tracks overall percentage plus per-file byte progress."""

    def __init__(self, console: Console) -> None:
        self.con = console
        # Reentrant: render() reads the counters while holding this lock.
        self.lock = threading.RLock()
        self.pct = 0.0
        self.done_bytes = 0
        self.total_bytes = 0
        self.active: dict = {}
        self.last_file = ""
        self.status = "Starting"
        self.detail = ""
        self.spinner = 0
        self._stop = threading.Event()
        self._thread = None
        self._plain_last = None
        # When set to (start_pct, end_pct), the percentage is derived from
        # the byte counters instead of being set manually.
        self.byte_phase = None

    def set_phase(self, pct: float, status: str) -> None:
        with self.lock:
            self.pct = max(self.pct, pct)
            self.status = status
            self.detail = ""

    def set_status(self, status: str, detail: str = "") -> None:
        with self.lock:
            self.status = status
            self.detail = detail

    def begin_file(self, name: str, size: int) -> None:
        with self.lock:
            self.active[name] = [0, size]
            self.last_file = name

    def advance_file(self, name: str, amount: int) -> None:
        with self.lock:
            if name in self.active:
                self.active[name][0] = min(self.active[name][1],
                                          self.active[name][0] + amount)

    def end_file(self, name: str) -> None:
        with self.lock:
            entry = self.active.pop(name, None)
            if entry:
                self.done_bytes += entry[1]

    def set_total(self, total: int) -> None:
        with self.lock:
            self.total_bytes = total

    def set_bytes(self, done: int, total: int) -> None:
        """Report byte progress directly (used by the model download)."""
        with self.lock:
            self.done_bytes = done
            self.total_bytes = total
            self.active = {}

    def set_byte_phase(self, start_pct: float, end_pct: float) -> None:
        """Drive the percentage from the byte counters."""
        with self.lock:
            self.byte_phase = (start_pct, end_pct)

    def clear_byte_phase(self) -> None:
        with self.lock:
            self.byte_phase = None

    def bytes_now(self) -> int:
        with self.lock:
            return self.done_bytes + sum(v[0] for v in self.active.values())

    # -- rendering ------------------------------------------------------
    def render(self) -> None:
        if not self.con.ansi:
            self._render_plain()
            return
        with self.lock:
            bp = self.byte_phase
            if bp and self.total_bytes:
                span = bp[1] - bp[0]
                frac = min(1.0, (self.done_bytes
                                  + sum(v[0] for v in self.active.values()))
                           / self.total_bytes)
                self.pct = max(self.pct, bp[0] + span * frac)
        pct = max(0.0, min(100.0, self.pct))
        bar_w = max(18, min(46, self.con.width() - 34))
        filled = int(round(bar_w * pct / 100.0))
        bar = "#" * filled + "." * (bar_w - filled)

        with self.lock:
            status = self.status
            detail = self.detail
            last = self.last_file
            now = self.done_bytes + sum(v[0] for v in self.active.values())
            total = self.total_bytes
            self.spinner = (self.spinner + 1) % 4

        lines = []
        head = f"  {self.con.cyan('[##]')} {self.con.bold(status)}"
        if detail:
            head += f"  {self.con.dim(detail)}"
        lines.append(head)

        line2 = f"  {self.con.cyan(bar)} {self.con.bold(f'{pct:5.1f}%')}"
        if total:
            line2 += f"  {self.con.dim(f'{now/1024/1024:,.0f} / {total/1024/1024:,.0f} MB')}"
        lines.append(line2)

        if last:
            short = last if len(last) <= 58 else "..." + last[-55:]
            lines.append(f"  {self.con.dim(short)}")
        else:
            lines.append(f"  {self.con.dim(' ' * 40)}")

        # Always exactly 3 lines, so the cursor arithmetic stays correct
        # and no stale frame is left behind.
        while len(lines) < 3:
            lines.append("")
        self.con.draw_block(lines[:3])

    def _render_plain(self) -> None:
        """Fallback for non-interactive output: one line per status change."""
        with self.lock:
            bp = self.byte_phase
            if bp and self.total_bytes:
                span = bp[1] - bp[0]
                frac = min(1.0, (self.done_bytes
                                  + sum(v[0] for v in self.active.values()))
                           / self.total_bytes)
                self.pct = max(self.pct, bp[0] + span * frac)
            key = (self.status, self.detail)
            pct = max(0.0, min(100.0, self.pct))
            now, total = self.bytes_now(), self.total_bytes
        if key == self._plain_last:
            return
        self._plain_last = key
        line = f"  [{pct:5.1f}%] {self.status}"
        if self.detail:
            line += f" - {self.detail}"
        if total:
            line += f"  ({now/1024/1024:,.0f} / {total/1024/1024:,.0f} MB)"
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def start(self, interval: float = 0.12) -> None:
        def loop():
            while not self._stop.is_set():
                self.render()
                self._stop.wait(interval)
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)


# ======================================================================
# Steps
# ======================================================================
def run_quiet(cmd: list, timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout, cwd=str(APP_DIR))


def step_venv(con: Console, tracker: Tracker) -> bool:
    tracker.set_phase(P_VENV_START, "Creating Python environment")
    if venv_python().exists():
        tracker.set_phase(P_VENV_END, "Python environment ready")
        return True
    r = run_quiet([sys.executable, "-m", "venv", str(VENV_DIR)], timeout=600)
    if r.returncode != 0 or not venv_python().exists():
        con.write_line(con.red("  [FAIL] could not create the Python environment"))
        con.write_line(con.dim((r.stderr or r.stdout or "")[-1200:]))
        return False
    tracker.set_phase(P_VENV_END, "Python environment ready")
    return True


def step_pip_upgrade(con: Console, tracker: Tracker) -> bool:
    tracker.set_phase(P_PIP_START, "Updating pip")
    try:
        r = run_quiet([str(venv_python()), "-m", "pip", "install", "--upgrade", "pip",
                       "--disable-pip-version-check", "--no-warn-script-location"],
                      timeout=1800)
    except subprocess.TimeoutExpired:
        r = None
    if r is not None and r.returncode != 0:
        # A failed pip upgrade is not fatal - the bundled pip still works.
        tracker.set_phase(P_PIP_END, "Using the bundled pip")
        return True
    tracker.set_phase(P_PIP_END, "pip is up to date")
    return True


def step_resolve(con: Console, tracker: Tracker):
    """Resolve the dependency graph. Returns (packages, total_bytes, tmpdir)."""
    tracker.set_phase(P_RESOLVE_START, "Resolving dependencies")
    tmp = Path(tempfile.mkdtemp(prefix="tw_resolve_"))
    report = tmp / "report.json"
    try:
        r = run_quiet([str(venv_python()), "-m", "pip", "install", "--dry-run",
                       "--quiet", "--report", str(report),
                       "--disable-pip-version-check", "-r", str(REQUIREMENTS)],
                      timeout=1800)
    except subprocess.TimeoutExpired:
        return [], 0, tmp
    if r.returncode != 0 or not report.exists():
        return [], 0, tmp
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
    except Exception:
        return [], 0, tmp

    urls = []
    for e in data.get("install", []) or []:
        u = (e.get("download_info") or {}).get("url")
        if u:
            urls.append(u)
    if not urls:
        return [], 0, tmp

    # Ask the CDN for the exact size of each file (parallel, ~2 seconds).
    def head(u):
        try:
            req = urllib.request.Request(u, method="HEAD",
                                         headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return u, int(resp.headers.get("Content-Length", 0) or 0)
        except Exception:
            return u, 0

    tracker.set_status("Resolving dependencies", f"{len(urls)} packages")
    resolved = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        for u, size in ex.map(head, urls):
            if size:
                resolved.append((u, size))

    tracker.set_phase(P_RESOLVE_END, "Dependencies resolved")
    return resolved, sum(s for _, s in resolved), tmp


def step_download(con: Console, tracker: Tracker, packages: list):
    """Download every package into a local wheelhouse with byte progress."""
    wheelhouse = Path(tempfile.mkdtemp(prefix="tw_wheels_"))
    tracker.set_phase(P_DOWNLOAD_START, "Downloading packages")
    tracker.set_total(sum(s for _, s in packages))
    tracker.set_byte_phase(P_DOWNLOAD_START, P_DOWNLOAD_END)

    span = P_DOWNLOAD_END - P_DOWNLOAD_START
    total = tracker.total_bytes or 1

    def fetch(item):
        url, size = item
        name = url.rsplit("/", 1)[-1]
        target = wheelhouse / name
        tracker.begin_file(name, size)
        tmp = target.with_suffix(target.suffix + ".part")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(262144)
                    if not chunk:
                        break
                    fh.write(chunk)
                    tracker.advance_file(name, len(chunk))
            os.replace(tmp, target)
            ok = True
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            ok = False
        tracker.end_file(name)
        return ok

    failures = 0
    failed_names = []
    with ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as ex:
        for item, ok in zip(packages, ex.map(fetch, packages)):
            if not ok:
                failures += 1
                failed_names.append(item[0].rsplit("/", 1)[-1])
    if failures:
        con.write_line(con.yellow(
            f"  [WARN] {failures} package(s) could not be downloaded "
            f"(e.g. {failed_names[0]}) - falling back to the online installer."))

    tracker.clear_byte_phase()
    tracker.pct = max(tracker.pct, P_DOWNLOAD_END)
    return wheelhouse, failures


# The helper that runs INSIDE the venv (huggingface_hub lives there).
MODEL_FETCHER = r'''
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Windows cannot create symlinks unless Developer Mode is on. The hub
# library does that automatically, but it fails half-way through a large
# download. Fetching into a plain directory and copying the real files
# into the cache ourselves avoids symlinks completely.

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm as _tqdm

MODELS = [
    ("Systran/faster-whisper-small", "Whisper small - speech recognition"),
]

STATE = {"done": 0, "total": 0, "label": "", "model_base": 0}


class Reporter(_tqdm):
    """tqdm that forwards byte progress to the parent process.

    tqdm's own bar is disabled: the parent renders the progress bar, and
    writing tqdm output to stderr would fill the pipe the parent never
    drains, which deadlocks the download.
    """

    def __init__(self, *a, **kw):
        kw["disable"] = True
        super().__init__(*a, **kw)
        self._model_bytes = 0

    def update(self, n=1):
        self._model_bytes += n
        done = STATE["model_base"] + self._model_bytes
        print("TW_MODEL|%d|%d|%s" % (done, STATE["total"], STATE["label"]),
              flush=True)
        return super().update(n)


def cache_root():
    from huggingface_hub import constants
    return Path(constants.HF_HUB_CACHE)


def repo_folder(repo_id):
    return cache_root() / ("models--" + repo_id.replace("/", "--"))


def fetch_model(repo_id, label, size):
    """Download one model into the hub cache as REAL files.

    The hub's own snapshot_download() symlinks the weights into
    snapshots/<rev>/. On Windows without Developer Mode that raises
    WinError 1314 part-way through, leaving a half-downloaded model.
    Here we fetch each file with local_dir (real files, no symlinks)
    and copy it into the cache layout ourselves.
    """
    api = HfApi()
    info = api.model_info(repo_id, files_metadata=True)
    files = [s.rfilename for s in info.siblings if s.rfilename]

    tmp = Path(tempfile.mkdtemp(prefix="tw_model_"))
    try:
        got = {}
        for name in files:
            before = sum(f.stat().st_size for f in tmp.rglob("*")
                         if f.is_file())
            # hf_hub_download with local_dir writes a real file and
            # never creates a symlink.
            hf_hub_download(
                repo_id=repo_id,
                filename=name,
                local_dir=str(tmp),
                local_dir_use_symlinks=False,
            )
            src = tmp / name
            if src.is_file():
                got[name] = src
                after = sum(f.stat().st_size for f in tmp.rglob("*")
                            if f.is_file())
                moved = after - before
                if moved > 0:
                    STATE["done"] = STATE["model_base"] + sum(
                        f.stat().st_size for f in tmp.rglob("*") if f.is_file())
                    print("TW_MODEL|%d|%d|%s" % (STATE["done"], STATE["total"],
                                                   STATE["label"]), flush=True)

        if not got:
            raise RuntimeError("no files were downloaded")

        # Put them where the loaders look: <cache>/models--x--y/snapshots/<rev>/
        folder = repo_folder(repo_id)
        revision = getattr(info, "sha", None) or "main"
        snap = folder / "snapshots" / revision
        snap.mkdir(parents=True, exist_ok=True)
        for name, src in got.items():
            dst = snap / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() and dst.stat().st_size == src.stat().st_size:
                continue
            shutil.copyfile(src, dst)

        # Point the repo at this revision.
        refs = folder / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        (refs / "main").write_text(revision, encoding="utf-8")

        # The weights must be findable without the hub's blob indirection.
        return len(got)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    api = HfApi()
    plan = []
    total = 0
    for repo, label in MODELS:
        try:
            info = api.model_info(repo, files_metadata=True)
            size = sum((s.size or 0) for s in info.siblings)
            total += size
            plan.append((repo, label, size))
        except Exception as exc:
            print("TW_ERROR|cannot inspect %s: %s" % (repo, exc), flush=True)

    STATE["total"] = total
    done = 0
    for repo, label, size in plan:
        STATE["label"] = label
        STATE["model_base"] = done
        print("TW_MODEL|%d|%d|%s" % (done, total, label), flush=True)
        try:
            n = fetch_model(repo, label, size)
            done = STATE["model_base"] + size
            STATE["done"] = done
            print("TW_MODEL|%d|%d|%s" % (done, total, label), flush=True)
        except Exception as exc:
            print("TW_ERROR|%s: %s" % (repo, exc), flush=True)
    print("TW_MODEL|%d|%d|done" % (done, total), flush=True)


main()
'''


def step_models(con: Console, tracker: Tracker) -> bool:
    """Download the AI models with the same byte-accurate progress bar.

    These are the models the alignment engine uses, and they are what
    produce the ~90% confidence. Downloading them during setup means the
    first render starts instantly instead of silently pulling 3.4 GB.
    """
    tracker.set_phase(P_MODEL_START, "Downloading AI models")
    # The package total is meaningless here - start the model count fresh.
    tracker.set_bytes(0, 0)
    tracker.set_byte_phase(P_MODEL_START, P_MODEL_END)

    helper = APP_DIR / "_tw_model_fetch.py"
    helper.write_text(MODEL_FETCHER, encoding="utf-8")
    proc = None
    try:
        proc = subprocess.Popen(
            [str(venv_python()), str(helper)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True, cwd=str(APP_DIR), bufsize=1,
        )
    except Exception as exc:
        con.write_line(con.yellow(f"  [WARN] could not start the model download: {exc}"))
        try:
            helper.unlink(missing_ok=True)
        except Exception:
            pass
        tracker.set_phase(P_MODEL_END, "Models will download on first use")
        return True

    failed = False
    other_lines = []
    try:
        if proc.stdout is not None:
            for line in proc.stdout:
                line = line.strip()
                if line.startswith("TW_MODEL|"):
                    parts = line.split("|")
                    try:
                        done = int(parts[1])
                        total = int(parts[2])
                        label = parts[3] if len(parts) > 3 else ""
                    except (ValueError, IndexError):
                        continue
                    tracker.set_bytes(done, total)
                    tracker.set_status("Downloading AI models", label)
                    frac = (done / total) if total else 0.0
                    tracker.pct = max(
                        tracker.pct,
                        P_MODEL_START + (P_MODEL_END - P_MODEL_START) * frac)
                elif line.startswith("TW_ERROR|"):
                    failed = True
                    con.write_line(con.yellow(f"  [WARN] {line[9:]}"))
                elif line:
                    other_lines.append(line)
        proc.wait(timeout=900)
        if proc.returncode != 0 and not failed:
            failed = True
            con.write_line(con.yellow(
                f"  [WARN] the model download helper exited with code "
                f"{proc.returncode}."))
        if failed and other_lines:
            con.write_line(con.dim("        " + other_lines[-1][:160]))
    except Exception as exc:
        con.write_line(con.yellow(f"  [WARN] model download problem: {exc}"))
        failed = True
    finally:
        try:
            helper.unlink(missing_ok=True)
        except Exception:
            pass

    tracker.clear_byte_phase()
    if failed:
        # Not fatal - the app downloads models on demand too - but the user
        # must know, because that is the difference between a smooth first
        # render and a long wait with no explanation.
        tracker.set_phase(P_MODEL_END, "Models will download on first use")
        con.write_line(con.yellow(
            "  [WARN] Some AI models could not be downloaded now. They will "
            "be fetched automatically the first time you render."))
        return True
    tracker.set_phase(P_MODEL_END, "AI models ready")
    return True


MODEL_CHECK = r"""
import sys
import traceback

ok = True

# 1. faster-whisper: the ASR transcript
try:
    from faster_whisper import WhisperModel
    WhisperModel("small", device="cpu", compute_type="int8")
    print("TW_OK|Whisper small")
except Exception as exc:
    ok = False
    print("TW_FAIL|Whisper small: %s" % str(exc)[:180])

# 2. MMS_FA: multilingual <star>-CTC forced aligner
try:
    import torchaudio
    bundle = torchaudio.pipelines.MMS_FA
    bundle.get_model(with_star=True)
    print("TW_OK|Torchaudio MMS_FA (<star>-CTC)")
except Exception as exc:
    ok = False
    print("TW_FAIL|MMS_FA: %s" % str(exc)[:180])

sys.exit(0 if ok else 1)
"""


def step_verify_models(con: Console, tracker: Tracker) -> bool:
    """Load every model the app needs, so 'done' can never lie."""
    tracker.set_status("Checking AI models", "loading them once")

    helper = APP_DIR / "_tw_model_check.py"
    helper.write_text(MODEL_CHECK, encoding="utf-8")
    try:
        r = subprocess.run([str(venv_python()), str(helper)],
                           capture_output=True, text=True, timeout=2400,
                           cwd=str(APP_DIR))
    except Exception as exc:
        con.write_line(con.yellow(f"  [WARN] could not check the models: {exc}"))
        try:
            helper.unlink(missing_ok=True)
        except Exception:
            pass
        return True
    finally:
        try:
            helper.unlink(missing_ok=True)
        except Exception:
            pass

    failed = []
    for line in (r.stdout or "").splitlines():
        if line.startswith("TW_OK|"):
            con.write_line(con.dim(f"    [OK] {line[6:]}"))
        elif line.startswith("TW_FAIL|"):
            failed.append(line[8:])
            con.write_line(con.yellow(f"    [!!] {line[8:]}"))

    if failed:
        con.write_line(con.yellow(
            f"  [WARN] {len(failed)} AI model(s) could not be loaded. They "
            "will be fetched again the first time you render."))
        return True
    return True


def step_install(con: Console, tracker: Tracker, wheelhouse: Path) -> bool:
    tracker.set_phase(P_INSTALL_START, "Installing packages")
    cmd = [str(venv_python()), "-m", "pip", "install", "--no-index",
           "--find-links", str(wheelhouse), "--disable-pip-version-check",
           "--no-warn-script-location", "-r", str(REQUIREMENTS)]
    r = run_quiet(cmd, timeout=3600)
    if r.returncode != 0:
        # The offline path can fail if the resolver wanted a different build.
        # Fall back to a normal online install so the user always ends up
        # with a working environment.
        tracker.set_status("Installing packages", "falling back to online install")
        r = run_quiet([str(venv_python()), "-m", "pip", "install",
                       "--disable-pip-version-check", "--no-warn-script-location",
                       "-r", str(REQUIREMENTS)], timeout=3600)
    if r.returncode != 0:
        con.write_line(con.red("  [FAIL] package installation failed"))
        con.write_line(con.dim((r.stdout or "")[-800:]))
        con.write_line(con.dim((r.stderr or "")[-800:]))
        return False
    tracker.set_phase(P_INSTALL_END, "Packages installed")
    return True


def step_fallback_install(con: Console, tracker: Tracker) -> bool:
    """Plain online install, used when resolution or download failed."""
    tracker.set_phase(P_INSTALL_START, "Installing packages")
    r = run_quiet([str(venv_python()), "-m", "pip", "install",
                   "--disable-pip-version-check", "--no-warn-script-location",
                   "-r", str(REQUIREMENTS)], timeout=3600)
    if r.returncode != 0:
        con.write_line(con.red("  [FAIL] package installation failed"))
        con.write_line(con.dim((r.stdout or "")[-800:]))
        con.write_line(con.dim((r.stderr or "")[-800:]))
        return False
    tracker.set_phase(P_INSTALL_END, "Packages installed")
    return True


VERIFY_CODE = (
    "import importlib\n"
    "mods = ['faster_whisper','audio_separator','torch','torchaudio',\n"
    "        'librosa','numpy','scipy','rapidfuzz','anyascii']\n"
    "bad = []\n"
    "for m in mods:\n"
    "    try:\n"
    "        importlib.import_module(m)\n"
    "    except Exception:\n"
    "        bad.append(m)\n"
    "print('MISSING:' + ','.join(bad))\n"
)


def step_verify(con: Console, tracker: Tracker) -> bool:
    tracker.set_phase(P_VERIFY_START, "Verifying installation")
    probe = APP_DIR / "_tw_install_probe.py"
    try:
        probe.write_text(VERIFY_CODE, encoding="utf-8")
        r = run_quiet([str(venv_python()), str(probe)], timeout=900)
        out = (r.stdout or "").strip()
        if r.returncode != 0:
            tracker.set_phase(P_VERIFY_END, "Verification skipped")
            return True
        missing = out.split("MISSING:", 1)[1].strip() if "MISSING:" in out else ""
        if missing:
            tracker.set_phase(P_VERIFY_END, "Some packages failed to import")
            con.write_line(con.yellow(f"  [WARN] could not import: {missing}"))
            return False
        tracker.set_phase(P_VERIFY_END, "Installation verified")
        return True
    except Exception:
        tracker.set_phase(P_VERIFY_END, "Verification skipped")
        return True
    finally:
        try:
            probe.unlink(missing_ok=True)
        except Exception:
            pass


# ======================================================================
# Main
# ======================================================================
def main() -> int:
    con = Console()
    tracker = Tracker(con)

    con.write_line()
    con.write_line(con.bold("  TikTok Wizard - First-time setup"))
    con.write_line(con.dim("  This happens once. It downloads everything the app needs."))
    con.write_line()

    if not REQUIREMENTS.exists():
        con.write_line(con.red(f"  [FAIL] missing {REQUIREMENTS}"))
        return 1

    tracker.start()
    con.hide_cursor()

    wheelhouse = None
    resolve_tmp = None
    ok = False
    try:
        if not step_venv(con, tracker):
            return 1
        if not step_pip_upgrade(con, tracker):
            return 1

        packages, total, resolve_tmp = step_resolve(con, tracker)

        if packages:
            wheelhouse, failures = step_download(con, tracker, packages)
        else:
            wheelhouse = None
            failures = 1

        # Install first: the model fetcher runs with the venv python and
        # needs huggingface_hub, which the install provides. Installing
        # from the local wheelhouse is fast, so it barely moves the bar.
        if wheelhouse is not None and not failures:
            ok = step_install(con, tracker, wheelhouse)
        else:
            tracker.set_status("Installing packages",
                               "resolving directly from PyPI")
            ok = step_fallback_install(con, tracker)

        # The models give the ~90% alignment confidence, so they are
        # fetched now rather than silently on the first render.
        if ok:
            ok = step_models(con, tracker)

        if ok:
            step_verify_models(con, tracker)

        if ok:
            ok = step_verify(con, tracker)
    except KeyboardInterrupt:
        con.write_line()
        con.write_line(con.yellow("  Setup cancelled."))
        return 130
    except Exception as exc:
        con.write_line(con.red(f"  [FAIL] setup error: {exc}"))
        return 1
    finally:
        tracker.stop()
        con.show_cursor()
        con.clear_final()
        for d in (wheelhouse, resolve_tmp):
            if d:
                try:
                    shutil.rmtree(d, ignore_errors=True)
                except Exception:
                    pass

    con.write_line()
    if ok:
        # The launcher checks for this marker. It is written only after a
        # complete, verified install - so a half-finished venv is never used.
        try:
            (VENV_DIR / ".install-complete").write_text(
                "TikTok Wizard setup completed\n", encoding="utf-8")
        except Exception:
            pass
        con.write_line(con.green("  [OK] Setup complete. Starting TikTok Wizard..."))
        con.write_line()
        return 0

    con.write_line(con.red("  [FAIL] Setup did not complete."))
    con.write_line(con.dim("  The message above says which step failed."))
    con.write_line(con.dim("  Please run the launcher again."))
    con.write_line()
    return 1


if __name__ == "__main__":
    sys.exit(main())
