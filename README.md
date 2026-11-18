# 🪄 TIKTOK WIZARD v5.1 🪄

[![Release](https://img.shields.io/badge/Release-v5.1-blue.svg)](https://github.com/R03LV1S/TikTokWizard/releases)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-green.svg)](https://github.com/R03LV1S/TikTokWizard)
[![License](https://img.shields.io/badge/License-Noncommercial%20(PolyForm)-purple.svg)](https://github.com/R03LV1S/TikTokWizard)

### Automatically create TikTok, YouTube Shorts, and Reels karaoke videos with word-by-word highlighted captions timed via the Greenfield Universal Alignment Engine, studio-grade atmospheric visual effects, two-tier persistent FX caching, and permanent configuration manifests.

---

## NEW IN VERSION 5.1 (TWO-TIER ATMOSPHERE & FX CACHING SYSTEM)

- **Instant 0.0s Atmosphere Reuse:** Eliminated redundant per-render Python/NumPy particle simulation by introducing a two-tier persistent caching system:
  1. **User & Stock Library (`_Assets/Overlays/`):** Pre-baked seamless looping master transparent overlays for core atmospheres (`cinematic_snow.mov`, `rising_embers.mov`, `bokeh_dust.mov`, `neon_stars.mov`).
  2. **Content-Addressable Cache (`DO NOT TOUCH/data/fx_cache/`):** Custom variations (color, density, velocity, resolution) are simulated once and permanently cached. Subsequent renders load in 0.0 milliseconds.
- **Hardware-Accelerated Stream Looping:** FFmpeg loops cached atmospheric transparent clips seamlessly across any song length via `-stream_loop -1` with zero CPU overhead.
- **Strict Single-Line Display:** Subtitle events strictly display exactly one (1) line at any time across all render and manifest pipelines.
- **Permanent Video Render Settings Manifests (.txt):** Every video rendered automatically generates a comprehensive companion `.txt` specification manifest in the project folder matching `<VideoName>.txt`.
- **100% Configurable & Setting Transparency:** Full cataloging of all active visualizer engines, color hex/RGB values, typography/font parameters, particle motion physics, background fit modes, TikTok UI safety zones, and creator handle placements.
- **Engine, Model & Silicon Accounting:** Explicit documentation of MMS_FA Viterbi alignment thresholds, Whisper models, vocal bandpass parameters, hardware accelerators (AMD Ryzen 9 7950X3D + Intel Arc Pro B70 QSV / DirectML), and exact render performance metrics ($0 API cost).
- **Universal Multi-Pipeline Integration:** Seamlessly integrated into both the interactive wizard (`cli_interactive.py`) and automated batch rendering (`automate.py`).

## NEW IN VERSION 4.8 (PIPELINE STABILITY, NON-BLOCKING STREAM PIPES & PROGRESS REPORTING)

- **Subprocess Pipe Deadlock Resolution:** Resolved Windows OS anonymous pipe buffer deadlocks during rawvideo generation of atmospheric particles and complex audio visualizers. Streams now drain with non-blocking DEVNULL pipe handles.
- **Two-Phase Interactive CLI Feedback:** Added explicit stage announcements (`[1/2] Preparing audio-visual layers...` with real-time frame progress counters and `[2/2] Compositing & encoding video...`).
- **Intel Arc QSV Hardware Acceleration:** Native `-global_quality` parameter mapping when utilizing `h264_qsv` hardware encoding on Intel Arc GPUs.

---

## NEW IN VERSION 4.7 (GREENFIELD v4.3 UNIVERSAL ALIGNMENT ENGINE - 100% CEMENTATION)

- **Dual-Anchor Linear Progression Scaffolding:** Dynamically interpolates unaligned words across high-confidence anchored windows.
- **Recursive Multi-Pass Surround & Fill:** Recursively resolves edge cases and low-confidence boundaries to achieve 100% cementation (confidence >= 0.95).

---

## NEW IN VERSION 4.6 (VIRAL ATMOSPHERIC PARTICLE & CINEMATIC MOTION EFFECTS ENGINE)

- **Studio-Grade Atmospheric Particles:** Replicates top-performing TikTok, Shorts visual atmospheres locally with high-performance procedural rendering (900+ FPS) directly into QuickTime MOV alpha streams via FFmpeg rawvideo pipe.
- **5 Viral Atmosphere Presets:**
  - `bokeh_dust`: Out-of-focus luminous golden orbs with soft falloff rings, sinusoidal sway, and beat-reactive breathing alpha (ideal for Lo-Fi, Bedroom Pop, Acoustic, and Chill tracks).
  - `rising_embers`: Upward fiery cinders with frame-to-frame alpha flicker and kick acceleration (ideal for Drift Phonk, Heavy Metal, and Hard Trap).
  - `cinematic_snow`: 3-layer parallax snowfall with natural wind drift and turbulence (ideal for Cinematic, Ambient, and Slowed & Reverb tracks).
  - `radial_burst`: High-energy explosive spark shockwaves and outward particle cascades triggered dynamically on kick and transient spikes (ideal for EDM, Festival Rave, and Club hits).
  - `neon_stars`: 4-point diamond cross twinkling stars with cosmic stardust (ideal for Synthwave, Cyberpunk, and Pop).
  - `none`: Clean background (disabled).
- **Transient Audio Reactivity:** Frame-by-frame RMS and spectral transient detection dynamically pulses particle brightness, scale, and velocity in lockstep with the song's beat.
- **Full Interactive Customizability:** Every atmospheric parameter is configurable in the CLI (`Option 13` under Customize Visuals): Style, Density (Low: 30, Medium: 65, High: 130), Drift Speed (Slow, Normal, Fast), Audio Reactivity (On/Off), and Color Palettes (Auto or Custom Hex).
- **Curated Genre Integration:** All 20 genre visual profiles now feature tailored atmospheric particle presets out of the box.

---

## STEP 1: HOW TO RUN

1. Open the folder named **`START HERE`**.
2. Double-click the launcher for your operating system:
   - **Windows:** `Double-Click to Start (Windows).bat`
   - **macOS:** `Double-Click to Start (Mac).command`
   - **Linux:** `Double-Click to Start (Linux).sh`

*Everything installs and sets up automatically in an isolated virtual environment on your first run.*

---

## STEP 2: HOW TO ADD YOUR OWN SONG

1. Open **`_Projects`** inside `START HERE`.
2. Create a new folder with your song's name (e.g., `_Projects/My Song`).
3. Add two files inside the folder:
   - **Your audio file:** `song.mp3` (or `.wav`, `.flac`, `.m4a`)
   - **Your lyrics file:** `lyrics.txt` (structured line-by-line as you want captions to appear)
4. (Optional) Custom fonts can be added to `_Projects/_Assets/Fonts`.

Launch **TikTok Wizard**! It will detect your song, isolate the vocal tracks, align every word to the audio, and render a vertical 1080x1920 MP4 video ready for publication.

---

## REQUIREMENTS

- **Python 3.10+**: Automatically detected or installed via launcher.
- **FFmpeg**: Required for final video rendering (automatically installed if missing on Windows via winget).

---

## PROJECT STRUCTURE

```text
TikTokWizard/
  START HERE/                 <- User interface directory
    _Projects/                <- Songs, lyrics, and rendered videos
    Double-Click to Start...  <- One-click OS launchers
  DO NOT TOUCH/               <- Core engine (models, venv, source)
    ARCHITECTURE.md           <- Engine technical specification
    config.json               <- Video, render, and model settings
    requirements.txt          <- Pinned python dependencies
    src/                      <- Greenfield Universal Engine v4.3 source
    tests/                    <- Invariant verification test suite
```
