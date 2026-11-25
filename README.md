# 🪄 TIKTOK WIZARD v5.7 🪄

[![Release](https://img.shields.io/badge/Release-v5.7-blue.svg)](https://github.com/R03LV1S/TikTokWizard/releases)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-green.svg)](https://github.com/R03LV1S/TikTokWizard)
[![License](https://img.shields.io/badge/License-Noncommercial%20(PolyForm)-purple.svg)](https://github.com/R03LV1S/TikTokWizard)

### Automatically create TikTok, YouTube Shorts, and Reels karaoke videos with word-by-word highlighted captions timed via the Greenfield Universal Alignment Engine, studio-grade atmospheric visual effects, isolated vocal stem / acapella export, two-tier persistent FX caching, and permanent configuration manifests.

---

## NEW IN VERSION 5.7 (DUAL MACHINE & HUMAN-READABLE SYNC EDITOR & TYPOGRAPHY CASING)

- **Dual Machine-Readable & Human-Readable Sync & Highlight Timings Editor (`lyrics_sync_editor.txt`):**
  - *Section 1 (Machine-Readable JSON):* Contains full precision metadata: line intervals, word onsets/offsets, durations, acoustic posteriors, and dual-model confidence scores.
  - *Section 2 (Human-Readable Easy-to-Fix Editor):* Intuitive plain text format:
    `Line 1: [00:14.79 -> 00:17.35] If (21cs) you (33cs) want (26cs) love (96cs)`
    Enables creators to freely correct typos, alter timestamps, or modify word highlight durations `(XXcs)` directly in Notepad!
  - *Self-Balancing Timing Engine:* Automatically and smoothly balances missing word durations across the available line window.
  - *Seamless CLI Reload & Auto-Sync:* Dedicated Project Menu Option 9 to open and reload manual edits directly into ASS and video renders.
- **Configurable Typography Text Casing & Font Size Homogenization:**
  - Added interactive Text Casing menu (Option 4 under Customize Visuals & Typography):
    1. *ALL CAPS (Homogenized font height & zero baseline jitter)* [Recommended for TikTok viral aesthetic]
    2. *Title Case (Capitalize Every Word)*
    3. *Sentence case (Capitalize first letter)*
    4. *all lowercase*
    5. *Original As-Is (Preserve exact lyric input casing)*
  - Eliminates visual height wobble caused by mixed lowercase ascenders and descenders.

---

## NEW IN VERSION 5.6 (FOUR-PHASE UNIVERSAL ADAPTIVE ENGINE UPGRADE)

- **Phase 1: Dynamic Characters-Per-Second (CPS) Readability Engine:** Replaced fixed scalar subtitle floors with an empirical physiological reading model ($14.5 \text{ chars/sec}$, floor $0.90\text{s}$, ceiling $3.50\text{s}$). Subtitles dynamically scale display hold based on actual character length, while strictly lookahead-clamping with zero collision to `next_start - 0.04s`.
- **Phase 2: Track-Calibrated WPS Monotonic Interpolation & VAD Clamps:** Self-calibrates empirical words-per-second (WPS) cadence directly from internal anchored word pairs ($[0.12, 0.90]\text{s/word}$) and clamps unanchored boundary extrapolation strictly between the first and last energy-active vocal frames (`[t_first_vocal, t_last_vocal]`), eliminating dead-air drift.
- **Phase 3: Universal Multi-Tier Multilingual Consensus Engine (`lang_detect.py`):** Multi-modal language resolution combining:
  1. *Unicode Script Analysis:* Instant recognition of CJK (Japanese Hiragana/Katakana, Korean Hangul, Hanzi), Cyrillic, Arabic/Persian, Devanagari (Hindi), Greek, Hebrew, and Thai scripts.
  2. *Latin-Script Multilingual Profiler:* 20-language grammatical function word frequency ranker (`en, es, pt, fr, de, it, nl, pl, ro, tr, sv, da, no, fi, cs, id, vi, hu...`).
  3. *Acoustic Bayesian Prior:* Fuses lyric text confidence with Whisper audio language probability vector for fail-safe consensus.
- **Phase 4: Adaptive Otsu Log-RMS Bimodal VAD Thresholding:** Replaced static RMS percentile heuristic with dynamic Otsu bimodal thresholding on the acapella log-energy distribution, mathematically isolating background instrumental bleed from genuine vocal bursts across all genres.

---

## NEW IN VERSION 5.5 (UNIVERSAL ADAPTIVE ALIGNMENT & READABILITY REFINEMENTS)

- **Lyric-First Language Detection & ASR Cache Isolation:** Determines primary language from lyric text stopword analysis before looking at disk caches, preventing stale Spanish ASR caches or hardcoded defaults from corrupting English tracks (MMS_FA and Whisper now lock to the true lyric language).
- **Smooth Monotonic Linear Interpolation:** Replaced the flat plateau ASR fallback (`word_times[i] = last_t`) with `np.interp` piecewise linear interpolation, preventing unmapped words from collapsing into previous vocal islands across long instrumental drops and breaks.
- **Adaptive Subtitle Readability Hold:** Subtitles now maintain an adaptive on-screen hold (up to 1.20s minimum reading time or +0.35s lead-out) across open silence, while strictly clamping with zero overlap on rapid successor lines.
- **Strict Single-Token Preservation for Spoken Digits:** Multi-word numeric expansions (e.g. `38 -> treinta y ocho`) expand to acoustic phonetics internally while preserving the exact single display token byte-for-byte in the manifest and `.ass` file.

---

## NEW IN VERSION 5.4 (GREENFIELD ENGINE v5 & DUAL-MODEL CTC CONSENSUS)

- **Universal Hardware Compute Profiles (`CPU` / `GPU` First-Window Selector):** Automatically detects NVIDIA (CUDA + NVENC), AMD (DirectML + AMF / Linux ROCm), Intel Arc (DirectML + QuickSync), Apple Silicon (MPS + VideoToolbox), or CPU-only setups at startup with a pre-project Compute Profile selector, hardware-accelerated `--fast-stems` (`Kim_Vocal_2.onnx` DirectML) and automatic `libx264` fallback.
- **Android & Web Studio PWA (`tiktokwizard.com/android`):** Installable mobile Progressive Web App (`docs/android/`) with offline Web Audio vocal-energy/Tap-to-Sync, Canvas2D rendering, MediaRecorder export, and a guided Termux Ubuntu `proot-distro` setup for full Python Greenfield v5 execution on ARM64 Android.
- **Greenfield Engine v5 Default (`--engine v5`):** Dual-model consensus architecture combining Wav2Vec2/MMS_FA global CTC forward-backward lattice with Echora acoustic alignment, eliminating ASR text drift across long drops, belted sustains, and repeating hooks.
- **Universal Acoustic Stems (`--hq-stems` default, `--fast-stems`):** Universal Option A defaults to high-fidelity BS-Roformer CPU stem separation (`overlap=2`) with upfront ETA calculation, chunk progress logging, and on-disk caching. Intel Arc / DirectML GPU users can invoke `--fast-stems` for ~25s vocal isolation.
- **Spanish Phonetic Digraph Normalization:** Accurate phonetic mapping for Spanish digraphs (`ll -> /ʝ/`), ensuring precise word highlighting on fast acoustic onsets.
- **Fail-Closed Anti-Fabrication Invariants:** Strict global word-level and line-level monotonicity guarantees, zero inverted time spans, and byte-for-byte UTF-8 preservation.

---

## NEW IN VERSION 5.3 (EXPLORER DEEP-LINKING & RELIABLE NAVIGATION)

- **Native Windows Shell Directory Deep-Linking:** Re-engineered project and export folder opening logic to invoke `os.startfile()` directly for directory paths, completely eliminating Windows Explorer quote parsing errors (`/select,"..."` argument fallback to `Documents`) on directory targets.
- **Precision File Explorer Selection:** Retains `/select` highlighting strictly when targeting concrete rendered video or stem files with unquoted flags (`explorer.exe /select,"<file>"`), ensuring zero-ambiguity navigation directly into the active project and stem directories.

---

## NEW IN VERSION 5.2 (ISOLATED VOCAL STEM & ACAPELLA EXPORT SUITE)

- **Isolated Vocal Stem & Acapella Export (`Vocal Stem/`):** Automatically saves high-fidelity isolated vocal tracks directly into the project folder (`START HERE/_Projects/<Project>/Vocal Stem/`):
  1. `Acapella (Full Vocals).wav` — Complete isolated vocal track with lead & backing harmonies (ideal for remixing & karaoke).
  2. `Acapella (Lead Vocals - Dry).wav` — Dry center-focused lead vocal stripped of room reverb and stereo reflections.
  3. `Acapella (Backing & Harmonies).wav` — Stereo backing vocals, ad-libs, and room acoustic echoes.
  4. `Acapella (Full Vocals).mp3` — High-quality 320kbps MP3 transcode for mobile listening and instant sharing.
  5. `Original Mix (Vocal Bandpass).wav` — Mid-channel harmonic vocal bandpass extracted directly from the original master.
- **On-Demand Project Menu Option (Option 8):** Extract and export vocal stems on demand directly from the interactive CLI project menu with one-click folder opening in File Explorer.
- **Acapella Video Rendering Mode:** Optionally render video with isolated acapella vocals as the audio track instead of the full mix.
- **Instant 0.0s Atmosphere Reuse:** Two-tier persistent caching system (`_Assets/Overlays/` + `data/fx_cache/`) eliminates redundant particle simulation.

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
