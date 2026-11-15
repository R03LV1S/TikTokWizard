# 🪄 TIKTOK WIZARD v4.4 🪄

[![Release](https://img.shields.io/badge/Release-v4.4-blue.svg)](https://github.com/R03LV1S/TikTokWizard/releases)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-green.svg)](https://github.com/R03LV1S/TikTokWizard)
[![License](https://img.shields.io/badge/License-Noncommercial%20(PolyForm)-purple.svg)](https://github.com/R03LV1S/TikTokWizard)

### Automatically create TikTok, YouTube Shorts, and Reels karaoke videos with word-by-word highlighted captions timed via the Greenfield Universal Alignment Engine.

---

## NEW IN VERSION 4.4 (VOLT RADIAL FLOWER REACTIVITY & PIPELINE FUSION)

- **Beat-Synchronized Volt Radial Flower:** Visualizer 7 (`volt_radial`) now dynamically pulses with beat envelopes and frequency sweeps (0 to 180px radial bloom) rather than hovering statically.
- **Pipeline Render Integration:** Fully integrated `volt_radial` into `src/pipeline/render.py` for automated video generation.

---

## VERSION 4.3 (REACTIVE AUDIO VISUALIZERS & DEFAULT SPECTRUM BARS)

- **Default Multi-Band Spectrum Bars:** Visualizer 5 (`spectrum_bars`) is now the default visualizer across all interactive profiles and automated renders, resting flush on top of the TikTok safety pad.
- **Dynamic Floor Subtraction & Power Expansion:** Eliminated static hovering visualizers by implementing per-band adaptive floor tracking, transient beat modulation, and 1.6 power expansion for punchy bouncing on kicks and synths.
- **Modern Open-Source Engine Integrations:** Integrated high-performance reactive visualizer engines with support for multi-band frequency bars, radial bass pulse, mirrored stereo, particle dots, neon ribbon waves, and radial rings.

---

## VERSION 4.2 (ADAPTIVE ACOUSTIC ENGINE)

- **Acoustic Azimuth Echo Suppression:** Real-time stereo azimuth ratio analysis (`echo_rms / lead_rms > 1.40`) detects and suppresses reverb tails and delayed vocal bounces without clipping true vocal stutters or ad-libs.
- **Dual-Stream ASR Interleaving & Fusion:** Fuses isolated dry lead acapella and mix vocal-band Whisper transcriptions, preserving quiet backing words and fast syllables while deduplicating overlapping synonyms.
- **Lexical Rarity (IDF) & Temporal Coherence DP:** Needleman-Wunsch now applies inverse document frequency weighting to unique anchor words and enforces intra-line continuity bounds, preventing repeated chorus lines from collapsing into instrumental drops.
- **Acoustic Burst Gap Interpolation:** Unanchored lines within large instrumental drops are automatically positioned around vocal RMS energy bursts rather than blindly glued to preceding lines.
- **Line-Bounded Discrepancy Resolution:** Step 6 Surround-and-Fill now strictly enforces phrase-level line boundaries, preventing uncemented words from drifting across instrumental drops.
- **MMS_FA <star>-CTC Forced Alignment:** Native multi-lingual character-level alignment via Meta's Massively Multilingual Speech model with explicit `<star>` wildcard log-probability normalization. Completely removes reliance on legacy WhisperX.
- **Proportional Diagonal Needleman-Wunsch:** Adaptive alignment band width scales dynamically with track length (`abs(n - m) + 75`), supporting tracks of any duration (from short hooks to multi-hour sets) without coordinate collapse or clipping.
- **Fresh-Audio Discrepancy Resolution:** Automatically slices uncemented gaps and extracts fresh neural emissions with -16 LUFS normalization, preventing quiet vocals and reverb tails from being masked by 20-second chunk normalization.
- **Multi-Stream Dual Recovery:** Seamlessly switches between isolated acapella vocal stems and the original mix's filtered mid-frequency vocal band when stem isolation artifacts drop fast consonants.
- **Full UTF-8 Character Fidelity:** Guarantees 100% byte-for-byte fidelity with the user's input lyric document (accents, multi-lingual characters, punctuation, and line breaks are fully preserved).
- **Streamlined Dependency Architecture:** Modernized runtime using `torch`, `torchaudio`, `faster-whisper`, `audio-separator`, `anyascii`, and `rapidfuzz`.

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
    src/                      <- Greenfield Universal Engine v4.1 source
    tests/                    <- Invariant verification test suite
```
