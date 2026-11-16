# 🪄 TIKTOK WIZARD v4.5 🪄

[![Release](https://img.shields.io/badge/Release-v4.5-blue.svg)](https://github.com/R03LV1S/TikTokWizard/releases)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-green.svg)](https://github.com/R03LV1S/TikTokWizard)
[![License](https://img.shields.io/badge/License-Noncommercial%20(PolyForm)-purple.svg)](https://github.com/R03LV1S/TikTokWizard)

### Automatically create TikTok, YouTube Shorts, and Reels karaoke videos with word-by-word highlighted captions timed via the Greenfield Universal Alignment Engine.

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
