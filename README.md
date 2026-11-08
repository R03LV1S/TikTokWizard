# 🪄 TIKTOK WIZARD v1.0.1 🪄

[![Release](https://img.shields.io/badge/Release-v1.0.1-blue.svg)](https://github.com/R03LV1S/TikTokWizard/releases)

[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-green.svg)](https://github.com/R03LV1S/TikTokWizard)

[![License](https://img.shields.io/badge/License-Noncommercial%20(PolyForm)-purple.svg)](https://github.com/R03LV1S/TikTokWizard)

### Automatically create TikTok, YouTube Shorts, and Reels karaoke videos with word-by-word highlighted captions, reactive audio visualizers, and cinematic backgrounds.

---

## 🚀 NEW IN VERSION 2.9

- 🌐 **Modern Interactive Web Studio & Documentation:** Complete redesign of the official landing page ([tiktokwizard.com](https://tiktokwizard.com)) with an interactive real-time visualizer studio (Canvas FFT spectrum, Lissajous vector orb, stereo oscilloscopes, and equalizer bars), 11 aesthetic profile showcases, and modern responsive typography (`text-wrap: balance` & `text-wrap: pretty`).
- 🛡️ **Protected Staging Directory:** Added `.Github__/` and release archive patterns to `.gitignore` to prevent accidental staging or commits of local binary bundles.

---

## 🚀 NEW IN VERSION 2.8

- 🛡️ **Non-Destructive Custom Subtitle Protection:** TikTok Wizard now automatically detects existing or hand-edited subtitle files (`.data/karaoke.ass`) before rendering! If you tweaked word timings, sync points, or styling in Notepad++, Aegisub, or VS Code, TikTok Wizard preserves your manual adjustments instead of blindly overwriting them.
- 💾 **Automated Pre-Render Subtitle Backups:** Any time subtitles are regenerated from lyrics, TikTok Wizard automatically creates a timestamped safety backup (`karaoke.ass.bak_YYYYMMDD_HHMMSS`) in `.data/`, ensuring manual edits can never be lost.
- ⚡ **Instant Skip-Alignment Workflow:** When choosing to use existing or custom subtitles, the Wizard bypasses the Whisper alignment stage entirely, jumping straight into video rendering in seconds.
- 🔀 **Review & Confirm Subtitle Toggle (`K`):** From the final confirmation screen, simply press **`K`** to toggle on-the-fly between your hand-edited subtitles and fresh Whisper AI alignment.

---

## 🚀 NEW IN VERSION 2.7

- ⚙️ **In-Script Interactive Settings Menu:** Press **`S`** at any time from the Main Menu, Profile Selection, or Render Confirmation screen to toggle **Auto-Open Folder** on/off, adjust **Video Render Quality (CRF)**, configure **Frame Rate (30/60 FPS)**, or reset preferences to defaults.
- 📂 **Automatic Output Folder Reveal:** Automatically highlights and reveals newly exported MP4 files in Windows Explorer, macOS Finder, or Linux upon render completion.

- 📂 **Automatic Output Folder Reveal:** As soon as video rendering finishes, TikTok Wizard automatically opens your system file manager (Windows Explorer, macOS Finder, Linux) with the newly exported MP4 video selected and highlighted for instant uploading or inspection.

### 🌟 Core v2.5 Features Included

- 🎨 **11 High-Impact Pre-Built Profiles:** Instant 1-click styling for any musical genre or creator style:

  - **Cyberpunk Neon:** Impact font, Electric Cyan & Hot Magenta, Constant-Q musical note spectrogram.

  - **808 Trap & Hip-Hop:** Extra large 80pt bold text in Flame Gold and White, dual peak-to-peak stereo soundwaves.

  - **80s Retro Synthwave:** Sunset Orange and Neon Purple tones with pulsing vertical frequency bars and mirror reflection.

  - **Lo-Fi Chill & Coffee:** Fraunces vintage serif in Soft Peach & Warm Cream with subtle centered line soundwave.

  - **EDM Festival Rave:** Acid Green and Cyan karaoke with spinning circular Lissajous vector scope orb.

  - **Minimalist Clean:** Modern Segoe UI with translucent pill-box highlight background and zero visual clutter.

  - **Heavy Metal & Rock:** Blood Crimson and Ghost White uppercase lyrics with high-contrast peak soundwaves.

  - **Viral Shorts Condensed:** Arial Narrow lettering for maximum words per line on mobile screens.

- 🔊 **Advanced Audio Visualizers (FFmpeg Native Filtergraphs):**

  - **Musical CQT Spectrogram (`showcqt`):** Musical note frequency visualizer rendering pitch notes across octaves.

  - **Circular Lissajous Vector Scope (`avectorscope`):** Spinning 3D-effect reactive sound orb.

  - **Dual Stereo Waveforms (`showwaves` p2p):** Mirrored stereo peak soundwaves.

  - **Frequency Equalizer Bars (`showfreqs`):** Pulsing vertical audio bars.

  - **Minimal Centered Soundwave Line (`showwaves` cline):** Clean horizontal soundwave line.

- 🔤 **Expanded Typography & Casing Transformation:**

  - Integrated fonts: *Impact*, *Arial Narrow*, *Chango*, *Fraunces*, *Segoe UI*, *Trebuchet MS*, *Arial*.

  - **UPPERCASE Headline Mode:** Punchy, high-impact all-caps text for viral mobile retention.

- 📐 **Custom Subtitle Placement & Pill Box Styles:**

  - Vertical Placement: **Top 15%** (Header), **Center-High 35%** (Clear of face/body), **Center 50%**, **Center-Low 70%** (TikTok/Reels default), **Bottom 85%** (Cinema subtitle).

  - Box Background: Modern translucent dark pill box (`border_style: 3`) for maximum legibility on busy video backgrounds.

- 🎬 **Cinematic Background Fit Modes:**

  - **Blur Pad:** 9:16 blurred background fill with sharp video centered (TikTok standard).

  - **Dark Pad:** Dimmed, contrast-enhanced dark background making center video pop with dramatic depth.

  - **Mirror Pad:** Horizontally mirrored ambient reflection.

  - **Crop Fill & Pad:** Full screen crop or classic black bars.

- 📱 **Standard Resolution Selection with Target Platforms:**

  - 1-9 menu with largest number first (e.g. `1920x1080 (9:16 Full HD)`) and explicit target platform benefits.

---

## ⚡ STEP 1: OPEN YOUR FOLDER

1. Open the folder named **`START HERE`**.

2. Click the folder matching your computer:

   - **Windows users:** Open the **`Windows`** folder.

   - **Mac users:** Open the **`Mac`** folder.

   - **Linux users:** Open the **`Linux`** folder.

3. Inside, simply **double-click**:

   👉 **`Double-Click to Start TikTok Wizard`**

*(Everything installs and sets up automatically on your first run!)*

---

## 🎵 STEP 2: HOW TO ADD YOUR OWN SONG

⚠️ **DO NOT drop loose files directly into `_Projects`!** Every song requires its own dedicated subfolder:

1. Open **`START HERE/_Projects`**.

2. **Create a NEW subfolder** for your song:

   - Either make a new folder named after your song (e.g. `START HERE/_Projects/My Song`), **OR**

   - Copy the included template folder `_Artist_-_Song-EXAMPLE ONLY` and rename it.

3. Open your new song folder and place **only your 2 files** inside:

   - **Audio file:** `song.mp3` (or `.wav`)

   - **Lyrics file:** `lyrics.txt` (formatted with 1 line per subtitle screen)

4. *(Optional)* Custom backgrounds (Videos, Pictures, GIFs) and fonts live in **`START HERE/_Projects/_Assets`**.

Now run **TikTok Wizard**! The interactive menu will automatically detect your new song folder, sync each word to the music, and export a vertical 9:16 video ready for TikTok!

---

## ⏱️ FINE-TUNING TIMESTAMPS & AD-LIB VOCALS

When TikTok Wizard runs a song for the first time, Whisper AI transcribes the words and saves a timing cache to:

```

START HERE/_Projects/<Your_Song>/.data/.whisper_cache.json

```

Subsequent runs load these timestamps **instantly** without waiting for AI re-transcription.

### When to Fine-Tune:

- **Background Vocals / Ad-libs:** When background singers chant overlapping words (e.g. `(Gracias, gracias)` or `(Hello?)`) over the lead melody.

- **Custom Pacing:** If you want words to appear earlier or hold longer.

### How to Adjust:

1. Open the hidden `.data` folder inside your song folder.

2. Open `.whisper_cache.json` in Notepad or VS Code.

3. Locate the word you want to adjust:

   ```json

   {

     "word": "Gracias,",

     "start": 50.26,

     "end": 50.90,

     "probability": 0.94

   }

   ```

4. Change the `start` and `end` times (in seconds) to match your music.

5. Save the file and re-run TikTok Wizard — it will use your edited timing immediately!

---

## 🛠️ REQUIREMENTS (ONE-TIME SETUP)

You only need **two free programs** on your computer (the wizard automatically installs them if missing):

### 1. Python 3.10+

- Download from: [python.org/downloads](https://www.python.org/downloads/)

- ⚠️ **IMPORTANT (Windows users):** When the installer opens, check the box that says **"Add Python to PATH"** before clicking Install!

### 2. FFmpeg (Free video engine)

- **Windows:** Open Command Prompt or PowerShell and run:

  ```cmd

  winget install Gyan.FFmpeg

  ```

- **Mac:** Open Terminal and run:

  ```bash

  brew install ffmpeg

  ```

- **Linux:** Open Terminal and run:

  ```bash

  sudo apt install ffmpeg

  ```

---

## 💬 COMMUNITY, SUPPORT & BUG REPORTS

Have a question, feedback, or encountered a bug?

- **Community Forum & Q&A:** [GitHub Discussions](https://github.com/R03LV1S/TikTokWizard/discussions) — Ask questions, suggest new features, or share videos created with TikTok Wizard!

- **Report a Bug:** [GitHub Issues](https://github.com/R03LV1S/TikTokWizard/issues) — Submit technical bugs, crashes, or platform-specific errors.

- **Commercial Licensing & Support:** [support@tiktokwizard.com](mailto:support@tiktokwizard.com) or [GitHub Discussions](https://github.com/R03LV1S/TikTokWizard/discussions).

---

## ❤️ SUPPORT & DONATIONS

TikTok Wizard is completely free and open-source. If this tool saves you time and helps your music or content creation, consider buying a coffee to support continued development and updates!

<p align="center">

  <a href="https://buymeacoffee.com/r03lv1s" target="_blank">

    <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" width="217" height="60">

  </a>

</p>

<p align="center">

  <a href="https://buymeacoffee.com/r03lv1s" target="_blank">

    <img src="assets/qr-coffee.png" alt="Scan to Buy Me A Coffee" width="180" height="180" style="border-radius: 12px; background: #fff; padding: 6px;">

  </a>

  <br>

  <sub><b>Scan with your phone camera or click above to donate</b></sub>

</p>

