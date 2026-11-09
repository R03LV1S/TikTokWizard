# 🪄 TIKTOK WIZARD v1.0.2 🪄

[![Release](https://img.shields.io/badge/Release-v1.0.2-blue.svg)](https://github.com/R03LV1S/TikTokWizard/releases)

[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-green.svg)](https://github.com/R03LV1S/TikTokWizard)

[![License](https://img.shields.io/badge/License-Noncommercial%20(PolyForm)-purple.svg)](https://github.com/R03LV1S/TikTokWizard)

<p align="center">
  <img src="assets/hero-banner.jpg" alt="TikTok Wizard Official" width="540" style="border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.6);" />
</p>

### Automatically create TikTok, YouTube Shorts, and Reels karaoke videos with word-by-word highlighted captions, reactive audio visualizers, and cinematic backgrounds.

---

## 🚀 NEW IN VERSION 1.0.2

- 📝 **Human-Friendly Easy Lyrics Editor (`lyrics_easy.txt`):** Never touch confusing `.ass` or `.json` subtitle files again! TikTok Wizard automatically generates an intuitive `lyrics_easy.txt` directly inside your project folder:
  ```text
  [00:18.03] Llevo (40cs) días (40cs) que (22cs) no (20cs) salgo (54cs) de (20cs) la (18cs) casa (44cs)
  [00:20.61] Que (28cs) no (20cs) salgo (52cs) de (20cs) mi (18cs) mente (56cs)
  ```
  Fix typos, change words, or tweak durations right in Notepad. TikTok Wizard automatically detects your edits and recompiles them into the video! Press **`E`** in the confirmation screen to launch the editor instantly.
- 🛡️ **Clean Project Folders (`DO NOT TOUCH/_Projects/<name>/.data`):** Relocated all internal `.data` caches and subtitle files out of `START HERE/_Projects` into protected storage, keeping user project directories tidy and free of clutter.
- 🎨 **Official Retro-Cyberpunk Wizard Art:** Integrated new official visual identity across the repository and landing page.

---

## 🚀 NEW IN VERSION 1.0.1

- 🌐 **Modern Interactive Web Studio & Documentation:** Complete redesign of the official landing page ([tiktokwizard.com](https://tiktokwizard.com)) with an interactive real-time visualizer studio (Canvas FFT spectrum, Lissajous vector orb, stereo oscilloscopes, and equalizer bars), 11 aesthetic profile showcases, and modern responsive typography (`text-wrap: balance` & `text-wrap: pretty`).
- 🛡️ **Protected Staging Directory:** Added `.Github__/` and release archive patterns to `.gitignore` to prevent accidental staging or commits of local binary bundles.

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

   - Make a new folder named after your song (e.g. `START HERE/_Projects/My Song`),

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

