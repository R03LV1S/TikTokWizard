================================================================================
                    HOW TO USE THE _PROJECTS FOLDER
================================================================================

Every song video project gets its own folder here!

QUICK START (3 STEPS):
----------------------
1. CLONE THE TEMPLATE:
   - Copy the folder `_Artist_-_Song-EXAMPLE ONLY` (Ctrl + C, then Ctrl + V).
   - Rename the copied folder to your song's name:
     e.g. "Bad Bunny - DtMF"

2. PUT YOUR 2 FILES INSIDE:
   - Audio file (.mp3, .wav, or .m4a)
   - Lyrics file (.txt) formatted with 1 line per subtitle display

3. RUN TIKTOK WIZARD:
   - Double-click the "Double-Click to Start TikTok Wizard" in `_START HERE/Windows/`
     (or Mac/Linux equivalent).
   - Pick your song from the interactive menu.
   - Choose your resolution (e.g. 1080x1920 TikTok or custom) and render!

FOLDER RULES:
-------------
- Only your music file, lyrics text, and output video (.mp4) live in your folder.
- All internal caches (.whisper_cache.json, karaoke.ass) are placed inside `.data/`
  automatically so your project stays clean.
- Existing renders are NEVER overwritten: each render increments versions (v1, v2, v3).


================================================================================
  HOW TO FINE-TUNE TIMESTAMPS & FIX BACKGROUND / AD-LIB VOCALS
================================================================================

When you run a song for the first time, TikTok Wizard uses Whisper AI to extract
precise word-level start and end timestamps. These timestamps are cached in:
   `_Projects\<Your_Song_Folder>\.data\.whisper_cache.json`

Because it is cached, you only have to transcribe once! Subsequent runs load instantly.

WHEN AND WHY TO FINE-TUNE:
--------------------------
1. LOUD BACKGROUND VOCALS & AD-LIBS:
   When background singers chant words (e.g. "(Gracias, gracias)" or "(Hello?)")
   while the lead singer is still singing, or when echoes/reverb repeat a phrase,
   AI models transcribe all words sequentially on a single timeline.
   
2. CUSTOM PACING:
   If you want a specific word to stay highlighted longer or appear earlier.

HOW TO EDIT TIMESTAMPS:
-----------------------
1. Open your project's hidden `.data` folder:
   `_Projects\<Your_Song_Folder>\.data\`
   *(Tip on Windows: If you don't see `.data`, enable "Hidden items" in File Explorer View)*

2. Open `.whisper_cache.json` in Notepad or VS Code.

3. Find the line or word you want to adjust. Each word entry looks like:
   {
     "word": "Gracias,",
     "start": 50.26,
     "end": 50.90,
     "probability": 0.94
   }

4. Adjust the `start` and `end` values (in seconds) to match your music playback:
   - `start`: exact second the word starts lighting up.
   - `end`: exact second the word finishes highlighting.

5. Save the file.

6. Re-run TikTok Wizard! The wizard will immediately detect your edited timestamps
   without re-transcribing, producing your custom perfected timing.
================================================================================
