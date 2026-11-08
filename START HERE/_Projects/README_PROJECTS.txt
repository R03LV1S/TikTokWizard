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

WORKSPACE CLEANLINESS:
----------------------
- Only your music file, lyrics text, output video (.mp4), and optional `lyrics_easy.txt` live in your folder.
- All internal caches and metadata (.whisper_cache.json, karaoke.ass) are safely kept inside
  `DO NOT TOUCH/_Projects/<Your_Song>/` automatically, keeping your workspace completely clean and clutter-free!
- Existing renders are NEVER overwritten: each render increments versions (v1, v2, v3).


================================================================================
     EASY LYRICS EDITOR — EDIT WORDS & TIMING WITHOUT COMPLICATED CODE!
================================================================================

TikTok Wizard automatically generates an easy-to-read `lyrics_easy.txt` directly inside
your song folder after alignment:

    [00:18.03] Llevo (40cs) días (40cs) que (22cs) no (20cs) salgo (54cs) de (20cs) la (18cs) casa (44cs)
    [00:20.61] Que (28cs) no (20cs) salgo (52cs) de (20cs) mi (18cs) mente (56cs)

HOW TO EDIT:
------------
1. Simply double-click `lyrics_easy.txt` to open it in Notepad (or press "E" in the TikTok Wizard render review screen).
2. Fix any word typos, add missing words, or tweak the start timestamps [MM:SS.cc].
3. Optional: (XXcs) after any word sets its sung highlight duration (100cs = 1.0 second).
   If you add new words without (XXcs), TikTok Wizard automatically balances durations smoothly!
4. Save the file and render: TikTok Wizard automatically compiles your changes directly into the video!
================================================================================
