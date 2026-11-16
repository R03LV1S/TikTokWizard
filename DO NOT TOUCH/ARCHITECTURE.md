# TikTokWizard Architecture (v4.2.0)

## Specification
- `spec_id`: `tiktok_wizard.architecture.v4_2`
- `version`: `4.2.0`
- `engine`: `Greenfield Adaptive Acoustic Alignment Engine (Real MMS_FA <star>-CTC)`
- `execution_mode`: `CLEAN_SLATE_REBUILD_ADDITIVE_STAGING`

## System Strata and Architectural Contracts

1. **Stage 0 - Ingest & Dual-Source Audio Preparation** (`src/sync/ingestor.py`, `src/audio/streams.py`, `src/audio/dsp.py`)
   - Parse user lyrics into `UserLyricDocument` with immutable `UserLyricWord.text` preserving exact UTF-8 character bytes and casing.
   - Decode original mix and derive `original_mix_vocal_band` (mid/center + 220-4200 Hz bandpass filter).
   - Extract acapella stems (`acapella_full`, `acapella_lead_dry`, `acapella_backing_echo`) using `audio-separator` (MDX-Net / Roformer architecture).
   - Compute short-time envelopes and acoustic features across acapella and vocal band streams.

2. **Stage 0B - Universal Multi-Script Phonetic & Articulatory Ingest** (`src/sync/ingestor.py`)
   - Preserve `word.text` byte-identical to user file.
   - Build `romanized_key` via `anyascii(...).lower()` + NFKD normalization to `[a-z0-9']+`.
   - Calculate universal articulatory phonetic weights across vowel and consonant classes.

3. **Stage 0C - Original Audio Beat & 16th-Note Subdivision Grid** (`src/audio/dsp.py`)
   - Extract tempo (BPM), beat boundaries, and adaptive 16th-note rhythmic subdivisions from instrumental references (`original_mix - acapella_full`).

4. **Step 1 - Sentence Synchronicity Validation & Monotonic Alignment** (`src/sync/line_synchronizer.py`)
   - Dual-Stream ASR Fusion: Fuses dry lead and mix vocal-band ASR words with synonym deduplication.
   - Acoustic Azimuth Echo Suppression: Real-time stereo azimuth ratio analysis (`echo_rms / lead_rms > 1.40`) suppresses reverb tails and delayed vocal bounces without clipping true vocal stutters or ad-libs.
   - Lexical Rarity (IDF) & Temporal Coherence DP: Needleman-Wunsch applies inverse document frequency weighting to unique anchor words and penalizes intra-line jumps (> 3.0s).
   - Dynamic band width `max(band, abs(n - m) + 85)` preventing reachability clipping across long tracks.
   - Enforce causal lookahead outlier rejection, locking monotonic line anchor windows and calculating dynamic `tau_song`.
   - Energy-Guided Gap Interpolation: Unanchored lines within large instrumental drops are automatically positioned around vocal RMS energy bursts.

5. **Step 2 - Per-Sentence Word Cementing via MMS_FA <star>-CTC** (`src/align/word_cementer.py`, `src/align/mms_fa.py`)
   - MMS_FA `<star>`-CTC state tracking via $2L + 1$ Viterbi forward-backward lattice.
   - Calibrated `<star>` wildcard log-probability penalty preserving sung vowel acoustic posteriors.
   - Dual-direction Viterbi convergence verification: jitter tolerance (`fb_jitter_sec <= 0.045s`), acoustic posterior confidence (`>= 0.95`).

6. **Step 3 - Discrepancy Resolver & Fresh-Audio Recovery Loop** (`src/sync/discrepancy_resolver.py`)
   - Recursively bracket uncemented gaps between confident cemented word anchors.
   - Local slice neural emission extraction (`cement_gap_fresh_audio`) with -16 LUFS normalization to prevent 20-second chunk suppression on low-energy vocal phrases.
   - Fallback recovery via `original_mix_vocal_band` when stem separation drops lead vocal transients.

7. **Step 4 - Subtitle & Manifest Export** (`src/subtitles/ass_writer.py`, `src/subtitles/lrc_exporter.py`, `src/pipeline/schema.py`)
   - Strict 1:1 line correspondence with user lyrics.
   - Advanced ASS karaoke format with centisecond `{\k<XXcs>}` word highlights.
   - Machine-readable `lyrics_synced.txt` and `lyrics_easy.txt` editors for rapid inspection.
   - JSON alignment manifest logging word-level start, end, duration, confidence, and source stream.

8. **Step 5 - Self-Healing Mathematical Invariant Verification Gate** (`src/align/integrity_verifier.py`)
   - START HERE input checksum preservation.
   - Byte-for-byte user text preservation (100% vocabulary coverage, no dropped or hallucinated words).
   - Strict temporal monotonicity (`start_sec < end_sec`, `line[i].end_sec <= line[i+1].start_sec`).
   - Adaptive duration bounds supporting long songs without artificial time cutoffs.

## Filesystem Boundaries

- `START HERE/`: **Read-only** user input area. Contains `_Projects/` (audio files, lyric text files, and rendered outputs) and OS launchers.
- `DO NOT TOUCH/`: Internal engine root. Contains the Python virtual environment (`venv/`), model weights, configuration, and engine source code.
- `DO NOT TOUCH/data/<project_slug>/`: Intermediate working directory for stems, extracted emissions, and cached alignment artifacts.
- Relocation policy: Legacy, scratch, and retired files are moved to `X:\scratch\` under the Safe Relocation Policy; no project files are permanently deleted.

## Multi-OS Runtime

- Windows: `START HERE\Double-Click to Start (Windows).bat` -> `DO NOT TOUCH\run.bat` -> `venv\Scripts\python.exe`
- macOS: `START HERE\Double-Click to Start (Mac).command` -> `DO NOT TOUCH\run.sh` -> `venv/bin/python`
- Linux: `START HERE\Double-Click to Start (Linux).sh` -> `DO NOT TOUCH\run.sh` -> `venv/bin/python`
- Zero external package dependencies outside `requirements.txt` (`torch`, `torchaudio`, `faster-whisper`, `audio-separator`, `anyascii`, `rapidfuzz`, `librosa`, `scipy`, `numpy`, `soxr`, `pytest`).
