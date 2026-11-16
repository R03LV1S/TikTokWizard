# TIKTOK WIZARD v4.2

[![Release](https://img.shields.io/badge/Release-v4.2-blue.svg)](https://github.com/R03LV1S/TikTokWizard/releases)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-green.svg)](https://github.com/R03LV1S/TikTokWizard)
[![License](https://img.shields.io/badge/License-Noncommercial%20(PolyForm)-purple.svg)](https://github.com/R03LV1S/TikTokWizard)

### Automatically create TikTok, YouTube Shorts, and Reels karaoke videos with word-by-word highlighted captions timed via the Greenfield Universal Alignment Engine.

---

## GREENFIELD ADAPTIVE ACOUSTIC ENGINE (v4.2)

- **Acoustic Azimuth Echo Suppression:** Real-time stereo azimuth ratio analysis (`echo_rms / lead_rms > 1.40`) detects and suppresses reverb tails and delayed vocal bounces without clipping true vocal stutters or ad-libs.
- **Dual-Stream ASR Interleaving & Fusion:** Fuses isolated dry lead acapella and mix vocal-band Whisper transcriptions, preserving quiet backing words and fast syllables while deduplicating overlapping synonyms.
- **Lexical Rarity (IDF) & Temporal Coherence DP:** Needleman-Wunsch applies inverse document frequency weighting to unique anchor words and enforces intra-line continuity bounds, preventing repeated chorus lines from collapsing into instrumental drops.
- **Acoustic Burst Gap Interpolation:** Unanchored lines within large instrumental drops are automatically positioned around vocal RMS energy bursts rather than blindly glued to preceding lines.
- **Line-Bounded Discrepancy Resolution:** Step 6 Surround-and-Fill strictly enforces phrase-level line boundaries, preventing uncemented words from drifting across instrumental drops.
- **MMS_FA <star>-CTC Alignment:** Multi-lingual character-level forced alignment via Meta's Massively Multilingual Speech model with explicit `<star>` wildcard log-probability normalization.
- **Proportional Needleman-Wunsch:** Dynamic alignment band width scaling dynamically with track length (`abs(n - m) + 75`), supporting tracks of any duration without coordinate collapse or clipping.
- **Fresh-Audio Discrepancy Resolution:** Slices uncemented gaps and extracts fresh neural emissions with -16 LUFS normalization, preventing quiet vocals and reverb tails from being masked by 20-second chunk normalization.
- **Multi-Stream Dual Recovery:** Seamlessly switches between isolated acapella vocal stems and the original mix's filtered mid-frequency vocal band when stem isolation artifacts drop fast consonants.
- **Full UTF-8 Character Fidelity:** Guarantees 100% byte-for-byte fidelity with the user's input lyric document.

---

## ARCHITECTURE & MODULES

- `src/align/`: MMS_FA forced alignment, character emission extraction, and word cementer.
- `src/sync/`: Monotonic line synchronization, Needleman-Wunsch global alignment, discrepancy resolver.
- `src/audio/`: DSP tools, stem separation interfaces, LUFS normalization, and stream management.
- `src/pipeline/`: Pipeline orchestration, schema definitions, ASS karaoke subtitles, and FFmpeg video rendering.
- `src/domain/`: Immutable data contracts (`UserLyricDocument`, `UserLyricWord`, `AlignmentManifest`).
- `tests/`: Mathematical invariant verification test suite.
