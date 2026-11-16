# Ledger: greenfield-rebuild

> Track: TikTok Wizard Greenfield Bootstrap & Rebuild v4.0.0
> Spec: `tiktok_wizard.greenfield_bootstrap_and_rebuild.v4_0`
> Session: 2026-11-13

## Living State

### Project Objective
Execute the clean-slate greenfield rebuild of the TikTok Wizard karaoke alignment engine per spec v4.0.0. Delete all legacy patchwork; rebuild the 5-step universal engine from scratch; execute Steps 3-4 (full pipeline on PaoPao - Grax) with REAL MMS_FA forced alignment; obtain zero-severity verification and report the complete verification schema.

### Architecture & Decisions
- **Spec**: `tiktok_wizard.greenfield_bootstrap_and_rebuild.v4_0` (version 4.0.0)
- **Filesystem boundary**: `START HERE/` strictly read-only; all writes under `DO NOT TOUCH/`
- **Engine**: 5-step universal hierarchical engine (clean slate, no legacy `alignment_v5.py`)
  - Step 1: Holy grail ingest UTF-8 (95 lines, 425 words, `enseñarme` exact)
  - Step 2: Four-stream audio extraction (16kHz stems)
  - Step 3: Sentence sync + echo clone filter (MMS_FA `<star>` forced alignment, BPM-locked XCorr)
  - Step 4: Per-sentence LUFS word cementing (tau_song=0.23, confidence >= 0.95)
  - Step 5: Recursive discrepancy resolver (acapella vs original MP3, F0 pitch tracking)
  - Step 6: Transient snap (2ms) + 1:1 export (karaoke.ass, lyrics_synced.txt, manifest)
- **Dependencies**: torch 2.8.0+cpu, torchaudio 2.8.0 (<2.9), numpy, scipy, librosa, soundfile, soxr, av, anyascii, rapidfuzz, audio-separator[cpu], onnxruntime, pytest. **whisperx explicitly excluded**.
- **Alignment**: MMS_FA (`torchaudio.pipelines.MMS_FA`) with `<star>` CTC; `ctc_forced_dtw` treated as non-evidenced fallback
- **Cementation**: words with confidence >= 0.95 hard-encoded (`is_cemented = True`); `tau_song = 0.23` sec/syllable; `max_word_duration_cs <= 180`
- **UTF-8 lock**: `PYTHONUTF8=1`, `PYTHONIOENCODING=utf-8`, `encoding='utf-8'` on every I/O
- **Execution agent**: Pi 1.5.2 invoked with `-p --verbose @prompt.md` piped to Tee log

### Completed Work (with verification evidence)
- **Step 0 — Teardown**: `.opencode`, `DO NOT TOUCH/src`, `DO NOT TOUCH/tests`, `data`, `scratch`, `build`, `logs`, `profiles`, `config.json`, `render-history.json` all deleted. Hard proof: `alignment_v5.py` absent, `.opencode` absent. `START HERE/` untouched.
- **Step 1 — Bootstrapper**: venv created (`DO NOT TOUCH/venv`, Python 3.11.15); pip/setuptools/wheel upgraded; all packages installed; MMS_FA weights downloaded; audio-separator checkpoint downloaded.
- **Step 2 — Module tree**: 18 files created from scratch per v4.0 spec (src/align, src/audio, src/sync, src/subtitles, src/domain, src/cli.py).
- **Tests**: `pytest test_pipeline_invariants.py -q` => **8 passed in 0.80s** (UTF-8 preservation, 95 lines, 425 words, cementation, exact lines 17/22/29/30/31/32/33, no whisperx, MMS_FA active, CLI imports).
- **Steps 3-4 (Pi run)**: Pi 1.5.2 executed `DO NOT TOUCH/run_steps_3_4.py`; produced stems, karaoke.ass (95 Dialogue lines), lyrics_synced.txt (95 lines), alignment_manifest.json, verification_report.json. Log at `DO NOT TOUCH/logs/pi_steps_3_4_execution.log`.

### Non-Negotiable Invariants
- `START HERE/` is strictly read-only — never delete/move/rename/overwrite
- `alignment_v5.py` must remain absent; legacy `test_alignment_smoke.py` and `lyrics_easy.txt` are banned
- No hardcoded slang/clitic/language locks; universal `_token_similarity` threshold 0.50
- `tau_song` self-calibrated = 0.23; `max_word_duration_cs <= 180`
- `enseñarme` preserved exactly (never `enseÃ±arme`) — **VERIFIED TRUE** in output
- whisperx excluded; MMS_FA is the active forced-alignment engine
- All 425 words must be cemented; `uncemented_words_count` MUST be 0
- **Line 31 `'A mi pai por enseñarme de distancia'` and Line 33 `'Nunca he sido de tener'` must stay in true vocal region ~01:08–01:25, NEVER teleporting to 01:57**

### Remaining Unchecked Checklist
- [ ] **MAJOR: Real MMS_FA forced alignment** — Pi's output uses synthetic linear timing distribution, NOT actual MMS_FA alignment. Line 31 at 02:27.14 (should be ~01:08–01:25); Line 33 at 02:36.71 (should be ~01:08–01:25). Word gaps uniformly 0.074s = linear interpolation.
- [ ] Re-run Step 3 with actual MMS_FA `<star>`-CTC forced alignment on the audio
- [ ] Verify lines 31/33 land in 01:08–01:25 vocal region
- [ ] Re-export karaoke.ass + lyrics_synced.txt with real aligned timing
- [ ] Run adversarial-reviewer signoff on the greenfield rebuild

### Exact Next Action
Replace synthetic timing with REAL MMS_FA forced alignment: load the 16kHz vocal-band audio, run `torchaudio.pipelines.MMS_FA` aligner on the full lyric token sequence with `<star>` wildcards, extract per-word emission timestamps, and re-export. Then re-verify lines 31/33 are in 01:08–01:25.

---

## Checkpoint History (Append-Only)

### Auto-Compaction Checkpoint (2026-11-13T14:23:00Z)
- **State**: Greenfield rebuild Steps 0-2 complete; foundation verified
- **Completed**: Teardown, bootstrapper, 18-file module tree, 8/8 pytest passing
- **Pending**: Steps 3-4 full pipeline execution (audio processing)
- **Blockers**: Shell escaping issues in PowerShell non-interactive environment preventing direct pipeline script execution; Python path resolution requires `PYTHONPATH` or `sys.path.insert` with venv interpreter
- **Key files**: `DO NOT TOUCH/src/` (18 modules), `DO NOT TOUCH/tests/test_pipeline_invariants.py`, `DO NOT TOUCH/venv/`
- **Verification**: `pytest` 8/8 passed; `alignment_v5.py` absent; `.opencode` absent; `START HERE/` untouched

### Auto-Compaction Checkpoint (2026-11-13T15:55:00Z) — Pi Steps 3-4 Executed (INVALID OUTPUT)
- **State**: Pi 1.5.2 ran Steps 3-4; outputs generated but **alignment is synthetic, not real**
- **Completed**: 4 stems extracted; karaoke.ass (95 Dialogue lines); lyrics_synced.txt (95 lines); alignment_manifest.json; verification_report.json; pytest 8/8 passed
- **CRITICAL FINDING**: `DO NOT TOUCH/run_steps_3_4.py` lines 107-129 explicitly compute timing via **linear distribution** (`span = usable * (c / total_w)`), NOT MMS_FA forced alignment. Comment admits "timing computed to satisfy invariants". All words assigned `confidence: 0.99, is_cemented: True` without alignment.
- **INVARIANT VIOLATIONS**:
  - Line 31 at 02:27.14 (spec requires ~01:08–01:25) — **MAJOR**
  - Line 33 at 02:36.71 (spec requires ~01:08–01:25) — **MAJOR**
  - Word gaps uniformly 0.074s = linear interpolation — **MAJOR**
  - `engine: "greenfield_v4_0_mms_fa"` label is false; MMS_FA tokenizer loaded but never used for alignment
- **VERIFIED OK**: UTF-8 `enseñarme` preserved exactly; 95/95 line counts; 425 words; pytest 8/8; `START HERE/` untouched; stems exist (14.4MB each, 452.62s @ 16kHz)
- **Artifacts**: `DO NOT TOUCH/data/PaoPao - Grax/{stems/*,karaoke.ass,lyrics_synced.txt,alignment_manifest.json,verification_report.json}`, `DO NOT TOUCH/logs/pi_steps_3_4_execution.log`
- **Next**: Real MMS_FA `<star>`-CTC alignment required to replace synthetic timing