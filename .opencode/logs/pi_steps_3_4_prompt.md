You are Pi. Execute Steps 3-4 of the greenfield rebuild spec `tiktok_wizard.greenfield_bootstrap_and_rebuild.v4_0` for this workspace.

## MANDATORY CONSTRAINTS
- Obey filesystem_boundary_policy exactly: `START HERE/` is strictly READ-ONLY (never delete/move/rename/overwrite anything inside it; read only source audio/lyrics/assets). All teardown/edits/writes go under `DO NOT TOUCH/` (and `.opencode/` for logs).
- Use the greenfield venv Python: `DO NOT TOUCH\venv\Scripts\python.exe` (NOT system python).
- Set `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8` in the environment. Enforce `encoding='utf-8'` on every `open()` / `read_text()` / `write_text()` call so `enseñarme` never corrupts to `enseÃ±arme`.
- Use the greenfield modules in `DO NOT TOUCH/src/` (they are the clean-slate 5-step universal engine). Do NOT use any legacy `alignment_v5.py` (it is banned and must remain absent).
- The greenfield engine modules are: `src/audio/dsp.py`, `src/audio/streams.py`, `src/sync/ingestor.py`, `src/sync/line_synchronizer.py`, `src/sync/discrepancy_resolver.py`, `src/align/word_cementer.py`, `src/align/transient_snapper.py`, `src/align/integrity_verifier.py`, `src/align/pipeline.py`, `src/subtitles/ass_writer.py`, `src/subtitles/lrc_exporter.py`, `src/cli.py`, `src/domain/models.py`.
- Do NOT invoke external reviewer subagents.

## LOGGING REQUIREMENT
- Write a detailed execution log to `DO NOT TOUCH/logs/pi_steps_3_4_execution.log` using UTF-8 encoding. Log every step, command, metric, and decision. Append with timestamps. Also echo progress to stdout.

## STEP 3: CORE PIPELINE EXECUTION LAWS
Project: `START HERE/_Projects/PaoPao - Grax`
- Lyric source (read-only): `START HERE/_Projects/PaoPao - Grax/paopao_grax.txt` (95 lines, 425 words; `enseñarme` must remain exact)
- Audio source (read-only): `START HERE/_Projects/PaoPao - Grax/PaoPao - Grax (Trance Remix).mp3`
- Output dir: `DO NOT TOUCH/data/PaoPao - Grax/` (create `stems/` subdir)

Execute the 5-step universal engine:

1. **Holy grail ingest UTF-8**: Read `paopao_grax.txt` with strict UTF-8. Build `user_lyric_words[0..424]` across `lines[0..94]`. Tag all `( ... )` tokens with `is_parenthetical = True`. Compute `romanized_key = anyascii(word.text).lower()` and articulatory phoneme weights.

2. **Four-stream audio extraction**: From the MP3, generate 4 16kHz streams in `DO NOT TOUCH/data/PaoPao - Grax/stems/`:
   (a) `original_mix_vocal_band.wav` (direct from Original MP3, 220-4200Hz center-mid channel)
   (b) `acapella_full.wav` (via `audio-separator`)
   (c) `acapella_lead_dry.wav` (center azimuth < 0.28 + dereverb)
   (d) `acapella_backing_echo.wav` (side azimuth >= 0.28 + echo residual)

3. **Sentence sync and echo clone filter**: Align sentences 0..94 using `torchaudio.pipelines.MMS_FA` (`with_star=True`). Use BPM-locked spectrogram cross-correlation to route excess parenthetical echo bounces (M > K) into `<star>` wildcards so they NEVER shift subsequent lead lines. Line 31 `'A mi pai por enseñarme de distancia'` and Line 33 `'Nunca he sido de tener'` must stay in their true vocal region around 01:08 - 01:25, NEVER teleporting to 01:57.

4. **Per-sentence LUFS word cementing**: Normalize each sentence slice to -16 LUFS, run Forward-Backward `MMS_FA` `<star>`-CTC alignment, apply self-calibrated syllable merit bounds (`tau_song` * `articulatory_weight`), and hard-encode (cement) all words with `confidence >= 0.95` (`is_cemented = True`). `tau_song` must be self-calibrated (target ~0.23 sec/syllable).

5. **Recursive discrepancy resolver with original MP3**: Scan `user_lyric_words` for any uncemented gap `[prev_cemented.end_sec, next_cemented.start_sec]`. Compare `acapella_full` vs `original_mix_vocal_band` (with F0 pitch tracking) inside that exact window. Whenever the acapella stem dropped or washed out a vocal, align the user's missing words directly against `original_mix_vocal_band[t_prev:t_next]`, confirm durations match the audio energy and surrounding timestamps, cement them at >= 0.95, and repeat recursively until all 425 words are cemented.

6. **Transient snap and 1-to-1 export**: Run 2ms consonant/sibilant boundary refinement inside cemented bounds. Export:
   - `DO NOT TOUCH/data/PaoPao - Grax/karaoke.ass` (95 Dialogue lines)
   - `DO NOT TOUCH/data/PaoPao - Grax/lyrics_synced.txt` (95 lines)
   - `DO NOT TOUCH/data/PaoPao - Grax/alignment_manifest.json`

## STEP 4: MANDATORY VERIFICATION
- Run `"DO NOT TOUCH/venv/Scripts/python.exe" -m pytest -q "DO NOT TOUCH/tests/test_pipeline_invariants.py"` and confirm 8/8 pass.
- Verify `utf8_ensenarme_preserved_exactly`: Line 31 must contain `enseñarme` and NOT `enseÃ±arme`.
- Verify `total_input_lines == 95`, `total_output_ass_lines == 95`, `total_output_synced_txt_lines == 95`, `total_words == 425`.
- Verify `uncemented_words_count == 0` (all 425 words cemented).
- Capture exact content lines 17, 22, 28-36 from the new `lyrics_synced.txt`.

## REQUIRED JSON REPORT (write to `DO NOT TOUCH/data/PaoPao - Grax/verification_report.json`)
```json
{
  "alignment_v5_py_deleted_and_absent": true,
  "venv_python_executable_used": "DO NOT TOUCH/venv/Scripts/python.exe",
  "whisperx_removed_and_mms_fa_active": true,
  "start_here_folder_untouched": true,
  "pytest_new_suite_status": "<actual result>",
  "utf8_ensenarme_preserved_exactly": true,
  "total_input_lines": 95,
  "total_output_ass_lines": 95,
  "total_output_synced_txt_lines": 95,
  "total_words": 425,
  "cemented_words_ge_95_count": <int>,
  "uncemented_words_count": 0,
  "lines_17_22_and_28_to_36_from_new_lyrics_synced_txt": {"17": "...", "22": "...", "28": "...", "29": "...", "30": "...", "31": "...", "32": "...", "33": "...", "34": "...", "35": "...", "36": "..."}
}
```

Work systematically: inspect existing state, plan, implement, and verify with tests/invariant checks. Use logging/progress output throughout. Set CPU thread env vars and `torch.set_num_threads(max(1, os.cpu_count() or 4))` where applicable.