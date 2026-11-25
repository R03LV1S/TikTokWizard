NARRATION STUDIO (Voiceover from Text)
======================================

This folder is for the optional Narration / Voiceover workflow (separate from karaoke video projects and the alignment/render pipeline).

Folder layout:
- Scripts/  : Put your narration text files here (.txt). Each file is a script to convert to speech (short or long-form lectures).
- Voices/   : Put reference voice samples here (.wav/.mp3/.flac/.m4a/.ogg). Use short, clean clips (about 10-15 seconds), single speaker, minimal background noise/music.
- Output/   : Generated narration job folders and final audio are saved here.

How to use (from TikTok Wizard):
1. Launch TikTok Wizard and choose "N. Narration Studio".
2. Select "Generate new narration (long-form, chunked)".
3. Choose a text script from Scripts/ and a reference voice sample from Voices/ that you own or have explicit permission to use.
4. Choose a backend:
   - MMS Persian (recommended for Persian .txt): accurate Persian speech/content (fixed MMS voice). Selected reference voice is recorded in the job manifest for consent/traceability.
   - F5-TTS (voice cloning): clones from the reference sample; best for supported languages (e.g. English/Chinese). Not reliably accurate for Persian with the base model.
   - Auto: prefers MMS Persian for Persian-script text when available.
5. (Optional) Set long-lecture options: chunk size, pause between chunks, keep chunk files for resume/debug.
6. Confirm rights/permission, then generate narration audio.
7. If a long run is interrupted, return later and choose "Resume existing narration job" to continue from completed chunks.

Long-lecture behavior (15 minutes to multiple hours):
- Scripts are automatically split into chunks and synthesized chunk-by-chunk for stability.
- Each run creates a job folder in Output/ containing:
  - narration_job.json (progress/manifest, backend, params)
  - chunks/ (per-chunk audio files, when kept)
  - final merged narration .wav (stream-merged with bounded memory)
- Resume support: completed chunks are reused when resuming the same job (if script/voice/settings/backend remain compatible).
- Recommended defaults for long lectures: chunk size 400-900 chars, pause 0.2-0.4s, keep chunk files enabled.

Important (rights & consent):
- Only use a reference voice you own or have explicit permission to clone/use.
- Do not use this feature to clone or impersonate someone without their authorization.
- You are responsible for ensuring you have the rights to the reference voice and the generated audio, and for complying with applicable laws and platform policies.

Setup notes (backends):
- Isolated narration env (recommended): DO NOT TOUCH\envs\narration (per-feature, avoids app dependency conflicts).
- F5-TTS backend: install/configure `f5-tts` in the narration env for voice-cloning workflows (models may download on first run).
- MMS Persian backend: requires `transformers` + `torch` and access to `facebook/mms-tts-fas` (cached/downloaded on first use). Provides accurate Persian narration with a fixed voice.
- For "Persian text + specific reference voice timbre" (e.g. llama.mp3), accurate results require Persian-capable TTS plus a compatible voice-conversion/cloning stage; this VC stage may require additional setup/models beyond MMS alone.

This is an optional module and does not change the karaoke/alignment workflow.
