# Narration Beta: Isolated Env + Worker Integration

## Tasks
- [x] Create isolated narration envs (portable + Arc B70)
  - `DO NOT TOUCH\envs\narration` (Python 3.11.15, uv venv) in both copies
- [x] Install `f5-tts` in portable narration env and validate worker synthesis path
  - Arc B70 env present; install/validation can be mirrored there as needed
- [x] Add subprocess worker/CLI mode to narration generator (JSONL progress, job contract)
- [x] Add multi-backend narration support
  - `f5_tts`: voice cloning (best for English/Chinese, per base model support)
  - `mms_fa`: Meta MMS Persian (`facebook/mms-tts-fas`) for accurate Persian text synthesis (fixed voice)
  - `auto`: prefers MMS for Persian-script text when available
- [x] Update CLI/app integration to use narration env worker and backend selection
- [x] Sync updated narration code to Arc B70 (generator/__init__/CLI + docs compatibility)
- [x] Verify compile/import in portable and Arc B70 app envs
- [x] Generate/verify accurate Persian sample (MMS) on desktop
  - MMS base Persian sample produced and ASR-checked (content-accurate relative to F5-TTS baseline)
- [ ] End-to-end long-form beta (interrupt/resume, 15m+) in both targets with chosen backend
- [ ] Optional: integrate true cross-lingual timbre cloning/conversion for Persian (e.g. OpenVoice/VC) so reference voice timbre can be applied to accurate Persian speech

## Notes
- `f5-tts` base model is not reliable for Persian text (produced unintelligible output in test). Use `mms_fa` for accurate Persian narration content.
- MMS Persian backend records the selected reference voice in job metadata for consent/traceability but does not clone timbre.
- User goal: Persian lecture (`.txt`) narrated using `llama.mp3` voice timbre -> requires Persian-accurate base TTS + voice conversion/cloning backend; architecture is set up, VC integration pending.
