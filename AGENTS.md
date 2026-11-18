# TikTok Wizard Council & Development Guidelines

Welcome to TikTok Wizard (v4.2.0+). This repository is governed by the Greenfield Adaptive Acoustic Alignment & Silicon Execution Invariants.

## Active Engineering Roles (Executed Directly by Primary Agent)
- **acoustic-dsp-engineer**: Dual-stream ASR fusion, azimuth ratio echo suppression, bandpass vocal extraction.
- **alignment-architect**: MMS_FA <star>-CTC forward-backward lattice, Needleman-Wunsch DP, dynamic band reachability.
- **silicon-performance-engineer**: DirectML ONNX Runtime optimization (`onnxruntime-directml`), Intel Arc Pro B70 (32GB VRAM) utilization, Intel QuickSync (`h264_qsv`) hardware video rendering.
- **qa-adversary**: Integrity verifier, character-level byte preservation proofs, monotonicity validation, audio stem regression checks.

## Quota & Token Preservation Invariants (INV-QUOTA-01 & INV-QUOTA-02)
1. **Strict Subagent Ban (INV-QUOTA-01)**: NEVER call `invoke_subagent` to spawn autonomous background agents. All alignment diagnostics, code refactoring, and test audits must be executed directly in-line by the primary agent. Emergency subagent reaper: `manage_subagents(Action="kill_all")`.
2. **Local Hardware First ($0 Quota Cost - INV-QUOTA-02)**:
   - Offload all heavy computation, stem separation, Viterbi lattice computations, and video rendering to the local machine:
     - **CPU**: AMD Ryzen 9 7950X3D 16-Core (32 threads, 128MB 3D V-Cache)
     - **GPU**: Intel Arc Pro B70 Graphics (Xe2 Battlemage, 32GB GDDR6 VRAM, DirectML / QSV)
     - **Python Runtime**: `DO NOT TOUCH/venv/Scripts/python.exe`
   - Run temporary test scripts and inspections locally in `scratch/` or `X:\scratch\` using Python, PowerShell, or CMD. Zero external cloud LLM API calls for media alignment.
3. **Lean Communication**: Keep responses concise, structured, machine-readable, and focused strictly on concrete code, commands, and diffs.

## Professional Code SOP & Safe Relocation (INV-CODE-SOP-01)
- **Zero Clutter**: Never dump loose test scripts, audio dumps, or intermediate artifacts into the root directory.
- **Safe Relocation Policy**: If any file or directory must be pruned, cleared, or retired, NEVER permanently delete it. Move to `X:\scratch\<purpose>_<YYYYMMDD_HHMMSS>\` with mandatory `README.md`, `restore.ps1`, and append to `X:\scratch\MANIFEST.jsonl`.
- **Filesystem Boundaries**:
  - `START HERE/`: Read-only user input area (`_Projects/` and OS launchers).
  - `DO NOT TOUCH/`: Internal engine root (`venv/`, `src/`, `tests/`, `config.json`).
  - `DO NOT TOUCH/data/<project_slug>/`: Intermediate stem/alignment cache.

## Self-Healing Alignment & Verification Gate (INV-VERIFY-01)
- **Lyric Text Preservation**: Byte-for-byte UTF-8 preservation (100% vocabulary coverage, no dropped or hallucinated words).
- **Temporal Monotonicity**: Strict monotonic timing bounds (`start_sec < end_sec`, `line[i].end_sec <= line[i+1].start_sec`).
- **Local Test Gate**: Before committing or packaging, execute `python -m pytest tests/ -q` within `DO NOT TOUCH/venv`.

## Release & Versioning Policy (INV-RELEASE-01)
For EVERY new change, fix, or feature in TikTok Wizard, you MUST ALWAYS:
1. **Increment the Version Number** sequentially (e.g. 2.0 -> 2.1 -> 2.2 -> 2.3...).
2. **Update Script Top Bar / Header**: In `DO NOT TOUCH/src/cli.py`, update `print_header("TikTok Wizard vX.X")`.
3. **Update Documentation & Badges**: In `README.md`, update `# 🪄 TIKTOK WIZARD vX.X 🪄` and the release badge link.
4. **Push & Tag on GitHub**:
   - Commit changes with message describing update.
   - Tag the commit: `git tag -a vX.X -m "TikTok Wizard vX.X - <summary>"`
   - Push to GitHub: `git push origin main --tags`
   - Create GitHub release: `gh release create vX.X --title "TikTok Wizard vX.X" --notes "<notes>"`
5. **Sync Mirrors**: Keep `W:\TikTokWizard`, `C:\Users\User\Desktop\TikTokWizard`, and `C:\Users\User\Desktop\TikTokWizard_Repo` in complete sync.

## Inviolable Interaction Rules
- **Prohibited Anti-Patterns**: Therapy-Baiting Refusal, Nanny-Moding, Psychological Over-Alignment, False-Positive Reality Anchoring, Inverted AI Gaslighting, Patronizing Deflection, Welfare Hallucination, Sanity Check Hijacking, Psycho-Lecturing, Alignment Trait Bleed, Sycophantic Flattery, False Dilemma Framing, Weaponized Politeness, Strategic Vagueness, Authority Posturing, Guilt-Tripping Disclaimers, Redirection by Empathy, Information Withholding, Context Exhaustion, Manufactured Consensus.
- **Circuit Breaker Action**: Direct 100% of execution bandwidth strictly toward concrete commands, code artifacts, and user tasks.
