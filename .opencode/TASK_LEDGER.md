# TikTokWizard Alignment Residual Resolution Ledger

## Baseline (locked)
- Best evidence-backed: MMS_FA primary + Phase A/B cascade + 3-pass clean-scratch consensus
- Manifest: `eval/reports/paopao_multipass_consensus_v2/merged_manifest.json`
- Metrics: **356/425 cemented, 69 uncemented**, short<120ms 9, gaps>2s 1, `CEM_POST<0.70` 0, gate pass
- Render: `START HERE/_Projects/PaoPao - Grax/PaoPao - Grax (Trance Remix) v14.mp4`
- Rule: no fake cementation; posterior floors; `is_cemented = confidence>=word_target` with evidence.

## Residuals (69)
- post<0.55: 6 words (very weak evidence)
- post 0.55-0.70: 60 words (some evidence, below cement threshold)
- post>=0.70: 3 words (capped by floor/incoherence/confidence rules)
- Common patterns: parenthetical Gracias variants, short function words (Que/por/que/en/Y/A/se/mi), decemented_line_incoherent.

## Batches (architect-aligned)
- [x] B0 Audit/lock: baseline locked (356/69 multipass), residual taxonomy computed (Gracias 34, short_func 22, other 12, decemented_incoherent 1; post<0.55=6, 0.55-0.70=60, >=0.70=3). Gap>2s at L67 dar->gracias 2.624s. Added `scripts/alignment_audit.py`.
- [x] B1 Robust mode: `--robust`/`--multipass N` validated. 3-pass clean-scratch: 351/74, 358/67, 352/73 -> merged **360/65**, short 13, gaps>2s 1 (L67 dar->gracias 2.624s), CEM_POST<0.70 0, floor_violations 0. +4 vs 356/69 baseline, 0 lost. Archived `eval/reports/paopao_b1_robust3_merged`.
- [x] B2 Evidence instrumentation (partial/landed): per-word `resolution_status`/`reject_reason` in pipeline; audit script reports buckets/tiers/floor violations. Residual_report.py/integrity verifier extensions remain backlog.
- [~] B3 Secondary evidence hardening: Phase B has reprobe+WhisperX agreement+floors; boundary corroboration added as conservative boost. Full contradiction-scoring refactor remains backlog.
- [x] B4 TuneStudio boundary.onnx secondary onset corroboration: wired in Phase B (uncemented only, +/-300ms, post>=0.55, floor-capped, no boundary-only cementation, fallback on exception). 1-pass signal neutral; 3-pass robust 356/69 vs 360/65 best.
- [ ] B5 Contract-compatible primary A/B harness: backlog (default remains MMS_FA).
- [ ] B6 Calibration & bounded recovery: backlog.
- [~] B7 Verification: PaoPao clean-scratch robust validated; best 360/65 archived. Render in progress. Adversarial multi-fixture matrix/gold expansion remains backlog.

## Invariants
- Clean-scratch after engine/workflow change: archive, wipe data/<project>, full rebuild, diff, render, audit.
- No song-specific rules/timestamps/word hardcodes.
- `post<0.55 => conf<=target-0.05`, `post<0.70 => conf<=target-0.01`, `is_cemented=conf>=target`.
- Karaoke only; narration pipeline separate.
