
from __future__ import annotations
from typing import Any, Dict, List
ENGINE_V5 = "greenfield_v5_dual_ctc_consensus"
ENGINE_V2 = ENGINE_V5  # Backwards-compatible alias
ENGINE_VERSION = "5.9"
SCHEMA_VERSION = "alignment_manifest.v5_compat.v1"

def validate_manifest(m: Dict[str, Any], word_target: float = 0.95) -> List[str]:
    errs = []
    words = m.get("words", [])
    lines = m.get("lines", [])
    if m.get("total_words") != len(words): errs.append("total_words_mismatch")
    if m.get("total_input_lines") != len(lines): errs.append("total_input_lines_mismatch")
    cem = sum(1 for w in words if bool(w.get("is_cemented")) and float(w.get("confidence",0))>=word_target)
    unc = len(words)-cem
    if m.get("cemented_words_ge_95_count") != cem: errs.append("cemented_count_mismatch")
    if m.get("uncemented_words_count") != unc: errs.append("uncemented_count_mismatch")
    prev_end = -1.0
    for i,w in enumerate(words):
        st=float(w.get("start_sec",0)); en=float(w.get("end_sec",0)); post=float(w.get("posterior",0) or 0); conf=float(w.get("confidence",0) or 0)
        if not (0.0 <= st < en <= float(m.get("total_duration_sec", en))+1e-3): errs.append(f"bounds:{i}")
        if st < prev_end - 1e-3: errs.append(f"mono:{i}")
        if bool(w.get("is_cemented")) and conf < word_target - 1e-6: errs.append(f"cement_conf:{i}")
        if post < 0.55 and conf > word_target - 0.05 + 1e-6: errs.append(f"floor55:{i}")
        if post < 0.70 and conf > word_target - 0.01 + 1e-6: errs.append(f"floor70:{i}")
        prev_end = en
    return errs
