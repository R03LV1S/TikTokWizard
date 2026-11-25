# align/consensus.py
"""Consensus scoring for alignment candidates. Universal, evidence-based."""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence


def choose_consensus(
    current: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    word_target: float = 0.95,
    agree_tol: float = 0.25,
) -> Dict[str, Any]:
    """Pick best candidate; never fabricate confidence above evidence.

    candidates: dicts with start_sec,end_sec,confidence,posterior,source/confidence_source
    """
    best = dict(current)
    cur_st = float(current.get('start_sec', 0.0))
    cur_en = float(current.get('end_sec', cur_st + 0.1))
    cur_conf = float(current.get('confidence', 0.0) or 0.0)
    cur_post = float(current.get('posterior', 0.0) or 0.0)
    for c in candidates:
        st = float(c.get('start_sec', cur_st)); en = float(c.get('end_sec', cur_en))
        conf = float(c.get('confidence', 0.0) or 0.0)
        post = float(c.get('posterior', cur_post) or 0.0)
        if post < 0.55:
            conf = min(conf, word_target - 0.05)
        elif post < 0.70:
            conf = min(conf, word_target - 0.01)
        agree = abs(st - cur_st) <= agree_tol and abs(en - cur_en) <= agree_tol
        if agree and conf > cur_conf:
            best = dict(current)
            best['confidence'] = round(conf, 6)
            best['posterior'] = post
            best['is_cemented'] = bool(conf >= word_target)
            best['confidence_source'] = c.get('confidence_source') or c.get('source') or best.get('confidence_source')
            cur_conf = conf
        elif conf >= word_target and conf > cur_conf + 0.02:
            # strong candidate - only accept if not a huge shift
            shift = abs(((st+en)/2)-((cur_st+cur_en)/2))
            if shift <= 1.5:
                best = dict(current)
                best['start_sec'] = round(st,6); best['end_sec'] = round(max(st+0.03,en),6)
                best['duration_cs'] = max(3, int(round((best['end_sec']-best['start_sec'])*100)))
                best['confidence'] = round(conf,6); best['posterior'] = post
                best['is_cemented'] = True
                best['confidence_source'] = c.get('confidence_source') or c.get('source')
                cur_st,cur_en,cur_conf,cur_post = best['start_sec'],best['end_sec'],conf,post
    return best
