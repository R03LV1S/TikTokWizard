#!/usr/bin/env python3
"""Alignment manifest audit - machine-readable, evidence-focused."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def audit(manifest_path: Path) -> dict:
    d = json.loads(manifest_path.read_text(encoding="utf-8"))
    words = d.get("words", [])
    total = len(words)
    target = 0.95
    cemented = [w for w in words if bool(w.get("is_cemented")) and float(w.get("confidence",0))>=target]
    unc = [w for w in words if not (bool(w.get("is_cemented")) and float(w.get("confidence",0))>=target)]
    buckets = {"post_lt_0.55":0,"post_0.55_0.70":0,"post_ge_0.70":0}
    tiers = {}
    sources = {}
    reasons = {}
    for w in unc:
        post = float(w.get("posterior",0) or 0)
        if post < 0.55: buckets["post_lt_0.55"] += 1
        elif post < 0.70: buckets["post_0.55_0.70"] += 1
        else: buckets["post_ge_0.70"] += 1
        t = str(w.get("anchor_tier") or "none")
        tiers[t] = tiers.get(t,0)+1
        s = str(w.get("confidence_source") or "unknown")
        sources[s] = sources.get(s,0)+1
    gaps_gt2=[]; short_lt120=0; floor_violations=[]
    by_line={}
    for w in words:
        li=int(w.get("line_index",0)); by_line.setdefault(li,[]).append(w)
        st=float(w.get("start_sec",0)); en=float(w.get("end_sec",st))
        if en-st < 0.12: short_lt120+=1
        post=float(w.get("posterior",0) or 0); conf=float(w.get("confidence",0) or 0); cem=bool(w.get("is_cemented"))
        if post < 0.55 and conf > target-0.05+1e-6: floor_violations.append(w.get("word"))
        if post < 0.70 and conf > target-0.01+1e-6: floor_violations.append(w.get("word"))
        if cem and conf < target-1e-6: floor_violations.append(w.get("word"))
    for li, wl in by_line.items():
        wl=sorted(wl,key=lambda x:float(x.get("start_sec",0)))
        for a,b in zip(wl,wl[1:]):
            gap=float(b.get("start_sec",0))-float(a.get("end_sec",0))
            if gap>2.0: gaps_gt2.append({"line":li+1,"from":a.get("word"),"to":b.get("word"),"gap_s":round(gap,4)})
    return {
        "manifest": str(manifest_path),
        "total_words": total,
        "cemented": len(cemented),
        "uncemented": len(unc),
        "buckets": buckets,
        "unc_tiers": tiers,
        "unc_sources": sources,
        "short_lt_120ms": short_lt120,
        "gaps_gt_2s": gaps_gt2,
        "floor_violations": floor_violations,
        "gate_pass": len(floor_violations)==0,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--json", default=None)
    args=ap.parse_args()
    r=audit(Path(args.manifest))
    txt=json.dumps(r, ensure_ascii=False, indent=2)
    print(txt)
    if args.json:
        Path(args.json).write_text(txt, encoding="utf-8")

if __name__=="__main__":
    main()
