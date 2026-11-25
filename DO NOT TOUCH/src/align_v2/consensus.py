
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass(frozen=True)
class ConsensusPolicy:
    word_target: float = 0.95
    agree_tol_sec: float = 0.20
    min_posterior_cement: float = 0.55
    min_null_llr_cement: float = -0.5

def merge_model_candidates(base_words: List[Dict[str, Any]], model_outputs: Dict[str, List[Dict[str, Any]]], *, policy: ConsensusPolicy) -> List[Dict[str, Any]]:
    n = len(base_words)
    out = []
    for i in range(n):
        bw = dict(base_words[i])
        cands = []
        for src, lst in model_outputs.items():
            if i < len(lst): cands.append((src, lst[i]))
        if not cands:
            bw["is_cemented"]=False; bw["anchor_tier"]="placement_unverified"; out.append(bw); continue
        # pick cluster: agreement on midpoint within tol
        best=None; best_score=(-1,-1)
        for src, c in cands:
            cluster=[(s,x) for s,x in cands if abs(((x["start_sec"]+x["end_sec"])/2)-((c["start_sec"]+c["end_sec"])/2)) <= policy.agree_tol_sec]
            score=(len(cluster), sum(float(x.get("confidence",0)) for _,x in cluster))
            if score>best_score: best_score=score; best=cluster
        cluster=best or [cands[0]]
        st=sum(float(x["start_sec"]) for _,x in cluster)/len(cluster)
        en=sum(float(x["end_sec"]) for _,x in cluster)/len(cluster)
        en=max(st+0.02,en)
        conf=sum(float(x.get("confidence",0)) for _,x in cluster)/len(cluster)
        post=sum(float(x.get("posterior",0)) for _,x in cluster)/len(cluster)
        llr=sum(float(x.get("null_llr",0)) for _,x in cluster)/len(cluster)
        agree=len(cluster) >= max(2, len(model_outputs))  # require all available models when 2
        contradict = any(abs(((x["start_sec"]+x["end_sec"])/2)-((st+en)/2)) > 0.40 and float(x.get("confidence",0))>=0.55 for _,x in cands if (x not in [y for _,y in cluster]))

        c_mms = next((x for s, x in cands if s == "mms_fa"), None)
        c_ech = next((x for s, x in cands if s == "echora_fixed"), None)
        tight_agree = False
        best_post = post
        best_llr = llr
        if c_mms is not None and c_ech is not None:
            d_st = abs(float(c_mms["start_sec"]) - float(c_ech["start_sec"]))
            d_en = abs(float(c_mms["end_sec"]) - float(c_ech["end_sec"]))
            if d_st <= 0.15 and d_en <= 0.20:
                tight_agree = True
                mms_p = float(c_mms.get("posterior", 0))
                ech_p = float(c_ech.get("posterior", 0))
                best_post = max(mms_p, ech_p)
                best_llr = max(float(c_mms.get("null_llr", 0)), float(c_ech.get("null_llr", 0)))
                # Evaluate calibrated conf using best_post if higher
                if best_post > post:
                    est_conf = 1.0 / (1.0 + math.exp(-(-1.8 + 2.8 * best_post + 0.35 * max(-2.0, min(4.0, best_llr)) + 1.2)))
                    conf = max(conf, est_conf)

        reason = None
        if contradict or not agree:
            conf = min(conf, policy.word_target - 0.01)
            if contradict: reason = "contradiction"
            elif not agree and len(model_outputs) < 2: reason = "single_model_no_consensus"
            elif not agree: reason = "no_model_agreement"
        elif tight_agree and not contradict:
            vad_ratio = min(float(c_mms.get("vocal_vad_ratio", 1.0)), float(c_ech.get("vocal_vad_ratio", 1.0)))
            if best_post < 0.35:
                conf = min(conf, policy.word_target - 0.05)
                reason = "below_posterior_floor"
            elif best_post < 0.50:
                conf = min(conf, policy.word_target - 0.01)
                reason = "posterior_below_0.50"
            elif best_llr < 0.50:
                conf = min(conf, policy.word_target - 0.01)
                reason = "below_null_llr_floor"
            elif vad_ratio < 0.30:
                conf = min(conf, policy.word_target - 0.01)
                reason = "low_vocal_vad_ratio"
        else:
            if post < policy.min_posterior_cement:
                conf = min(conf, policy.word_target - 0.05)
                reason = "below_posterior_floor"
            elif llr < policy.min_null_llr_cement:
                conf = min(conf, policy.word_target - 0.01)
                reason = "below_null_llr_floor"
            elif post < 0.70:
                conf = min(conf, policy.word_target - 0.01)
                reason = "posterior_below_0.70"

        cement = conf >= policy.word_target and not contradict and agree
        if not cement and reason is None:
            reason = "below_target_confidence"
        bw.update({
            "start_sec":round(st,4),"end_sec":round(en,4),"duration_cs":max(2,int(round((en-st)*100))),
            "posterior":round(post,6),"null_llr":round(llr,4),"confidence":round(conf,6),
            "is_cemented":bool(cement),
            "anchor_tier":"cemented_consensus" if cement else "placement_unverified",
            "confidence_source": "+".join(sorted(set(s for s,_ in cluster)))+"+consensus",
            "immutable_anchor":bool(cement),
            "model_posteriors": {s: float(x.get("posterior",0)) for s,x in cands},
            "contradicting_sources": [s for s,x in cands if abs(((x["start_sec"]+x["end_sec"])/2)-((st+en)/2)) > 0.40 and float(x.get("confidence",0))>=0.55],
            "reject_reason": None if cement else reason,
        })
        if bw["contradicting_sources"]:
            bw["is_cemented"]=False; bw["immutable_anchor"]=False; bw["confidence"]=min(float(bw["confidence"]), policy.word_target-0.01); bw["reject_reason"]="contradiction"
        out.append(bw)
    # monotonic placement for uncemented (do not move immutable anchors)
    prev=0.0
    for i,w in enumerate(out):
        nxt = float(out[i+1]["start_sec"]) if i+1<len(out) else 99999.0
        if w.get("immutable_anchor") and w.get("is_cemented"):
            w["start_sec"]=max(prev, float(w["start_sec"])); w["end_sec"]=max(w["start_sec"]+0.02,float(w["end_sec"])); prev=float(w["end_sec"]); continue
        st=max(prev,float(w["start_sec"])); en=min(nxt,max(st+0.02,float(w["end_sec"])))
        if en-st < 0.02: en=min(nxt, st+0.02)
        w["start_sec"]=round(st,4); w["end_sec"]=round(max(st+0.02,en),4); w["duration_cs"]=max(2,int(round((w["end_sec"]-w["start_sec"])*100)))
        w["is_cemented"]=bool(float(w["confidence"])>=policy.word_target and w.get("immutable_anchor"))
        prev=float(w["end_sec"])
    return out
