
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import numpy as np
import torch

NEG_INF = -1e30

@dataclass
class WordPosterior:
    start_frame: int
    end_frame: int
    posterior: float
    null_llr: float
    confidence: float
    marginal_mass: float = 0.0

def build_global_star_targets(word_token_ids: Sequence[Sequence[int]], star_id:int, blank_id:int):
    # [*, w1, *, w2, *, ..., *]
    targets=[int(star_id)]
    spans=[]
    for i, toks in enumerate(word_token_ids):
        s=len(targets)
        for t in toks:
            t=int(t)
            if t==blank_id: continue
            targets.append(t)
        if len(targets)==s:
            targets.append(int(blank_id)+1 if int(blank_id)+1!=int(blank_id) else 1)
        e=len(targets)
        spans.append((s,e))
        if i < len(word_token_ids)-1:
            targets.append(int(star_id))
    targets.append(int(star_id))
    return targets, spans

def _viterbi_path(
    log_probs: np.ndarray,
    targets: Sequence[int],
    blank_id: int,
    is_intra_word_blank: np.ndarray | None = None,
    intra_blank_penalty: float = 0.30,
    is_leading_blank_or_star: np.ndarray | None = None,
    leading_vocal_penalty: float = 0.08,
    active_vocal_mask: np.ndarray | None = None,
    is_intra_line_blank_or_star: np.ndarray | None = None,
    intra_line_silence_penalty: float = 0.12,
):
    lp=np.asarray(log_probs,dtype=np.float32)
    T,nc=lp.shape; L=len(targets)
    S=2*L+1
    tok=np.empty(S,dtype=np.int32); tok[0::2]=blank_id; tok[1::2]=np.asarray(targets,dtype=np.int32)
    can_skip=np.zeros(S,dtype=bool)
    for k in range(1,L):
        if targets[k]!=targets[k-1]: can_skip[2*k+1]=True
    dp=np.full(S,NEG_INF,dtype=np.float32)
    dp[0]=lp[0,tok[0]]
    if S>1: dp[1]=lp[0,tok[1]]
    if leading_vocal_penalty > 0.0 and is_leading_blank_or_star is not None and active_vocal_mask is not None and active_vocal_mask[0]:
        if is_leading_blank_or_star[0]: dp[0] -= leading_vocal_penalty
        if S > 1 and is_leading_blank_or_star[1]: dp[1] -= leading_vocal_penalty
    if intra_line_silence_penalty > 0.0 and is_intra_line_blank_or_star is not None and active_vocal_mask is not None and not active_vocal_mask[0]:
        dp[is_intra_line_blank_or_star] -= intra_line_silence_penalty
    bp=np.zeros((T,S),dtype=np.uint8)
    idx=np.arange(S)
    for t in range(1,T):
        emit=lp[t,tok].copy()
        if intra_blank_penalty > 0.0 and is_intra_word_blank is not None:
            emit[is_intra_word_blank] -= intra_blank_penalty
        if leading_vocal_penalty > 0.0 and is_leading_blank_or_star is not None and active_vocal_mask is not None and active_vocal_mask[t]:
            emit[is_leading_blank_or_star] -= leading_vocal_penalty
        if intra_line_silence_penalty > 0.0 and is_intra_line_blank_or_star is not None and active_vocal_mask is not None and not active_vocal_mask[t]:
            emit[is_intra_line_blank_or_star] -= intra_line_silence_penalty
        c0=dp
        c1=np.where(idx>=1,dp[np.maximum(0,idx-1)],NEG_INF)
        c2=np.where(can_skip,dp[np.maximum(0,idx-2)],NEG_INF)
        best=c0; choice=np.zeros(S,dtype=np.uint8)
        m1=c1>best; best=np.where(m1,c1,best); choice=np.where(m1,1,choice)
        m2=c2>best; best=np.where(m2,c2,best); choice=np.where(m2,2,choice)
        dp=best+emit; bp[t]=choice
    end=S-1
    if S>1 and dp[S-2]>=dp[S-1]: end=S-2
    st=np.empty(T,dtype=np.int32); cur=end
    for t in range(T-1,-1,-1):
        st[t]=cur; cur=int(cur-int(bp[t,cur]))
    frame_lp=lp[np.arange(T),tok[st]].astype(np.float32)
    return st, frame_lp, tok

def _alpha_beta_marginals(
    log_probs: np.ndarray,
    targets: Sequence[int],
    blank_id: int,
    is_intra_word_blank: np.ndarray | None = None,
    intra_blank_penalty: float = 0.30,
    is_leading_blank_or_star: np.ndarray | None = None,
    leading_vocal_penalty: float = 0.08,
    active_vocal_mask: np.ndarray | None = None,
    is_intra_line_blank_or_star: np.ndarray | None = None,
    intra_line_silence_penalty: float = 0.12,
):
    lp=np.asarray(log_probs,dtype=np.float64); T=lp.shape[0]; L=len(targets); S=2*L+1
    tok=np.empty(S,dtype=np.int32); tok[0::2]=blank_id; tok[1::2]=np.asarray(targets,dtype=np.int32)
    can_skip=np.zeros(S,dtype=bool)
    for k in range(1,L):
        if targets[k]!=targets[k-1]: can_skip[2*k+1]=True
    alpha=np.full((T,S),NEG_INF,dtype=np.float64)
    alpha[0,0]=lp[0,tok[0]];
    if S>1: alpha[0,1]=lp[0,tok[1]]
    if leading_vocal_penalty > 0.0 and is_leading_blank_or_star is not None and active_vocal_mask is not None and active_vocal_mask[0]:
        if is_leading_blank_or_star[0]: alpha[0, 0] -= leading_vocal_penalty
        if S > 1 and is_leading_blank_or_star[1]: alpha[0, 1] -= leading_vocal_penalty
    if intra_line_silence_penalty > 0.0 and is_intra_line_blank_or_star is not None and active_vocal_mask is not None and not active_vocal_mask[0]:
        alpha[0, is_intra_line_blank_or_star] -= intra_line_silence_penalty
    for t in range(1,T):
        prev=alpha[t-1]
        s0=prev
        s1=np.empty(S,dtype=np.float64); s1[0]=NEG_INF; s1[1:]=prev[:-1]
        acc=np.logaddexp(s0,s1)
        s2=np.full(S,NEG_INF,dtype=np.float64); s2[2:]=np.where(can_skip[2:],prev[:-2],NEG_INF)
        curr_lp=lp[t,tok].astype(np.float64, copy=True)
        if intra_blank_penalty > 0.0 and is_intra_word_blank is not None:
            curr_lp[is_intra_word_blank] -= intra_blank_penalty
        if leading_vocal_penalty > 0.0 and is_leading_blank_or_star is not None and active_vocal_mask is not None and active_vocal_mask[t]:
            curr_lp[is_leading_blank_or_star] -= leading_vocal_penalty
        if intra_line_silence_penalty > 0.0 and is_intra_line_blank_or_star is not None and active_vocal_mask is not None and not active_vocal_mask[t]:
            curr_lp[is_intra_line_blank_or_star] -= intra_line_silence_penalty
        alpha[t]=np.logaddexp(acc,s2)+curr_lp
    log_z=np.logaddexp(alpha[-1,S-1], alpha[-1,S-2] if S>1 else NEG_INF)
    beta=np.full((T,S),NEG_INF,dtype=np.float64)
    beta[-1,S-1]=0.0
    if S>1: beta[-1,S-2]=0.0
    can_skip_fwd=np.zeros(S,dtype=bool); can_skip_fwd[:-2]=can_skip[2:]
    for t in range(T-2,-1,-1):
        nxt_lp=lp[t+1,tok].astype(np.float64, copy=True)
        if intra_blank_penalty > 0.0 and is_intra_word_blank is not None:
            nxt_lp[is_intra_word_blank] -= intra_blank_penalty
        if leading_vocal_penalty > 0.0 and is_leading_blank_or_star is not None and active_vocal_mask is not None and active_vocal_mask[t+1]:
            nxt_lp[is_leading_blank_or_star] -= leading_vocal_penalty
        if intra_line_silence_penalty > 0.0 and is_intra_line_blank_or_star is not None and active_vocal_mask is not None and not active_vocal_mask[t+1]:
            nxt_lp[is_intra_line_blank_or_star] -= intra_line_silence_penalty
        nxt=beta[t+1]+nxt_lp
        b0=nxt
        b1=np.empty(S,dtype=np.float64); b1[-1]=NEG_INF; b1[:-1]=nxt[1:]
        acc=np.logaddexp(b0,b1)
        b2=np.full(S,NEG_INF,dtype=np.float64); b2[:-2]=np.where(can_skip_fwd[:-2],nxt[2:],NEG_INF)
        beta[t]=np.logaddexp(acc,b2)
    log_gamma=alpha+beta-log_z
    gamma=np.exp(np.clip(log_gamma,-40,0)).astype(np.float32)
    return gamma, float(log_z)

def align_global_words(
    log_probs: torch.Tensor,
    word_token_ids: Sequence[Sequence[int]],
    blank_id: int,
    star_id: int,
    sec_per_frame: float = 0.02,
    word_target: float = 0.95,
    silence_frame_mask: "np.ndarray | None" = None,
    active_vocal_mask: "np.ndarray | None" = None,
    leading_vocal_penalty: float = 0.08,
    line_indices: "Sequence[int] | None" = None,
    intra_line_silence_penalty: float = 0.12,
) -> List[WordPosterior]:
    lp = (log_probs[0] if log_probs.dim()==3 else log_probs).detach().cpu().numpy().astype(np.float32).copy()
    T=lp.shape[0]
    # Vocal-silence character masking: on silent frames, penalize non-blank lexical chars
    # so word starts/ends cannot latch onto synth intros or drop transients.
    if silence_frame_mask is not None:
        sm = np.asarray(silence_frame_mask, dtype=bool).reshape(-1)
        if sm.shape[0] >= T:
            sm = sm[:T]
        else:
            sm = np.pad(sm, (0, T-sm.shape[0]), constant_values=False)
        if sm.any():
            lexical = np.ones(lp.shape[1], dtype=bool)
            lexical[blank_id] = False
            if 0 <= star_id < lp.shape[1]:
                lexical[star_id] = False
            lp[sm][:, lexical] -= 6.0
            lp = lp - np.logaddexp.reduce(lp, axis=1, keepdims=True)
    targets, spans = build_global_star_targets(word_token_ids, star_id=star_id, blank_id=blank_id)
    L = len(targets)
    S = 2 * L + 1
    is_intra_word_blank = np.zeros(S, dtype=bool)
    for ws, we in spans:
        for k in range(ws, we - 1):
            is_intra_word_blank[2 * (k + 1)] = True

    is_leading_blank_or_star = np.zeros(S, dtype=bool)
    if spans:
        first_word_s = spans[0][0]
        is_leading_blank_or_star[: 2 * first_word_s + 1] = True

    is_intra_line_blank_or_star = np.zeros(S, dtype=bool)
    if line_indices is not None and len(spans) > 1:
        for i in range(len(spans) - 1):
            if i < len(line_indices) - 1 and line_indices[i] == line_indices[i + 1]:
                star_target_idx = spans[i][1]
                is_intra_line_blank_or_star[2 * star_target_idx : 2 * star_target_idx + 3] = True

    act_mask = None
    if active_vocal_mask is not None:
        am = np.asarray(active_vocal_mask, dtype=bool).reshape(-1)
        if am.shape[0] >= T:
            act_mask = am[:T]
        else:
            act_mask = np.pad(am, (0, T - am.shape[0]), constant_values=False)
    elif silence_frame_mask is not None:
        act_mask = ~sm

    st, frame_lp, tok = _viterbi_path(
        lp,
        targets,
        blank_id,
        is_intra_word_blank=is_intra_word_blank,
        intra_blank_penalty=0.30,
        is_leading_blank_or_star=is_leading_blank_or_star,
        leading_vocal_penalty=leading_vocal_penalty,
        active_vocal_mask=act_mask,
        is_intra_line_blank_or_star=is_intra_line_blank_or_star,
        intra_line_silence_penalty=intra_line_silence_penalty,
    )
    # local alpha-beta marginal mass per word span (exact on local window band)
    gamma, log_z = _alpha_beta_marginals(
        lp,
        targets,
        blank_id,
        is_intra_word_blank=is_intra_word_blank,
        intra_blank_penalty=0.30,
        is_leading_blank_or_star=is_leading_blank_or_star,
        leading_vocal_penalty=leading_vocal_penalty,
        active_vocal_mask=act_mask,
        is_intra_line_blank_or_star=is_intra_line_blank_or_star,
        intra_line_silence_penalty=intra_line_silence_penalty,
    )
    tgt_idx = np.where((st%2)==1,(st-1)//2,-1)
    null_lp = np.maximum(lp[:,blank_id], lp[:,star_id]) if star_id < lp.shape[1] else lp[:,blank_id]
    out=[]
    for ws,we in spans:
        frames=[i for i,ti in enumerate(tgt_idx.tolist()) if ws<=ti<we]
        if frames:
            f0,f1=min(frames), max(frames)+1
        else:
            # fallback to marginal peak
            state_s = 2*ws+1
            mass = gamma[:,state_s] if state_s < gamma.shape[1] else np.zeros(T)
            peak=int(np.argmax(mass)); f0=max(0,peak-1); f1=min(T,peak+2)
        # honest mean char prob (no power-law inflation)
        char_lps=[]; llrs=[]
        for f in range(f0,f1):
            ti=int(tgt_idx[f])
            if ws<=ti<we:
                char_lps.append(float(frame_lp[f]))
                llrs.append(float(frame_lp[f])-float(null_lp[f]))
        mean_lp=float(np.mean(char_lps)) if char_lps else -10.0
        post=float(math.exp(max(-20.0,min(0.0,mean_lp))))
        null_llr=float(np.mean(llrs)) if llrs else -5.0
        # marginal mass in [f0,f1] for word states
        masses=[]
        for ci in range(ws,we):
            si=2*ci+1
            if si < gamma.shape[1]:
                masses.append(float(np.sum(gamma[max(0,f0-2):min(T,f1+2),si]))/max(1e-6,float(np.sum(gamma[:,si]))))
        marginal=float(np.clip(np.mean(masses) if masses else post,0,1))
        # calibrated confidence (no inflation): conservative logistic-like map
        conf = 1/(1+math.exp(-( -1.8 + 2.8*post + 0.35*max(-2, min(4, null_llr)) + 1.2*marginal )))
        conf=float(np.clip(conf,0.01,0.999))
        if marginal < 0.25 or null_llr < -0.5: conf=min(conf, word_target-0.05)
        elif post < 0.70: conf=min(conf, word_target-0.01)
        out.append(WordPosterior(f0,f1,round(post,6),round(null_llr,4),round(conf,6)))
    return out
