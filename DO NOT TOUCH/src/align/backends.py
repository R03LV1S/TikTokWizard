# align/backends.py
"""Pluggable alignment backends for karaoke word timing.

MMS_FA is the default speech-trained backend; future backends can add
singing-robust/lyrics-specific alignment without changing the pipeline contract.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Protocol
import torch

class AlignerBackend(Protocol):
    name: str
    def extract_emissions(self, waveform_1d: torch.Tensor, sr: int = 16000) -> torch.Tensor: ...
    def align_words(self, emissions: torch.Tensor, words: Sequence[str], romanized: Sequence[str], is_parenthetical: Sequence[bool], window_start_sec: float, sec_per_frame: float) -> List[Dict[str, Any]]: ...

class MmsFaBackend:
    name = "mms_fa_star_ctc"
    def extract_emissions(self, waveform_1d: torch.Tensor, sr: int = 16000) -> torch.Tensor:
        from .mms_fa import extract_emissions_chunked
        return extract_emissions_chunked(waveform_1d, sr=sr)
    def align_words(self, emissions: torch.Tensor, words: Sequence[str], romanized: Sequence[str], is_parenthetical: Sequence[bool], window_start_sec: float, sec_per_frame: float) -> List[Dict[str, Any]]:
        from .mms_fa import align_words_star_ctc
        return align_words_star_ctc(emissions, words, romanized, is_parenthetical, window_start_sec, sec_per_frame)
