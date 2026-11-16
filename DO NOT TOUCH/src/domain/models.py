"""Domain models and immutable data contracts for TikTok Wizard pipeline (v4.1)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ProjectConfig:
    """Configuration for a karaoke project."""

    project_name: str
    source_audio: str
    lyric_source: str
    output_dir: str
    word_target: float = 0.95
    tau_song: float = 0.23
    whisper_size: str = "small"
    sample_rate: int = 16000


@dataclass
class UserLyricWord:
    """Immutable Holy-Grail word token and its aligned acoustic state."""

    word: str
    line_index: int = 0
    word_index_in_line: int = 0
    global_index: int = 0
    romanized: str = ""
    consonant_key: str = ""
    is_parenthetical: bool = False
    syllable_count: int = 1
    articulatory_weight: float = 1.0
    start_sec: float = 0.0
    end_sec: float = 0.0
    duration_cs: int = 0
    posterior: float = 0.0
    fb_jitter_sec: float = 0.0
    rms: float = 0.0
    confidence: float = 0.0
    is_cemented: bool = False
    source_stream: str = "acapella_full"
    vocal_effect: str = "NONE"

    @property
    def text(self) -> str:
        return self.word

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UserLyricLine:
    """Single lyric line in strict 1:1 correspondence with user input file."""

    index: int  # 1-based line index
    text: str
    start_sec: float = 0.0
    end_sec: float = 0.0
    is_parenthetical: bool = False
    words: List[UserLyricWord] = field(default_factory=list)
    sync_status: str = "VERIFIED"

    @property
    def line_index(self) -> int:
        return max(0, self.index - 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "start_sec": round(float(self.start_sec), 4),
            "end_sec": round(float(self.end_sec), 4),
            "text": self.text,
            "is_parenthetical": self.is_parenthetical,
            "sync_status": self.sync_status,
        }


@dataclass
class UserLyricDocument:
    """Complete Holy-Grail parsed lyric document."""

    lines: List[UserLyricLine] = field(default_factory=list)
    words: List[UserLyricWord] = field(default_factory=list)

    @property
    def total_lines(self) -> int:
        return len(self.lines)

    @property
    def total_words(self) -> int:
        return len(self.words)


@dataclass
class DiscrepancyGap:
    """Contiguous span of uncemented words bracketed by cemented anchors."""

    start_idx: int
    end_idx: int
    gap_start_sec: float
    gap_end_sec: float
    prev_cemented: Optional[Dict[str, Any]] = None
    next_cemented: Optional[Dict[str, Any]] = None
    words: List[Dict[str, Any]] = field(default_factory=list)
    selected_stream: str = "original_mix_vocal_band"


@dataclass
class EchoTrainCluster:
    """Detected studio delay/echo bounce cluster."""

    root_token: str
    start_sec: float
    end_sec: float
    total_bounces: int
    written_parenthetical_count: int
    suppressed_clone_count: int


@dataclass
class BeatGrid:
    """Extracted musical tempo and 16th-note subdivision grid."""

    bpm: float = 120.0
    beat_times: List[float] = field(default_factory=list)
    subgrid_16th: List[float] = field(default_factory=list)


@dataclass
class AcousticStreams:
    """Paths and metadata for the 4 synchronized 16 kHz mono audio streams."""

    original_mix_vocal_band: str
    acapella_full: str
    acapella_lead_dry: str
    acapella_backing_echo: str
    sample_rate: int = 16000
    duration_sec: float = 0.0


@dataclass
class AlignmentResult:
    """Summary result of the alignment pipeline."""

    verified_words: int
    total_words: int
    conflict_low_evidence: int
    cemented_words: int
    uncemented_words: int
    max_word_duration_cs: int
    tau_song: float
    quality_gate_pass: bool
    total_duration_sec: float = 0.0
    alignment_confidence_mean: float = 0.0


@dataclass
class AlignmentManifest:
    """Persisted machine-readable output manifest for a project alignment."""

    project: str
    engine: str = "greenfield_v4_1_real_mms_fa"
    tau_song: float = 0.23
    word_target: float = 0.95
    total_duration_sec: float = 0.0
    total_input_lines: int = 0
    total_words: int = 0
    cemented_words_ge_95_count: int = 0
    uncemented_words_count: int = 0
    asr_words: int = 0
    words: List[Dict[str, Any]] = field(default_factory=list)
    lines: List[Dict[str, Any]] = field(default_factory=list)
    verification: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
