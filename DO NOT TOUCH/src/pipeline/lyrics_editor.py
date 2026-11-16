import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from .schema import KaraokeEntry, KaraokeWord


def parse_ass_to_easy_lyrics(ass_path: Path) -> List[Dict]:
    """
    Parse an ASS karaoke file into structured lines with words and timestamps.
    """
    if not ass_path.exists():
        return []

    content = ass_path.read_text(encoding="utf-8")
    dialogues = []

    def to_sec(ts_str: str) -> float:
        parts = ts_str.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return 0.0

    for line in content.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",", 9)
        if len(parts) < 10:
            continue
        start_sec = to_sec(parts[1].strip())
        end_sec = to_sec(parts[2].strip())
        text_raw = parts[9]

        # Strip standard styling tags {\an5\pos(X,Y)}
        cleaned = re.sub(r"^{\\an\d+\\pos\([^)]+\)}", "", text_raw)

        # Match {\k<duration>}Word tokens
        tokens = re.findall(r"\{\\k(\d+)\}([^{]*)", cleaned)
        words = []
        for cs_str, word_txt in tokens:
            dur_cs = int(cs_str)
            w = word_txt.strip()
            if w:
                words.append({"word": w, "duration_cs": dur_cs})

        if words:
            dialogues.append({
                "start": start_sec,
                "end": end_sec,
                "words": words,
            })
    return dialogues


def export_easy_lyrics_file(ass_path: Path, output_file: Path) -> bool:
    """
    Export an intuitive, editable lyric file with timestamp and word durations.
    """
    dialogues = parse_ass_to_easy_lyrics(ass_path)
    if not dialogues:
        return False

    out = [
        "# ==========================================================================",
        "# 🪄 TIKTOK WIZARD — EASY-EDIT LYRICS & SYNC EDITOR",
        "# ==========================================================================",
        "# HOW TO EDIT THIS FILE:",
        "# 1. You can freely edit words, fix typos, or change punctuation.",
        "# 2. [MM:SS.cc] at the beginning of each line is the start timestamp.",
        "# 3. Optional: (XXcs) after any word sets its sung highlight duration.",
        "#    (100cs = 1.0 second. If you remove (XXcs) or add new words, TikTok Wizard",
        "#     automatically balances durations smoothly across the line!)",
        "# 4. Save this file! TikTok Wizard automatically detects your edits and updates",
        "#    the render without needing to touch complex .ass or .json files.",
        "# ==========================================================================\n"
    ]

    for d in dialogues:
        m = int(d["start"] // 60)
        s = d["start"] % 60
        ts_str = f"[{m:02d}:{s:05.2f}]"
        word_parts = []
        for w in d["words"]:
            word_parts.append(f"{w['word']} ({w['duration_cs']}cs)")
        out.append(f"{ts_str} " + " ".join(word_parts))

    output_file.write_text("\n".join(out), encoding="utf-8")
    return True


def import_easy_lyrics_to_entries(easy_lyrics_path: Path, default_line_dur: float = 2.5) -> List[KaraokeEntry]:
    """
    Parse an easy_lyrics.txt file into KaraokeEntry objects suitable for write_ass_karaoke.
    """
    if not easy_lyrics_path.exists():
        return []

    lines = easy_lyrics_path.read_text(encoding="utf-8").splitlines()
    entries: List[KaraokeEntry] = []

    for l_idx, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        m = re.match(r"^\[([0-9:.]+)\]\s*(.*)$", line)
        start_sec = 0.0
        rest = line
        if m:
            ts_str = m.group(1)
            rest = m.group(2)
            parts = ts_str.split(":")
            try:
                if len(parts) == 3:
                    start_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                elif len(parts) == 2:
                    start_sec = int(parts[0]) * 60 + float(parts[1])
                else:
                    start_sec = float(parts[0])
            except ValueError:
                start_sec = 0.0

        tokens = re.findall(r"([^\s()]+)(?:\s*\(([0-9]+)cs\))?", rest)
        if not tokens:
            continue

        raw_words = []
        for w_text, cs_str in tokens:
            dur_cs = int(cs_str) if cs_str else None
            raw_words.append((w_text, dur_cs))

        # Calculate durations if any word omitted (XXcs)
        assigned_cs = [c for _, c in raw_words if c is not None]
        total_assigned_cs = sum(assigned_cs)
        unassigned_count = len([1 for _, c in raw_words if c is None])

        default_dur_cs = int(default_line_dur * 100)
        if unassigned_count > 0:
            remaining_cs = max(unassigned_count * 20, default_dur_cs - total_assigned_cs)
            per_w_cs = max(15, remaining_cs // unassigned_count)
        else:
            per_w_cs = 30

        k_words: List[KaraokeWord] = []
        curr_time = start_sec
        for w_text, cs_val in raw_words:
            dur_cs = cs_val if cs_val is not None else per_w_cs
            dur_sec = dur_cs / 100.0
            k_words.append(KaraokeWord(
                text=w_text,
                duration_cs=dur_cs,
                start=curr_time,
                end=curr_time + dur_sec,
            ))
            curr_time += dur_sec

        line_end = curr_time + 0.35  # Linger slightly after last word
        entries.append(KaraokeEntry(
            line_index=l_idx,
            words=k_words,
            start=round(start_sec, 2),
            end=round(line_end, 2),
            raw_text=" ".join(w.text for w in k_words),
        ))

    return entries
