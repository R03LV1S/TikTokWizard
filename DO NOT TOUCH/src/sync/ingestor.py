"""Holy-Grail lyric ingestion, UTF-8 byte-preserving parsing, and multi-script transliteration (v4.1).

Invariants:
  1. Never alters, strips, or re-encodes the user's surface word tokens (`UserLyricWord.text`
     is 100% byte-for-byte identical to the token in `<lyrics>.txt`).
  2. Never emits whitespace strings as word tokens (`425` words in -> `425` `UserLyricWord` objects out).
  3. Properly tracks parenthetical state across multi-word `( ... )` spans (e.g. `(Gracias, gracias)`).
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Tuple

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


@dataclass
class UserLyricWord:
    """A single word from the user's Holy-Grail lyric file with positional and phonetic metadata."""

    text: str
    is_parenthetical: bool = False
    start_sec: float = 0.0
    end_sec: float = 0.0
    confidence: float = 1.0
    line_index: int = 0
    word_index_in_line: int = 0
    global_index: int = 0
    romanized_key: str = ""
    syllable_count: int = 1


def romanize_word(word_text: str) -> str:
    """Convert any Unicode word to a lowercase Romanized `[a-z']+` key for acoustic models."""
    if not word_text:
        return ""
    raw = str(word_text).replace("\u00f1", "n").replace("\u00d1", "n")
    try:
        from anyascii import anyascii

        folded = anyascii(raw).lower()
    except Exception:
        nfkd = unicodedata.normalize("NFKD", raw)
        folded = "".join(c for c in nfkd if not unicodedata.combining(c)).lower()

    clean = re.sub(r"[^a-z']+", "", folded)
    return clean or folded.strip()


def _count_syllables_from_key(roman_key: str) -> int:
    if not roman_key:
        return 1
    groups = re.findall(r"[aeiouy]+", roman_key.lower())
    return max(1, len(groups))


def parse_user_lyric(raw_lyric: str) -> Tuple[List[str], List[UserLyricWord]]:
    """Parse user lyric text into 1:1 non-empty lines and immutable `UserLyricWord` tokens.

    Tracks open/close parentheses across tokens on a line so both single-word
    `'(Gracias)'` and multi-word `'(Gracias, gracias)'` tokens are tagged with
    `is_parenthetical = True` while preserving the exact surface text (including
    parentheses, punctuation, and UTF-8 characters like `'enseñarme'`).
    """
    cleaned_raw = (
        (raw_lyric or "")
        .lstrip("\ufeff")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )
    lines = [line.strip() for line in cleaned_raw.split("\n") if line.strip()]
    user_lyric_words: List[UserLyricWord] = []

    global_idx = 0
    for line_idx, line in enumerate(lines):
        tokens = line.split()
        line_enclosed = line.startswith("(") and line.endswith(")")
        in_paren_span = False

        for word_idx, tok in enumerate(tokens):
            if "(" in tok:
                in_paren_span = True

            is_paren = bool(
                line_enclosed or in_paren_span or ("(" in tok) or (")" in tok)
            )
            rom = romanize_word(tok)
            syl = _count_syllables_from_key(rom)

            user_lyric_words.append(
                UserLyricWord(
                    text=tok,
                    is_parenthetical=is_paren,
                    start_sec=0.0,
                    end_sec=0.0,
                    confidence=1.0,
                    line_index=line_idx,
                    word_index_in_line=word_idx,
                    global_index=global_idx,
                    romanized_key=rom,
                    syllable_count=syl,
                )
            )
            global_idx += 1

            if ")" in tok:
                in_paren_span = False

    return lines, user_lyric_words
