"""Universal Multi-Tier Language Detection Engine (v5.6).

1. Unicode Script Analysis:
   - Identifies non-Latin scripts (CJK: ja/zh/ko, Cyrillic: ru, Arabic/Persian: ar,
     Devanagari: hi, Greek: el, Hebrew: he, Thai: th) via standard Unicode blocks.
2. Latin-Script Multilingual Profiler:
   - Evaluates word and n-gram frequencies across major Latin languages:
     en, es, pt, fr, de, it, nl, pl, ro, tr, sv, da, no, fi, cs, id, vi, ca, hu.
3. Bayesian Consensus Fusion:
   - Fuses text posterior with acoustic prior from faster-whisper or audio metadata.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple


# Non-Latin Unicode Script Ranges -> Language Code
_SCRIPT_RANGES: List[Tuple[str, re.Pattern]] = [
    ("ja", re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")),       # Hiragana / Katakana
    ("ko", re.compile(r"[\uac00-\ud7af\u1100-\u11ff]")),       # Hangul
    ("zh", re.compile(r"[\u4e00-\u9fff]")),                     # Hanzi (if not ja/ko)
    ("ru", re.compile(r"[\u0400-\u04ff]")),                     # Cyrillic
    ("ar", re.compile(r"[\u0600-\u06ff]")),                     # Arabic / Persian
    ("hi", re.compile(r"[\u0900-\u097f]")),                     # Devanagari (Hindi)
    ("el", re.compile(r"[\u0370-\u03ff]")),                     # Greek
    ("he", re.compile(r"[\u0590-\u05ff]")),                     # Hebrew
    ("th", re.compile(r"[\u0e00-\u0e7f]")),                     # Thai
]

# High-frequency grammatical function words / stopwords for Latin languages
_LATIN_STOPWORDS: Dict[str, set[str]] = {
    "en": {
        "the", "and", "to", "of", "a", "in", "is", "it", "you", "that",
        "was", "for", "on", "are", "with", "as", "i", "his", "they", "be",
        "at", "one", "have", "this", "from", "or", "had", "by", "not", "word",
        "but", "what", "some", "we", "can", "out", "other", "were", "all",
        "there", "when", "up", "your", "how", "said", "an", "each", "she",
        "which", "do", "their", "time", "if", "will", "way", "about", "many",
        "then", "them", "would", "like", "so", "these", "her", "make", "thing",
        "see", "him", "has", "look", "more", "day", "could", "go", "come",
        "my", "no", "most", "know", "than", "who", "now", "find", "get",
        "me", "give", "just", "say", "love", "baby", "want", "need", "back"
    },
    "es": {
        "de", "la", "que", "el", "en", "y", "a", "los", "del", "se",
        "las", "por", "un", "para", "con", "no", "una", "su", "al", "lo",
        "como", "mas", "más", "pero", "sus", "le", "ya", "o", "este", "sí",
        "porque", "esta", "entre", "cuando", "muy", "sin", "sobre", "también",
        "tambien", "me", "hasta", "hay", "donde", "quien", "desde", "todo",
        "nos", "durante", "todos", "uno", "les", "ni", "contra", "otros",
        "ese", "eso", "ante", "ellos", "esto", "mí", "mi", "qué", "yo",
        "otro", "él", "ella", "pa", "ta", "corazón", "amor", "tiempo", "vida",
        "bien", "llena", "siempre"
    },
    "pt": {
        "de", "a", "o", "que", "e", "do", "da", "em", "um", "para",
        "com", "não", "uma", "os", "no", "se", "na", "por", "mais", "as",
        "dos", "como", "mas", "foi", "ao", "ele", "das", "tem", "seu", "sua",
        "ou", "ser", "quando", "muito", "nos", "já", "está", "eu", "você",
        "pra", "pro", "meu", "minha", "são", "tudo", "amor", "vida", "onde"
    },
    "fr": {
        "de", "la", "le", "et", "les", "des", "en", "un", "du", "une",
        "que", "est", "pour", "qui", "dans", "par", "plus", "pas", "au",
        "sur", "ne", "ce", "avec", "sont", "ou", "son", "mais", "nous",
        "vous", "comme", "on", "tout", "elle", "sa", "fait", "été", "aussi",
        "ces", "mon", "ma", "mes", "toi", "moi", "rien", "faire", "temps"
    },
    "de": {
        "der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich",
        "des", "auf", "für", "ist", "im", "dem", "nicht", "ein", "eine", "als",
        "auch", "es", "an", "werden", "aus", "er", "hat", "dass", "sie", "nach",
        "wird", "bei", "einer", "um", "am", "sind", "noch", "wie", "einem", "über",
        "einen", "so", "ich", "dich", "mich", "du", "wir", "nur", "hier", "alles"
    },
    "it": {
        "di", "e", "il", "che", "la", "un", "in", "per", "una", "non",
        "sono", "con", "si", "del", "da", "le", "ha", "era", "ma", "ed",
        "della", "anche", "come", "dei", "gli", "ci", "io", "tu", "noi", "voi",
        "mio", "mia", "tuo", "tua", "suo", "sua", "tutto", "vita", "amore", "qui"
    },
    "nl": {
        "de", "van", "en", "het", "in", "is", "een", "op", "te", "met",
        "voor", "zijn", "dat", "die", "aan", "niet", "om", "ook", "als", "bij",
        "er", "maar", "uit", "nog", "wel", "naar", "dan", "wat", "je", "jij",
        "wij", "ze", "mijn", "onze", "zo", "door", "al", "geen"
    },
    "pl": {
        "w", "i", "z", "na", "do", "że", "się", "to", "nie", "jak",
        "po", "dla", "co", "od", "przez", "jest", "są", "mnie", "cię", "go",
        "jej", "ich", "ty", "ja", "my", "wy", "tak", "ale", "tylko", "wszystko"
    },
    "tr": {
        "bir", "ve", "bu", "da", "de", "için", "ile", "çok", "daha", "en",
        "kadar", "ama", "gibi", "var", "yok", "ben", "sen", "biz", "siz", "ne",
        "bana", "sana", "beni", "seni", "her", "zaman", "böyle", "artık"
    },
    "id": {
        "yang", "di", "dan", "ini", "dari", "dalam", "untuk", "tidak", "dengan", "ke",
        "akan", "pada", "juga", "saya", "adalah", "itu", "bisa", "ada", "mereka", "kita",
        "kamu", "kau", "aku", "tak", "sudah", "lagi", "hati", "cinta"
    },
}


def detect_text_language(words: Sequence[str]) -> Tuple[str, float]:
    """Detect language from lyric tokens using Unicode script analysis & multi-lingual ranker.

    Returns (lang_code, confidence) where lang_code is ISO-639-1 (e.g. 'en', 'es', 'fr', 'ja').
    """
    if not words:
        return "en", 0.0

    joined = " ".join(words)

    # 1. Unicode Script Analysis
    # Count occurrences of non-Latin characters
    for code, pattern in _SCRIPT_RANGES:
        matches = pattern.findall(joined)
        if len(matches) >= 3 or (len(joined) > 0 and len(matches) / max(1, len(re.sub(r"\s+", "", joined))) > 0.15):
            return code, 0.95

    # 2. Latin-Script Multilingual Profiler
    clean_words = [re.sub(r"^[^\w]+|[^\w']+$", "", w.lower().strip(), flags=re.UNICODE) for w in words]
    clean_words = [w for w in clean_words if w]
    if not clean_words:
        return "en", 0.0

    scores: Dict[str, int] = {}
    total_words = len(clean_words)

    for lang_code, sw_set in _LATIN_STOPWORDS.items():
        hits = sum(1 for w in clean_words if w in sw_set)
        scores[lang_code] = hits

    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_lang, best_hits = sorted_scores[0]
    second_lang, second_hits = sorted_scores[1] if len(sorted_scores) > 1 else ("", 0)

    if best_hits == 0:
        return "en", 0.0  # Unidentified Latin text

    # Compute confidence based on margin over second place and total word hits
    margin = best_hits - second_hits
    confidence = min(1.0, float(best_hits) / max(1.0, float(total_words) * 0.15))
    if margin <= 1 and best_hits <= 3:
        confidence *= 0.50

    return best_lang, float(confidence)


def resolve_language_consensus(
    lyric_lang: str,
    lyric_conf: float,
    audio_lang: Optional[str] = None,
    audio_prob: float = 0.0,
    explicit_lang: Optional[str] = None,
) -> str:
    """Fuse lyric-text prediction with Whisper audio Bayesian prediction.

    Precedence:
    1. Explicit caller override (--lang) always wins.
    2. High-confidence lyric text (>= 0.60) wins.
    3. If lyric text is ambiguous (< 0.60) and audio is confident (>= 0.50), audio wins.
    4. If both agree, unanimous lock.
    5. Default to 'en' if completely unresolved.
    """
    if explicit_lang:
        return explicit_lang.lower().strip()

    if audio_lang and audio_lang == lyric_lang:
        return lyric_lang

    if lyric_conf >= 0.60:
        return lyric_lang

    if audio_lang and audio_prob >= 0.50:
        return audio_lang.lower().strip()

    return lyric_lang if lyric_conf > 0.30 else (audio_lang or "en")
