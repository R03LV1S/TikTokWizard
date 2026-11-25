"""Lightweight language-aware number-to-words converter for acoustic alignment (v4.5).

Converts integers (0-9999) and alphanumeric tokens (e.g., '9C') to phonetic words in Spanish ('es')
and English ('en'). Used ONLY for phonetic alignment key generation; display tokens remain byte-for-byte exact.
"""

from __future__ import annotations
import re
from typing import Optional

_ES_UNITS = [
    "cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve",
    "diez", "once", "doce", "trece", "catorce", "quince", "dieciseis", "diecisiete", "dieciocho", "diecinueve"
]

_ES_TENS = [
    "", "", "veinte", "treinta", "cuarenta", "cincuenta", "sesenta", "setenta", "ochenta", "noventa"
]

_ES_HUNDREDS = [
    "", "ciento", "doscientos", "trescientos", "cuatrocientos", "quinientos",
    "seiscientos", "setecientos", "ochocientos", "novecientos"
]

_EN_UNITS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"
]

_EN_TENS = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"
]


def int_to_spanish(n: int) -> str:
    """Convert integer 0-9999 to Spanish words."""
    if n < 0 or n > 9999:
        return str(n)
    if n < 20:
        return _ES_UNITS[n]
    if n == 20:
        return "veinte"
    if n < 30:
        # 21-29: veintiuno, veintidos, veintitres, ...
        return "veinti" + _ES_UNITS[n - 20]
    if n < 100:
        tens = n // 10
        rem = n % 10
        if rem == 0:
            return _ES_TENS[tens]
        return f"{_ES_TENS[tens]} y {_ES_UNITS[rem]}"
    if n == 100:
        return "cien"
    if n < 1000:
        h = n // 100
        rem = n % 100
        prefix = _ES_HUNDREDS[h]
        if rem == 0:
            return prefix
        return f"{prefix} {int_to_spanish(rem)}"
    # 1000 - 9999
    thousands = n // 1000
    rem = n % 1000
    th_str = "mil" if thousands == 1 else f"{int_to_spanish(thousands)} mil"
    if rem == 0:
        return th_str
    return f"{th_str} {int_to_spanish(rem)}"


def int_to_english(n: int) -> str:
    """Convert integer 0-9999 to English words."""
    if n < 0 or n > 9999:
        return str(n)
    if n < 20:
        return _EN_UNITS[n]
    if n < 100:
        tens = n // 10
        rem = n % 10
        if rem == 0:
            return _EN_TENS[tens]
        return f"{_EN_TENS[tens]} {_EN_UNITS[rem]}"
    if n < 1000:
        h = n // 100
        rem = n % 100
        prefix = f"{_EN_UNITS[h]} hundred"
        if rem == 0:
            return prefix
        return f"{prefix} {int_to_english(rem)}"
    # 1000 - 9999
    thousands = n // 1000
    rem = n % 1000
    th_str = f"{int_to_english(thousands)} thousand"
    if rem == 0:
        return th_str
    return f"{th_str} {int_to_english(rem)}"


def expand_token_numbers(token: str, lang: str = "es") -> str:
    """Expand numeric digits inside a token to words in Spanish ('es') or English ('en').

    Examples:
        expand_token_numbers('48', 'es') -> 'cuarenta y ocho'
        expand_token_numbers('38', 'es') -> 'treinta y ocho'
        expand_token_numbers('9C', 'es') -> 'nueve ce'
        expand_token_numbers('4', 'es')  -> 'cuatro'
        expand_token_numbers('48', 'en') -> 'forty eight'
    """
    if not token or not any(ch.isdigit() for ch in token):
        return token

    t = token.strip()
    is_spanish = lang.lower().startswith("es") if lang else True

    # Check for direct integer
    if t.isdigit():
        val = int(t)
        if 0 <= val <= 9999:
            return int_to_spanish(val) if is_spanish else int_to_english(val)
        return t

    # Alphanumeric like '9C' or '9c'
    m = re.fullmatch(r"(\d+)([a-zA-Z]+)", t)
    if m:
        num_part = int(m.group(1))
        alpha_part = m.group(2).lower()
        num_word = int_to_spanish(num_part) if is_spanish else int_to_english(num_part)
        if is_spanish:
            # Common letter names in Spanish
            letter_map = {"c": "ce", "b": "be", "d": "de", "k": "ka", "m": "eme", "p": "pe", "t": "te"}
            alpha_word = letter_map.get(alpha_part, alpha_part)
        else:
            alpha_word = alpha_part
        return f"{num_word} {alpha_word}"

    # General regex replacement of digits inside token
    def _replace_match(match):
        val = int(match.group(0))
        if 0 <= val <= 9999:
            return int_to_spanish(val) if is_spanish else int_to_english(val)
        return match.group(0)

    return re.sub(r"\b\d+\b", _replace_match, t)
