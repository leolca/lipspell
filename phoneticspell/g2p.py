"""Grapheme-to-phoneme (G2P) backends emitting IPA phone sequences."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


ARPABET_TO_IPA: Dict[str, str] = {
    "AA": "ɑ",
    "AE": "æ",
    "AH": "ʌ",
    "AO": "ɔ",
    "AW": "aʊ",
    "AX": "ə",
    "AXR": "ɚ",
    "AY": "aɪ",
    "EH": "ɛ",
    "ER": "ɝ",
    "EY": "eɪ",
    "IH": "ɪ",
    "IX": "ɨ",
    "IY": "i",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "UH": "ʊ",
    "UW": "u",
    "B": "b",
    "CH": "tʃ",
    "D": "d",
    "DH": "ð",
    "DX": "ɾ",
    "F": "f",
    "G": "ɡ",
    "HH": "h",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "P": "p",
    "R": "ɹ",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "V": "v",
    "W": "w",
    "WH": "ʍ",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
}

# Ordered longest-match English grapheme-to-IPA rules for OOV / misspelled words
_G2P_MULTI_RULES: List[Tuple[str, str]] = [
    ("tion", "ʃən"),
    ("sion", "ʒən"),
    ("ough", "oʊ"),
    ("augh", "ɔ"),
    ("eigh", "eɪ"),
    ("igh", "aɪ"),
    ("tch", "tʃ"),
    ("dge", "dʒ"),
    ("sch", "sk"),
    ("qu", "kw"),
    ("ph", "f"),
    ("sh", "ʃ"),
    ("ch", "tʃ"),
    ("th", "θ"),
    ("wh", "w"),
    ("ng", "ŋ"),
    ("ck", "k"),
    ("dg", "dʒ"),
    ("gh", ""),
    ("kn", "n"),
    ("gn", "n"),
    ("wr", "ɹ"),
    ("ps", "s"),
    ("ee", "i"),
    ("ea", "i"),
    ("oo", "u"),
    ("ou", "aʊ"),
    ("ow", "oʊ"),
    ("oi", "ɔɪ"),
    ("oy", "ɔɪ"),
    ("ai", "eɪ"),
    ("ay", "eɪ"),
    ("ei", "eɪ"),
    ("ey", "eɪ"),
    ("au", "ɔ"),
    ("aw", "ɔ"),
    ("ar", "ɑɹ"),
    ("er", "ɚ"),
    ("ir", "ɝ"),
    ("or", "ɔɹ"),
    ("ur", "ɝ"),
]

_G2P_SINGLE: Dict[str, str] = {
    "a": "æ",
    "b": "b",
    "c": "k",
    "d": "d",
    "e": "ɛ",
    "f": "f",
    "g": "ɡ",
    "h": "h",
    "i": "ɪ",
    "j": "dʒ",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "o": "ɑ",
    "p": "p",
    "q": "k",
    "r": "ɹ",
    "s": "s",
    "t": "t",
    "u": "ʌ",
    "v": "v",
    "w": "w",
    "x": "ks",
    "y": "j",
    "z": "z",
}

_STRESS_RE = re.compile(r"[0-2]$")


_LONG_VOWEL_E: Dict[str, str] = {
    "a": "eɪ",
    "e": "i",
    "i": "aɪ",
    "o": "oʊ",
    "u": "u",
}


class EnglishG2P:
    """Fast English G2P: CMUdict lookup with rule-based fallback for typos/OOV."""

    def __init__(self, cmudict_path: str | Path | None = None) -> None:
        self.lexicon: Dict[str, str] = {}
        self._cache: Dict[str, str] = {}
        if cmudict_path is not None:
            self._load_cmudict(Path(cmudict_path))

    def _load_cmudict(self, path: Path) -> None:
        with path.open("r", encoding="latin-1") as fh:
            for line in fh:
                if not line or line.startswith(";;;"):
                    continue
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                word = parts[0]
                # Skip alternate pronunciations like WORD(1)
                if "(" in word:
                    continue
                word_l = word.lower()
                ipa_parts: List[str] = []
                for phone in parts[1:]:
                    base = _STRESS_RE.sub("", phone)
                    ipa = ARPABET_TO_IPA.get(base)
                    if ipa:
                        ipa_parts.append(ipa)
                if ipa_parts:
                    self.lexicon[word_l] = "".join(ipa_parts)

    def transcribe(self, word: str) -> str:
        """Return an IPA transcription string for `word`."""
        w = word.lower()
        if w in self.lexicon:
            return self.lexicon[w]
        cached = self._cache.get(w)
        if cached is not None:
            return cached
        ipa = self._rule_based(w)
        self._cache[w] = ipa
        return ipa

    @staticmethod
    def _rule_based(word: str) -> str:
        w = re.sub(r"[^a-z]", "", word.lower())
        if not w:
            return ""
        # Collapse consecutive duplicate consonants/letters except 'ee', 'oo'
        out: List[str] = []
        i = 0
        n = len(w)
        while i < n:
            # Silent final 'e' on words longer than 3 chars
            if i == n - 1 and w[i] == "e" and n > 3 and w[i - 1] not in "aeiou":
                i += 1
                continue
            matched = False
            for pat, repl in _G2P_MULTI_RULES:
                if w.startswith(pat, i):
                    out.append(repl)
                    i += len(pat)
                    matched = True
                    break
            if matched:
                continue
            ch = w[i]
            # Magic-e (VCe) at end of word, e.g. 'nite' -> naɪt, 'foen' handled by multi-rules
            if (
                ch in _LONG_VOWEL_E
                and i + 2 == n - 1
                and w[i + 1] not in "aeiou"
                and w[i + 2] == "e"
            ):
                out.append(_LONG_VOWEL_E[ch])
                i += 1
                continue
            # Soft c / g before e, i, y
            if ch == "c" and i + 1 < n and w[i + 1] in "eiy":
                out.append("s")
            elif ch == "g" and i + 1 < n and w[i + 1] in "eiy":
                out.append("dʒ")
            else:
                out.append(_G2P_SINGLE.get(ch, ""))
            # Skip doubled letter (e.g. 'accommodate' vs 'acommodate' -> same consonant)
            if i + 1 < n and w[i + 1] == ch and ch not in "eo":
                i += 2
            else:
                i += 1
        return "".join(out)


_PT_MULTI_RULES: List[Tuple[str, str]] = [
    ("ch", "ʃ"),
    ("lh", "ʎ"),
    ("nh", "ɲ"),
    ("rr", "h"),
    ("ss", "s"),
    ("qu", "k"),
    ("gu", "ɡ"),
    ("ão", "aʊ"),
    ("õe", "oɪ"),
    ("ãe", "aɪ"),
    ("ou", "o"),
    ("ei", "eɪ"),
    ("ai", "aɪ"),
    ("oi", "ɔɪ"),
    ("au", "aʊ"),
]

_PT_VOWELS = set("aeiouáàâãéêíóôõú")

_PT_SINGLE: Dict[str, str] = {
    "a": "a",
    "á": "a",
    "à": "a",
    "â": "ɐ",
    "ã": "ɐ",
    "b": "b",
    "c": "k",
    "ç": "s",
    "d": "d",
    "e": "e",
    "é": "ɛ",
    "ê": "e",
    "f": "f",
    "g": "ɡ",
    "h": "",
    "i": "i",
    "í": "i",
    "j": "ʒ",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "o": "o",
    "ó": "ɔ",
    "ô": "o",
    "õ": "o",
    "p": "p",
    "q": "k",
    "r": "ɹ",
    "s": "s",
    "t": "t",
    "u": "u",
    "ú": "u",
    "v": "v",
    "w": "w",
    "x": "ʃ",
    "y": "i",
    "z": "z",
}


class PortugueseG2P:
    """Rule-based Brazilian Portuguese G2P emitting IPA phones in the feature table."""

    def __init__(self) -> None:
        self._cache: Dict[str, str] = {}

    def transcribe(self, word: str) -> str:
        w = word.lower()
        cached = self._cache.get(w)
        if cached is not None:
            return cached
        ipa = self._rule_based(w)
        self._cache[w] = ipa
        return ipa

    @staticmethod
    def _rule_based(word: str) -> str:
        w = "".join(ch for ch in word.lower() if ch in _PT_SINGLE)
        if not w:
            return ""
        out: List[str] = []
        i = 0
        n = len(w)
        while i < n:
            # 'sc' / 'xc' before e, i -> s
            if w.startswith(("sc", "xc"), i) and i + 2 < n and w[i + 2] in "eéêií":
                out.append("s")
                i += 2
                continue
            matched = False
            for pat, repl in _PT_MULTI_RULES:
                if w.startswith(pat, i):
                    out.append(repl)
                    i += len(pat)
                    matched = True
                    break
            if matched:
                continue

            ch = w[i]
            if ch == "c" and i + 1 < n and w[i + 1] in "eéêií":
                out.append("s")
            elif ch == "g" and i + 1 < n and w[i + 1] in "eéêií":
                out.append("ʒ")
            elif ch == "s" and 0 < i < n - 1 and w[i - 1] in _PT_VOWELS and w[i + 1] in _PT_VOWELS:
                out.append("z")
            elif ch == "r" and i == 0:
                out.append("h")
            else:
                out.append(_PT_SINGLE.get(ch, ""))
            i += 1
        return "".join(out)


_TR_SINGLE: Dict[str, str] = {
    "a": "a",
    "â": "a",
    "b": "b",
    "c": "dʒ",
    "ç": "tʃ",
    "d": "d",
    "e": "e",
    "f": "f",
    "g": "ɡ",
    "ğ": "",
    "h": "h",
    "ı": "ɯ",
    "i": "i",
    "î": "i",
    "j": "ʒ",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "o": "o",
    "ö": "ø",
    "p": "p",
    "r": "ɾ",
    "s": "s",
    "ş": "ʃ",
    "t": "t",
    "u": "u",
    "û": "u",
    "ü": "y",
    "v": "v",
    "y": "j",
    "z": "z",
}


class TurkishG2P:
    """Rule-based Turkish G2P emitting IPA phones in the feature table."""

    def __init__(self) -> None:
        self._cache: Dict[str, str] = {}

    def transcribe(self, word: str) -> str:
        # Handle Turkish uppercase İ/I properly before lowercasing
        w = word.replace("İ", "i").replace("I", "ı").lower()
        cached = self._cache.get(w)
        if cached is not None:
            return cached
        ipa = "".join(_TR_SINGLE.get(ch, "") for ch in w)
        self._cache[w] = ipa
        return ipa


_FR_MULTI_RULES: List[Tuple[str, str]] = [
    ("eaux", "o"),
    ("eau", "o"),
    ("tion", "sjɔn"),
    ("œu", "ø"),
    ("au", "o"),
    ("ou", "u"),
    ("eu", "ø"),
    ("oi", "wa"),
    ("ai", "ɛ"),
    ("ei", "ɛ"),
    ("ch", "ʃ"),
    ("ph", "f"),
    ("gn", "ɲ"),
    ("qu", "k"),
    ("gu", "ɡ"),
    ("ss", "s"),
]

_FR_VOWELS = set("aeiouyàâäéèêëîïôöùûüœ")

_FR_SINGLE: Dict[str, str] = {
    "a": "a",
    "à": "a",
    "â": "ɑ",
    "ä": "a",
    "b": "b",
    "c": "k",
    "ç": "s",
    "d": "d",
    "e": "ə",
    "é": "e",
    "è": "ɛ",
    "ê": "ɛ",
    "ë": "ɛ",
    "f": "f",
    "g": "ɡ",
    "h": "",
    "i": "i",
    "î": "i",
    "ï": "i",
    "j": "ʒ",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "o": "o",
    "ô": "o",
    "ö": "ø",
    "œ": "ø",
    "p": "p",
    "q": "k",
    "r": "ɹ",
    "s": "s",
    "t": "t",
    "u": "y",
    "ù": "y",
    "û": "y",
    "ü": "y",
    "v": "v",
    "w": "w",
    "x": "ks",
    "y": "i",
    "z": "z",
}


class FrenchG2P:
    """Rule-based French G2P approximation emitting IPA phones in the feature table."""

    def __init__(self) -> None:
        self._cache: Dict[str, str] = {}

    def transcribe(self, word: str) -> str:
        w = word.lower()
        cached = self._cache.get(w)
        if cached is not None:
            return cached
        ipa = self._rule_based(w)
        self._cache[w] = ipa
        return ipa

    @staticmethod
    def _rule_based(word: str) -> str:
        w = "".join(ch for ch in word.lower() if ch in _FR_SINGLE)
        if not w:
            return ""
        # Strip common silent final consonants/e on words > 3 chars
        if len(w) > 3 and w.endswith(("s", "t", "d", "x", "p", "z")):
            w = w[:-1]
        if len(w) > 3 and w.endswith("e"):
            w = w[:-1]
        out: List[str] = []
        i = 0
        n = len(w)
        while i < n:
            matched = False
            for pat, repl in _FR_MULTI_RULES:
                if w.startswith(pat, i):
                    out.append(repl)
                    i += len(pat)
                    matched = True
                    break
            if matched:
                continue
            ch = w[i]
            if ch == "c" and i + 1 < n and w[i + 1] in "eéèêëiîïy":
                out.append("s")
            elif ch == "g" and i + 1 < n and w[i + 1] in "eéèêëiîïy":
                out.append("ʒ")
            elif ch == "s" and 0 < i < n - 1 and w[i - 1] in _FR_VOWELS and w[i + 1] in _FR_VOWELS:
                out.append("z")
            else:
                out.append(_FR_SINGLE.get(ch, ""))
            # Collapse doubled consonants (e.g. 'emmenager' vs 'eménager')
            if i + 1 < n and w[i + 1] == ch and ch not in "aeiouy":
                i += 2
            else:
                i += 1
        return "".join(out)


_RU_SINGLE: Dict[str, str] = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "ɡ",
    "д": "d",
    "е": "je",
    "ё": "jo",
    "ж": "ʒ",
    "з": "z",
    "и": "i",
    "й": "j",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "ɾ",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "x",
    "ц": "ts",
    "ч": "tʃ",
    "ш": "ʃ",
    "щ": "ʃ",
    "ъ": "",
    "ы": "ɨ",
    "ь": "",
    "э": "e",
    "ю": "ju",
    "я": "ja",
}


class RussianG2P:
    """Rule-based Russian G2P approximation emitting IPA phones in the feature table."""

    def __init__(self) -> None:
        self._cache: Dict[str, str] = {}

    def transcribe(self, word: str) -> str:
        w = word.lower()
        cached = self._cache.get(w)
        if cached is not None:
            return cached
        out: List[str] = []
        i = 0
        n = len(w)
        while i < n:
            if w.startswith("ого", i) and i + 3 == n:
                out.append("ovo")
                i += 3
                continue
            if w.startswith("его", i) and i + 3 == n:
                out.append("jevo")
                i += 3
                continue
            out.append(_RU_SINGLE.get(w[i], ""))
            i += 1
        ipa = "".join(out)
        self._cache[w] = ipa
        return ipa


_HE_SINGLE: Dict[str, str] = {
    "א": "ʔ",
    "ב": "b",
    "ג": "ɡ",
    "ד": "d",
    "ה": "h",
    "ו": "v",
    "ז": "z",
    "ח": "x",
    "ט": "t",
    "י": "j",
    "כ": "k",
    "ך": "x",
    "ל": "l",
    "מ": "m",
    "ם": "m",
    "נ": "n",
    "ן": "n",
    "ס": "s",
    "ע": "ʔ",
    "פ": "p",
    "ף": "f",
    "צ": "ts",
    "ץ": "ts",
    "ק": "k",
    "ר": "ɾ",
    "ש": "ʃ",
    "ת": "t",
}


class HebrewG2P:
    """Rule-based Hebrew abjad G2P approximation emitting IPA phones."""

    def __init__(self) -> None:
        self._cache: Dict[str, str] = {}

    def transcribe(self, word: str) -> str:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        ipa = "".join(_HE_SINGLE.get(ch, "") for ch in word)
        self._cache[word] = ipa
        return ipa
