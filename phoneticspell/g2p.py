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
    ("lh", "l"),
    ("nh", "n"),
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
