"""Unified phonetic + keyboard + frequency spellchecker."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from phoneticspell.candidates import CandidateIndex
from phoneticspell.distance import keyboard_dl, orthographic_dl, phonetic_dl
from phoneticspell.features import DistinctiveFeatures
from phoneticspell.g2p import EnglishG2P
from phoneticspell.keyboard import KeyboardLayout


_WORD_RE = re.compile(r"[a-zA-Z]+")


def load_corpus_counts(corpus_path: str | Path) -> Counter[str]:
    """Count lowercase alphabetic tokens in a text corpus."""
    text = Path(corpus_path).read_text(encoding="utf-8", errors="ignore")
    return Counter(w.lower() for w in _WORD_RE.findall(text))


def merge_wordlist(
    counts: Counter[str], wordlist_path: str | Path, floor: int = 1
) -> Counter[str]:
    """Ensure all pure-alphabetic words in `wordlist_path` exist with at least `floor`."""
    merged = Counter(counts)
    with Path(wordlist_path).open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            w = line.strip()
            if w.isalpha() and w.isascii():
                wl = w.lower()
                if wl not in merged:
                    merged[wl] = floor
    return merged


class SpellChecker:
    """Multi-mode spellchecker supporting Norvig, Keyboard, Phonetic, and Unified scoring."""

    def __init__(
        self,
        word_counts: Counter[str],
        features: DistinctiveFeatures,
        g2p: EnglishG2P,
        keyboard: KeyboardLayout | None = None,
        min_freq: int = 1,
        max_orth_edit: int = 2,
    ) -> None:
        self.counts: Counter[str] = Counter(
            {w: c for w, c in word_counts.items() if c >= min_freq}
        )
        self.total_tokens = max(sum(self.counts.values()), 1)
        self.log_total = math.log(self.total_tokens)
        self.features = features
        self.g2p = g2p
        self.keyboard = keyboard or KeyboardLayout()
        self.index = CandidateIndex(
            self.counts.keys(),
            features=self.features,
            g2p=self.g2p,
            max_orth_edit=max_orth_edit,
            word_counts=self.counts,
        )

    def log_prob(self, word: str) -> float:
        """Smoothed log unigram probability log P(word)."""
        c = self.counts.get(word.lower(), 0)
        return math.log(c + 0.5) - self.log_total

    def extract_candidate_features(
        self,
        word: str,
        mode: str = "unified",
    ) -> List[Tuple[str, Tuple[float, float, float, float]]]:
        """Return `(candidate, (log_prob, -d_orth, -d_kb, -d_phon))` for each candidate."""
        w_low = word.lower()
        if w_low in self.counts:
            return [(w_low, (self.log_prob(w_low), 0.0, 0.0, 0.0))]

        orth_cands = self.index.query_orthographic(w_low)
        if mode in ("norvig", "keyboard"):
            cands = orth_cands
        elif mode in ("phonetic", "unified"):
            cands = orth_cands | self.index.query_phonetic(w_low)
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        if not cands:
            return [(w_low, (float("-inf"), 0.0, 0.0, 0.0))]

        query_phones = (
            self.features.tokenize(self.g2p.transcribe(w_low))
            if mode in ("phonetic", "unified")
            else []
        )

        out: List[Tuple[str, Tuple[float, float, float, float]]] = []
        for c in cands:
            lp = self.log_prob(c)
            d_orth = float(orthographic_dl(w_low, c)) if mode in ("norvig", "unified") else 0.0
            d_kb = keyboard_dl(w_low, c, self.keyboard) if mode in ("keyboard", "unified") else 0.0
            if mode in ("phonetic", "unified"):
                c_phones = self.index.word_phones.get(c, [])
                d_ph = phonetic_dl(
                    query_phones,
                    c_phones,
                    self.features.sub_matrix,
                    self.features.indel_costs,
                )
            else:
                d_ph = 0.0
            out.append((c, (lp, -d_orth, -d_kb, -d_ph)))
        return out

    def rank_candidates(
        self,
        word: str,
        mode: str = "unified",
        weights: Tuple[float, float, float, float] = (1.0, 4.0, 3.0, 2.0),
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Return top-k `(candidate, score)` pairs sorted descending by score.

        weights = (w_freq, w_orth, w_kb, w_phon)
        """
        w_low = word.lower()
        if w_low in self.counts:
            return [(w_low, 0.0)]

        feats = self.extract_candidate_features(w_low, mode=mode)
        if not feats or feats[0][1][0] == float("-inf"):
            return [(w_low, float("-inf"))]

        if mode == "norvig":
            # Classic Norvig: prefer edit distance 1 over 2, then highest frequency
            scored = [
                (c, 100.0 * neg_d_orth + lp)
                for c, (lp, neg_d_orth, _, _) in feats
                if -neg_d_orth <= 2.0
            ]
            if not scored:
                scored = [(c, 100.0 * neg_d_orth + lp) for c, (lp, neg_d_orth, _, _) in feats]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]

        w_freq, w_orth, w_kb, w_phon = weights
        scored = [
            (
                c,
                w_freq * lp
                + w_orth * neg_d_orth
                + w_kb * neg_d_kb
                + w_phon * neg_d_ph,
            )
            for c, (lp, neg_d_orth, neg_d_kb, neg_d_ph) in feats
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def correct(
        self,
        word: str,
        mode: str = "unified",
        weights: Tuple[float, float, float, float] = (1.0, 4.0, 3.0, 2.0),
    ) -> str:
        """Return the top-1 correction for `word`."""
        return self.rank_candidates(word, mode=mode, weights=weights, top_k=1)[0][0]
