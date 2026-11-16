"""Fast candidate generation via SymSpell deletes and coarse phonetic skeletons."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Set

from phoneticspell.features import DistinctiveFeatures
from phoneticspell.g2p import EnglishG2P


def _generate_deletes(word: str, max_dist: int) -> Set[str]:
    """Return all strings formed by deleting 0..max_dist chars from `word`."""
    deletes: Set[str] = {word}
    queue = [word]
    for _ in range(max_dist):
        next_q: List[str] = []
        for item in queue:
            if len(item) <= 1:
                continue
            for i in range(len(item)):
                d = item[:i] + item[i + 1 :]
                if d not in deletes:
                    deletes.add(d)
                    next_q.append(d)
        queue = next_q
    return deletes


class CandidateIndex:
    """Indexes a vocabulary for fast orthographic and phonetic candidate retrieval."""

    def __init__(
        self,
        vocab: Iterable[str],
        features: DistinctiveFeatures,
        g2p: EnglishG2P,
        max_orth_edit: int = 2,
        word_counts: Dict[str, int] | None = None,
    ) -> None:
        self.features = features
        self.g2p = g2p
        self.max_orth_edit = max_orth_edit
        self.word_counts = word_counts or {}
        self.vocab: Set[str] = set()
        self.delete_index: Dict[str, List[str]] = defaultdict(list)
        self.skel_index: Dict[str, List[str]] = defaultdict(list)
        self.word_phones: Dict[str, List[int]] = {}

        self._build(vocab)

    def _build(self, vocab: Iterable[str]) -> None:
        for word in vocab:
            w = word.lower()
            if not w or w in self.vocab:
                continue
            self.vocab.add(w)
            for d in _generate_deletes(w, self.max_orth_edit):
                self.delete_index[d].append(w)

            ipa = self.g2p.transcribe(w)
            pids = self.features.tokenize(ipa)
            self.word_phones[w] = pids
            skel = self.features.skeleton(pids)
            if skel:
                self.skel_index[skel].append(w)
                # Also index 1-delete of skeleton for missing/extra broad class
                for d in _generate_deletes(skel, 1):
                    if d != skel:
                        self.skel_index[d].append(w)

        if self.word_counts:
            for key in self.skel_index:
                self.skel_index[key].sort(
                    key=lambda word: self.word_counts.get(word, 0), reverse=True
                )

    def query_orthographic(self, word: str) -> Set[str]:
        """Return candidate words from the symmetric delete index."""
        w = word.lower()
        if w in self.vocab:
            return {w}
        hits: Set[str] = set()
        for d in _generate_deletes(w, self.max_orth_edit):
            if d in self.delete_index:
                hits.update(self.delete_index[d])
        return hits

    def query_phonetic(self, word: str, max_hits: int = 300) -> Set[str]:
        """Return candidate words sharing or near the query's phonetic skeleton."""
        w = word.lower()
        ipa = self.g2p.transcribe(w)
        pids = self.features.tokenize(ipa)
        skel = self.features.skeleton(pids)
        if not skel:
            return set()
        hits: Set[str] = set()
        for d in _generate_deletes(skel, 1):
            if d in self.skel_index:
                for cand in self.skel_index[d]:
                    hits.add(cand)
                    if len(hits) >= max_hits:
                        return hits
        return hits
