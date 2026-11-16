"""Distinctive-feature phone distance tables with O(1) precomputed lookup."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


# ALINE-inspired salience weights for the 24 Hulden/Chomsky-Halle features:
# syll, cons, son, cont, delrel, approx, nas, voi, spread, constr,
# lab, round, labdent, cor, ant, dist, strid, lat, dor, hi, lo, fr, bk, tns
DEFAULT_SALIENCE: Dict[str, float] = {
    "syll": 2.0,
    "cons": 1.8,
    "son": 1.5,
    "cont": 1.5,
    "delrel": 1.0,
    "approx": 1.2,
    "nas": 1.5,
    "voi": 0.6,     # voicing confusion (p/b, t/d, s/z) is very common -> lower penalty
    "spread": 0.5,
    "constr": 0.5,
    "lab": 1.5,
    "round": 0.8,
    "labdent": 1.0,
    "cor": 1.5,
    "ant": 0.8,
    "dist": 0.6,
    "strid": 1.0,
    "lat": 1.0,
    "dor": 1.5,
    "hi": 0.8,
    "lo": 0.8,
    "fr": 0.8,
    "bk": 0.8,
    "tns": 0.5,     # tense/lax vowel confusion (i/ɪ, e/ɛ) is common -> lower penalty
}


DEFAULT_FEATURES_CSV = Path(__file__).resolve().parent / "data" / "distinctivefeatures_ipa.csv"


class DistinctiveFeatures:
    """Loads an IPA distinctive-feature CSV and precomputes phone-to-phone distances."""

    def __init__(
        self,
        csv_path: str | Path = DEFAULT_FEATURES_CSV,
        metric: str = "salience",
        salience: Dict[str, float] | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.metric = metric
        self.phones: List[str] = []
        self.phone_to_id: Dict[str, int] = {}
        self.feature_names: List[str] = []
        self.vectors: np.ndarray = np.empty((0, 0), dtype=np.float32)

        self._load_csv(self.csv_path)
        self.sub_matrix, self.indel_costs = self._precompute_distances(
            metric=metric, salience=salience or DEFAULT_SALIENCE
        )
        # Precompute longest-match tokenizer order (e.g. 'tʃ' before 't')
        self._sorted_phones = sorted(self.phones, key=len, reverse=True)
        # Coarse articulatory class map for fast phonetic-skeleton indexing
        self.coarse_class: Dict[str, str] = self._build_coarse_classes()

    def _load_csv(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = [col.strip() for col in next(reader)]
            self.feature_names = header[1:]
            rows: List[List[float]] = []
            for row in reader:
                if not row or not row[0].strip():
                    continue
                phon = row[0].strip()
                if phon in self.phone_to_id:
                    continue
                vec = [float(x.strip()) for x in row[1:]]
                self.phone_to_id[phon] = len(self.phones)
                self.phones.append(phon)
                rows.append(vec)
        self.vectors = np.asarray(rows, dtype=np.float32)

    def _precompute_distances(
        self, metric: str, salience: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        n, f = self.vectors.shape
        if metric == "l1":
            weights = np.ones(f, dtype=np.float32)
        elif metric == "salience":
            weights = np.array(
                [salience.get(name, 1.0) for name in self.feature_names],
                dtype=np.float32,
            )
        elif metric == "cosine":
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            normed = self.vectors / norms
            sim = normed @ normed.T
            sub = np.clip(1.0 - sim, 0.0, 2.0) / 2.0
            np.fill_diagonal(sub, 0.0)
            indel = np.full(n, 0.5, dtype=np.float32)
            return sub.astype(np.float32), indel
        else:
            raise ValueError(f"Unknown phone distance metric: {metric!r}")

        # Weighted L1 over {-1, 0, +1}: max feature diff is 2.0 per feature
        max_total = float(2.0 * np.sum(weights))
        diff = np.abs(self.vectors[:, None, :] - self.vectors[None, :, :])
        sub = np.tensordot(diff, weights, axes=([2], [0])) / max_total
        np.fill_diagonal(sub, 0.0)

        # Indel cost proportional to feature density (vowels/glides cheaper to drop than stops)
        indel = (
            np.sum(np.abs(self.vectors) * weights[None, :], axis=1) / max_total
        ).astype(np.float32)
        # Keep indel in a sensible range around [0.35, 0.75]
        indel = np.clip(indel * 1.4, 0.35, 0.75)
        return sub.astype(np.float32), indel

    def _build_coarse_classes(self) -> Dict[str, str]:
        """Map each IPA phone to a single-char broad articulatory class for indexing."""
        idx = {name: i for i, name in enumerate(self.feature_names)}
        classes: Dict[str, str] = {}
        for p, pid in self.phone_to_id.items():
            v = self.vectors[pid]
            if v[idx["syll"]] > 0:
                classes[p] = "V"  # vowel / syllabic
            elif v[idx["nas"]] > 0:
                classes[p] = "N"  # nasal
            elif v[idx["approx"]] > 0 or v[idx["lat"]] > 0:
                classes[p] = "L"  # liquid / glide
            elif v[idx["strid"]] > 0:
                classes[p] = "S"  # sibilant / affricate
            elif v[idx["lab"]] > 0 or v[idx["labdent"]] > 0:
                classes[p] = "P"  # labial / labiodental
            elif v[idx["cor"]] > 0:
                classes[p] = "T"  # coronal
            else:
                classes[p] = "K"  # dorsal / glottal
        return classes

    def tokenize(self, ipa_str: str) -> List[int]:
        """Segment an IPA string into known phone IDs using greedy longest match."""
        ids: List[int] = []
        i = 0
        n = len(ipa_str)
        while i < n:
            matched = False
            for ph in self._sorted_phones:
                if ipa_str.startswith(ph, i):
                    ids.append(self.phone_to_id[ph])
                    i += len(ph)
                    matched = True
                    break
            if not matched:
                i += 1
        return ids

    def skeleton(self, phone_ids: Sequence[int]) -> str:
        """Convert phone IDs into a collapsed coarse-class skeleton (e.g. V/N/L/S/P/T/K)."""
        out: List[str] = []
        for pid in phone_ids:
            cls = self.coarse_class[self.phones[pid]]
            if not out or out[-1] != cls:
                out.append(cls)
        return "".join(out)
