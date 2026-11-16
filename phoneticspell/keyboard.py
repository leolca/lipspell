"""Keyboard layout coordinates and precomputed O(1) key-distance matrix."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


QWERTY_ROWS = [
    "`1234567890-=",
    "qwertyuiop[]\\",
    "asdfghjkl;'",
    "zxcvbnm,./",
]

# Horizontal stagger per row on a physical QWERTY keyboard
ROW_OFFSETS = [0.0, 0.5, 0.75, 1.25]


class KeyboardLayout:
    """Computes normalized physical key-to-key distances."""

    def __init__(
        self,
        coords: Dict[str, Tuple[float, float]] | None = None,
        saturation: float = 3.0,
    ) -> None:
        self.saturation = saturation
        self.coords: Dict[str, Tuple[float, float]] = coords or self._default_qwerty()
        self.char_to_id: Dict[str, int] = {
            ch: idx for idx, ch in enumerate(sorted(self.coords))
        }
        self.matrix = self._precompute_matrix()

    @staticmethod
    def _default_qwerty() -> Dict[str, Tuple[float, float]]:
        coords: Dict[str, Tuple[float, float]] = {}
        for r_idx, row in enumerate(QWERTY_ROWS):
            offset = ROW_OFFSETS[r_idx]
            for c_idx, ch in enumerate(row):
                coords[ch] = (float(r_idx), float(c_idx) + offset)
        return coords

    @classmethod
    def abnt2(cls, saturation: float = 3.0) -> "KeyboardLayout":
        """Brazilian Portuguese ABNT2 keyboard layout (adds 'ç' and accents)."""
        abnt2_rows = [
            "'1234567890-=",
            "qwertyuiop´[",
            "asdfghjklç~]",
            "\\zxcvbnm,.;/",
        ]
        coords: Dict[str, Tuple[float, float]] = {}
        for r_idx, row in enumerate(abnt2_rows):
            offset = ROW_OFFSETS[r_idx]
            for c_idx, ch in enumerate(row):
                coords[ch] = (float(r_idx), float(c_idx) + offset)
        return cls(coords=coords, saturation=saturation)

    @classmethod
    def from_json(cls, json_path: str | Path, saturation: float = 3.0) -> "KeyboardLayout":
        with Path(json_path).open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Support [lowercase_rows, uppercase_rows] format from qwertyKeymap.json
        rows = data[0] if data and isinstance(data[0], list) and isinstance(data[0][0], list) else data
        coords: Dict[str, Tuple[float, float]] = {}
        for r_idx, row in enumerate(rows):
            offset = ROW_OFFSETS[r_idx] if r_idx < len(ROW_OFFSETS) else 0.0
            for c_idx, ch in enumerate(row):
                if ch and ch != " ":
                    coords[ch.lower()] = (float(r_idx), float(c_idx) + offset)
        return cls(coords=coords, saturation=saturation)

    def _precompute_matrix(self) -> np.ndarray:
        n = len(self.char_to_id)
        mat = np.ones((n, n), dtype=np.float32)
        chars = [None] * n
        for ch, idx in self.char_to_id.items():
            chars[idx] = ch
        for i in range(n):
            mat[i, i] = 0.0
            x1, y1 = self.coords[chars[i]]  # type: ignore[index]
            for j in range(i + 1, n):
                x2, y2 = self.coords[chars[j]]  # type: ignore[index]
                dist = min(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5, self.saturation)
                # Normalize to [0, 1] where adjacent key (~1.0) costs ~0.33
                norm_dist = dist / self.saturation
                mat[i, j] = norm_dist
                mat[j, i] = norm_dist
        return mat

    def distance(self, c1: str, c2: str) -> float:
        c1, c2 = c1.lower(), c2.lower()
        if c1 == c2:
            return 0.0
        i = self.char_to_id.get(c1)
        j = self.char_to_id.get(c2)
        if i is None or j is None:
            return 1.0
        return float(self.matrix[i, j])
