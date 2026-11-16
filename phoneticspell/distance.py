"""Fast weighted Damerau-Levenshtein edit distances."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from phoneticspell.keyboard import KeyboardLayout


def orthographic_dl(s1: str, s2: str) -> int:
    """Optimal String Alignment (restricted Damerau-Levenshtein) distance."""
    n1, n2 = len(s1), len(s2)
    if n1 == 0:
        return n2
    if n2 == 0:
        return n1
    prev2 = list(range(n2 + 1))
    prev1 = list(range(n2 + 1))
    curr = [0] * (n2 + 1)

    for i in range(1, n1 + 1):
        curr[0] = i
        c1 = s1[i - 1]
        for j in range(1, n2 + 1):
            c2 = s2[j - 1]
            cost = 0 if c1 == c2 else 1
            best = min(
                prev1[j] + 1,      # deletion
                curr[j - 1] + 1,   # insertion
                prev1[j - 1] + cost,
            )
            if i > 1 and j > 1 and c1 == s2[j - 2] and s1[i - 2] == c2:
                trans = prev2[j - 2] + 1
                if trans < best:
                    best = trans
            curr[j] = best
        prev2, prev1, curr = prev1, curr, prev2
    return prev1[n2]


def keyboard_dl(s1: str, s2: str, kb: KeyboardLayout) -> float:
    """Keyboard-weighted Damerau-Levenshtein distance."""
    n1, n2 = len(s1), len(s2)
    if n1 == 0:
        return float(n2)
    if n2 == 0:
        return float(n1)

    prev2 = [float(j) for j in range(n2 + 1)]
    prev1 = [float(j) for j in range(n2 + 1)]
    curr = [0.0] * (n2 + 1)

    for i in range(1, n1 + 1):
        curr[0] = float(i)
        c1 = s1[i - 1]
        # Adjacent-key accidental insertion/deletion is cheaper than random insertion
        del_cost = 0.7
        if i > 1:
            del_cost = min(del_cost, 0.4 + 0.6 * kb.distance(c1, s1[i - 2]))
        if i < n1:
            del_cost = min(del_cost, 0.4 + 0.6 * kb.distance(c1, s1[i]))

        for j in range(1, n2 + 1):
            c2 = s2[j - 1]
            sub_cost = 0.0 if c1 == c2 else kb.distance(c1, c2)
            ins_cost = 0.7
            if j > 1:
                ins_cost = min(ins_cost, 0.4 + 0.6 * kb.distance(c2, s2[j - 2]))
            if j < n2:
                ins_cost = min(ins_cost, 0.4 + 0.6 * kb.distance(c2, s2[j]))

            best = min(
                prev1[j] + del_cost,
                curr[j - 1] + ins_cost,
                prev1[j - 1] + sub_cost,
            )
            if i > 1 and j > 1 and c1 == s2[j - 2] and s1[i - 2] == c2:
                trans = prev2[j - 2] + 0.45
                if trans < best:
                    best = trans
            curr[j] = best
        prev2, prev1, curr = prev1, curr, prev2
    return prev1[n2]


def phonetic_dl(
    p1: Sequence[int],
    p2: Sequence[int],
    sub_matrix: np.ndarray,
    indel_costs: np.ndarray,
) -> float:
    """Feature-weighted phonetic Damerau-Levenshtein using O(1) lookup tables."""
    n1, n2 = len(p1), len(p2)
    if n1 == 0:
        return float(sum(float(indel_costs[pid]) for pid in p2))
    if n2 == 0:
        return float(sum(float(indel_costs[pid]) for pid in p1))

    prev2 = [0.0] * (n2 + 1)
    prev1 = [0.0] * (n2 + 1)
    for j in range(1, n2 + 1):
        prev1[j] = prev1[j - 1] + float(indel_costs[p2[j - 1]])
    curr = [0.0] * (n2 + 1)

    for i in range(1, n1 + 1):
        id1 = p1[i - 1]
        d_cost = float(indel_costs[id1])
        curr[0] = prev1[0] + d_cost
        sub_row = sub_matrix[id1]
        for j in range(1, n2 + 1):
            id2 = p2[j - 1]
            i_cost = float(indel_costs[id2])
            s_cost = 0.0 if id1 == id2 else float(sub_row[id2])
            best = prev1[j] + d_cost
            ins_v = curr[j - 1] + i_cost
            if ins_v < best:
                best = ins_v
            sub_v = prev1[j - 1] + s_cost
            if sub_v < best:
                best = sub_v
            if i > 1 and j > 1 and id1 == p2[j - 2] and p1[i - 2] == id2:
                trans_v = prev2[j - 2] + 0.45
                if trans_v < best:
                    best = trans_v
            curr[j] = best
        prev2, prev1, curr = prev1, curr, prev2
    return prev1[n2]
