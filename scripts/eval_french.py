"""Evaluate Norvig, Keyboard, Phonetic, and Unified on French Wikipedia typos."""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path
from typing import List, Tuple

from scipy.stats import binomtest

from phoneticspell.features import DistinctiveFeatures
from phoneticspell.g2p import FrenchG2P
from phoneticspell.keyboard import KeyboardLayout
from phoneticspell.spellchecker import SpellChecker
from scripts.benchmark import evaluate
from scripts.eval_portuguese import load_tsv_pairs


def load_fr_counts(freq_path: Path, max_words: int = 200_000) -> Counter[str]:
    counts: Counter[str] = Counter()
    with freq_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            word, cnt_s = parts[0].lower(), parts[1]
            if word.isalpha() and cnt_s.isdigit():
                counts[word] += int(cnt_s)
                if len(counts) >= max_words:
                    break
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fr-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "fr",
    )
    parser.add_argument("--max-words", type=int, default=200_000)
    args = parser.parse_args()

    t0 = time.perf_counter()
    counts = load_fr_counts(args.fr_dir / "fr_full.txt", max_words=args.max_words)
    features = DistinctiveFeatures(metric="cosine")
    g2p = FrenchG2P()
    kb = KeyboardLayout.azerty()
    checker = SpellChecker(counts, features=features, g2p=g2p, keyboard=kb, min_freq=1)
    print(f"Indexed {len(checker.counts)} French words in {time.perf_counter() - t0:.2f}s")

    pairs = load_tsv_pairs(args.fr_dir / "fr_wikipedia_typos.tsv")
    nonword_pairs = [(x, t) for x, t in pairs if x not in counts and t in counts]
    print(
        f"French Wikipedia typos: total={len(pairs)}, "
        f"clean non-word in vocab={len(nonword_pairs)}"
    )

    weights = (1.0, 4.0, 1.5, 4.0)
    print(f"{'mode':<12} {'top1 %':>8} {'top5 %':>8} {'ms/word':>9}")
    for mode in ("norvig", "keyboard", "phonetic", "unified"):
        res = evaluate(checker, nonword_pairs, mode=mode, weights=weights)
        print(f"{mode:<12} {res['top1']:>8.2f} {res['top5']:>8.2f} {res['ms_per_word']:>9.2f}")

    b01 = b10 = 0
    for x, t in nonword_pairs:
        u = checker.correct(x, mode="unified", weights=weights) == t
        n = checker.correct(x, mode="norvig", weights=weights) == t
        if u and not n:
            b01 += 1
        elif n and not u:
            b10 += 1
    p = float(binomtest(min(b01, b10), n=b01 + b10, p=0.5, alternative="two-sided").pvalue) if (b01 + b10) else 1.0
    print(f"McNemar unified vs norvig: b01={b01}, b10={b10}, p={p:.4g}")


if __name__ == "__main__":
    main()
