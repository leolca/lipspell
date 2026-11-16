"""Ablation benchmark comparing Norvig, Keyboard, Phonetic, and Unified modes."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

from phoneticspell.features import DistinctiveFeatures
from phoneticspell.g2p import EnglishG2P
from phoneticspell.keyboard import KeyboardLayout
from phoneticspell.spellchecker import SpellChecker, load_corpus_counts, merge_wordlist


def load_testset(path: Path, max_items: int | None = None) -> List[Tuple[str, str]]:
    """Parse `target: typo1 typo2 ...` testset files into `(typo, target)` pairs."""
    pairs: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line or ":" not in line:
                continue
            target, typos = line.split(":", 1)
            target = target.strip().lower()
            if not target.isalpha():
                continue
            for typo in typos.split():
                typo = typo.strip().lower()
                if typo.isalpha():
                    pairs.append((typo, target))
                    if max_items is not None and len(pairs) >= max_items:
                        return pairs
    return pairs


def evaluate(
    checker: SpellChecker,
    pairs: List[Tuple[str, str]],
    mode: str,
    weights: Tuple[float, float, float, float] = (0.25, 1.0, 2.0, 2.5),
) -> Dict[str, float]:
    t0 = time.perf_counter()
    top1 = 0
    top5 = 0
    for typo, target in pairs:
        ranked = [w for w, _ in checker.rank_candidates(typo, mode=mode, weights=weights, top_k=5)]
        if ranked and ranked[0] == target:
            top1 += 1
        if target in ranked:
            top5 += 1
    elapsed = time.perf_counter() - t0
    n = max(len(pairs), 1)
    return {
        "n": len(pairs),
        "top1": 100.0 * top1 / n,
        "top5": 100.0 * top5 / n,
        "ms_per_word": 1000.0 * elapsed / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/leoca/ee/research/spellcheck/data"),
    )
    parser.add_argument("--metric", default="salience", choices=["l1", "salience", "cosine"])
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--wordlist", type=Path, default=None)
    parser.add_argument(
        "--weights",
        type=float,
        nargs=4,
        default=[1.0, 4.0, 1.5, 4.0],
        metavar=("W_FREQ", "W_ORTH", "W_KB", "W_PHON"),
    )
    args = parser.parse_args()
    weights = tuple(args.weights)

    t0 = time.perf_counter()
    counts = load_corpus_counts(args.data_dir / "big.txt")
    if args.wordlist is not None and args.wordlist.exists():
        counts = merge_wordlist(counts, args.wordlist, floor=max(args.min_freq, 1))
    features = DistinctiveFeatures(metric=args.metric)
    g2p = EnglishG2P(args.data_dir / "cmudict-0.7b")
    kb = KeyboardLayout.from_json(args.data_dir / "qwertyKeymap.json")
    checker = SpellChecker(counts, features=features, g2p=g2p, keyboard=kb, min_freq=args.min_freq)
    print(
        f"Indexed {len(checker.counts)} words (metric={args.metric}) in {time.perf_counter() - t0:.2f}s"
    )

    testsets = [
        ("norvig1", args.data_dir / "spell-testset-norvig1.txt", None),
        ("norvig2", args.data_dir / "spell-testset-norvig2.txt", None),
        ("aspell", args.data_dir / "spell-testset-aspell.txt", None),
        ("wikipedia", args.data_dir / "spell-testset-wikipedia.txt", 300),
        ("holbrook", args.data_dir / "spell-testset-holbrook.txt", 300),
    ]

    modes = ["norvig", "keyboard", "phonetic", "unified"]
    for name, path, limit in testsets:
        pairs = load_testset(path, max_items=limit)
        print(f"\n=== {name} (n={len(pairs)}) ===")
        print(f"{'mode':<10} {'top1 %':>8} {'top5 %':>8} {'ms/word':>9}")
        for mode in modes:
            res = evaluate(checker, pairs, mode=mode, weights=weights)
            print(
                f"{mode:<10} {res['top1']:>8.2f} {res['top5']:>8.2f} {res['ms_per_word']:>9.2f}"
            )


if __name__ == "__main__":
    main()
