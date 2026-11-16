"""Compare phone-distance metrics (`l1`, `salience`, `cosine`) on phonetic & unified modes."""

from __future__ import annotations

import argparse
from pathlib import Path

from phoneticspell.features import DistinctiveFeatures
from phoneticspell.g2p import EnglishG2P
from phoneticspell.keyboard import KeyboardLayout
from phoneticspell.spellchecker import SpellChecker, load_corpus_counts, merge_wordlist
from scripts.benchmark import evaluate, load_testset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/leoca/ee/research/spellcheck/data"),
    )
    parser.add_argument(
        "--wordlist",
        type=Path,
        default=Path("/usr/share/dict/american-english"),
    )
    args = parser.parse_args()

    counts = load_corpus_counts(args.data_dir / "big.txt")
    if args.wordlist.exists():
        counts = merge_wordlist(counts, args.wordlist, floor=1)
    g2p = EnglishG2P(args.data_dir / "cmudict-0.7b")
    kb = KeyboardLayout.from_json(args.data_dir / "qwertyKeymap.json")
    weights = (1.0, 4.0, 1.5, 4.0)

    testsets = [
        ("norvig1", load_testset(args.data_dir / "spell-testset-norvig1.txt")),
        ("norvig2", load_testset(args.data_dir / "spell-testset-norvig2.txt")),
        ("aspell", load_testset(args.data_dir / "spell-testset-aspell.txt")),
        ("wikipedia", load_testset(args.data_dir / "spell-testset-wikipedia.txt", max_items=300)),
        ("holbrook", load_testset(args.data_dir / "spell-testset-holbrook.txt", max_items=300)),
    ]

    for metric in ("l1", "salience", "cosine"):
        features = DistinctiveFeatures(
            args.data_dir / "distinctivefeatures_ipa_mhulden.csv", metric=metric
        )
        checker = SpellChecker(counts, features=features, g2p=g2p, keyboard=kb, min_freq=1)
        print(f"\n=== metric={metric} ===")
        print(f"{'dataset':<12} {'phonetic_t1':>12} {'unified_t1':>12} {'unified_t5':>12}")
        for name, pairs in testsets:
            ph = evaluate(checker, pairs, mode="phonetic", weights=weights)
            un = evaluate(checker, pairs, mode="unified", weights=weights)
            print(f"{name:<12} {ph['top1']:>11.2f}% {un['top1']:>11.2f}% {un['top5']:>11.2f}%")


if __name__ == "__main__":
    main()
