"""Evaluate aspell, hunspell, norvig, and unified on TOEFL-Spell (L2 English errors)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

from phoneticspell.features import DistinctiveFeatures
from phoneticspell.g2p import EnglishG2P
from phoneticspell.keyboard import KeyboardLayout
from phoneticspell.spellchecker import SpellChecker, load_corpus_counts, merge_wordlist
from scripts.benchmark import evaluate
from scripts.compare_system_spellers import eval_ispell_pipe


def load_toefl_pairs(path: Path, max_items: int | None = None) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    seen = set()
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            typo = (row.get("Misspelling") or "").strip().lower()
            target = (row.get("Correction") or "").strip().lower()
            if (
                typo.isalpha()
                and target.isalpha()
                and typo != target
                and (typo, target) not in seen
            ):
                seen.add((typo, target))
                pairs.append((typo, target))
                if max_items is not None and len(pairs) >= max_items:
                    break
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/leoca/ee/research/spellcheck/data"),
    )
    parser.add_argument(
        "--toefl-path",
        type=Path,
        default=Path("/home/leoca/ee/research/phoneticspellchecker/data/toefl_spell_annotations.tsv"),
    )
    parser.add_argument(
        "--wordlist",
        type=Path,
        default=Path("/usr/share/dict/american-english"),
    )
    parser.add_argument("--max-items", type=int, default=1000)
    args = parser.parse_args()

    pairs = load_toefl_pairs(args.toefl_path, max_items=args.max_items)
    counts = load_corpus_counts(args.data_dir / "big.txt")
    if args.wordlist.exists():
        counts = merge_wordlist(counts, args.wordlist, floor=1)

    features = DistinctiveFeatures(
        args.data_dir / "distinctivefeatures_ipa_mhulden.csv", metric="cosine"
    )
    g2p = EnglishG2P(args.data_dir / "cmudict-0.7b")
    kb = KeyboardLayout.from_json(args.data_dir / "qwertyKeymap.json")
    checker = SpellChecker(counts, features=features, g2p=g2p, keyboard=kb, min_freq=1)
    weights = (1.0, 4.0, 1.5, 4.0)

    print(f"=== TOEFL-Spell (n={len(pairs)} unique L2 typos) ===")
    print(f"{'system':<12} {'top1 %':>8} {'top5 %':>8} {'ms/word':>9}")
    for label, fn in (
        ("aspell", lambda p: eval_ispell_pipe(["aspell", "-a", "-l", "en_US"], p)),
        ("hunspell", lambda p: eval_ispell_pipe(["hunspell", "-a", "-d", "en_US"], p)),
        ("norvig", lambda p: evaluate(checker, p, mode="norvig", weights=weights)),
        ("phonetic", lambda p: evaluate(checker, p, mode="phonetic", weights=weights)),
        ("unified", lambda p: evaluate(checker, p, mode="unified", weights=weights)),
    ):
        res = fn(pairs)
        print(f"{label:<12} {res['top1']:>8.2f} {res['top5']:>8.2f} {res['ms_per_word']:>9.2f}")


if __name__ == "__main__":
    main()
