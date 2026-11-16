"""Evaluate aspell, hunspell, norvig, phonetic, and unified on Portuguese Wikipedia typos."""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path
from typing import List, Tuple

from phoneticspell.features import DistinctiveFeatures
from phoneticspell.g2p import PortugueseG2P
from phoneticspell.keyboard import KeyboardLayout
from phoneticspell.spellchecker import SpellChecker
from scripts.benchmark import evaluate
from scripts.compare_system_spellers import eval_ispell_pipe


def load_pt_counts(freq_path: Path, lexicon_path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    with freq_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if "," not in line:
                continue
            word, cnt_s = line.rsplit(",", 1)
            word = word.strip().lower()
            if word.isalpha() and cnt_s.isdigit():
                counts[word] += int(cnt_s)
    with lexicon_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            w = line.strip().lower()
            if w.isalpha() and w not in counts:
                counts[w] = 1
    return counts


def load_tsv_pairs(path: Path, max_items: int | None = None) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            typo, target = parts[0].strip().lower(), parts[1].strip().lower()
            if typo.isalpha() and target.isalpha() and typo != target:
                pairs.append((typo, target))
                if max_items is not None and len(pairs) >= max_items:
                    break
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=Path("/home/leoca/ee/research/spellcheck/data/distinctivefeatures_ipa_mhulden.csv"),
    )
    parser.add_argument(
        "--pt-dir",
        type=Path,
        default=Path("/home/leoca/ee/research/phoneticspellchecker/data"),
    )
    parser.add_argument("--max-items", type=int, default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    counts = load_pt_counts(
        args.pt_dir / "enelvo/freq-cgu.txt",
        args.pt_dir / "enelvo/unitex-full-clean+enelvo-ja-corrigido.txt",
    )
    features = DistinctiveFeatures(args.features_csv, metric="cosine")
    g2p = PortugueseG2P()
    kb = KeyboardLayout.abnt2()
    checker = SpellChecker(counts, features=features, g2p=g2p, keyboard=kb, min_freq=1)
    print(f"Indexed {len(checker.counts)} Portuguese words in {time.perf_counter() - t0:.2f}s")

    pairs = load_tsv_pairs(args.pt_dir / "pt_wikipedia_typos.tsv", max_items=args.max_items)
    weights = (1.0, 4.0, 1.5, 4.0)

    print(f"\n=== PT-BR Wikipedia Typos (n={len(pairs)}) ===")
    print(f"{'system':<12} {'top1 %':>8} {'top5 %':>8} {'ms/word':>9}")
    for label, fn in (
        ("aspell", lambda p: eval_ispell_pipe(["aspell", "-a", "-l", "pt_BR"], p)),
        ("hunspell", lambda p: eval_ispell_pipe(["hunspell", "-a", "-d", "pt_BR"], p)),
        ("norvig", lambda p: evaluate(checker, p, mode="norvig", weights=weights)),
        ("phonetic", lambda p: evaluate(checker, p, mode="phonetic", weights=weights)),
        ("unified", lambda p: evaluate(checker, p, mode="unified", weights=weights)),
    ):
        res = fn(pairs)
        print(f"{label:<12} {res['top1']:>8.2f} {res['top5']:>8.2f} {res['ms_per_word']:>9.2f}")


if __name__ == "__main__":
    main()
