"""Evaluate Norvig, Phonetic, and Unified on Turkish Wikipedia nonword typos."""

from __future__ import annotations

import argparse
import random
import time
from collections import Counter
from pathlib import Path
from typing import List, Tuple

from scipy.stats import binomtest

from phoneticspell.features import DistinctiveFeatures
from phoneticspell.g2p import TurkishG2P
from phoneticspell.keyboard import KeyboardLayout
from phoneticspell.spellchecker import SpellChecker
from scripts.benchmark import evaluate


def load_tr_counts(freq_path: Path, max_words: int = 250_000) -> Counter[str]:
    counts: Counter[str] = Counter()
    with freq_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            word, cnt_s = parts[0].replace("İ", "i").replace("I", "ı").lower(), parts[1]
            if word.isalpha() and cnt_s.isdigit():
                counts[word] += int(cnt_s)
                if len(counts) >= max_words:
                    break
    return counts


def load_tr_wiki_sample(
    tsv_path: Path, sample_size: int = 1000, seed: int = 42
) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    seen = set()
    with tsv_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 8:
                continue
            typo, target, err_cat, word_type = (
                cols[0].strip(),
                cols[1].strip(),
                cols[6].strip(),
                cols[7].strip(),
            )
            if (
                word_type == "nonword"
                and err_cat.startswith("noise:")
                and typo.isalpha()
                and target.isalpha()
            ):
                tl = typo.replace("İ", "i").replace("I", "ı").lower()
                cl = target.replace("İ", "i").replace("I", "ı").lower()
                if tl != cl and (tl, cl) not in seen:
                    seen.add((tl, cl))
                    pairs.append((tl, cl))
    rng = random.Random(seed)
    if len(pairs) > sample_size:
        pairs = rng.sample(pairs, sample_size)
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tr-dir",
        type=Path,
        default=Path("/home/leoca/ee/research/phoneticspellchecker/data/tr"),
    )
    parser.add_argument("--max-words", type=int, default=250_000)
    parser.add_argument("--sample-size", type=int, default=1000)
    args = parser.parse_args()

    t0 = time.perf_counter()
    counts = load_tr_counts(args.tr_dir / "tr_full.txt", max_words=args.max_words)
    features = DistinctiveFeatures(metric="cosine")
    g2p = TurkishG2P()
    kb = KeyboardLayout.turkish_q()
    checker = SpellChecker(counts, features=features, g2p=g2p, keyboard=kb, min_freq=1)
    print(f"Indexed {len(checker.counts)} Turkish words in {time.perf_counter() - t0:.2f}s")

    pairs = load_tr_wiki_sample(args.tr_dir / "SpellingMistakes.tsv", sample_size=args.sample_size)
    in_vocab = sum(1 for _, t in pairs if t in checker.counts)
    cand_rec = sum(
        1 for x, t in pairs if t in (checker.index.query_orthographic(x) | checker.index.query_phonetic(x))
    )
    print(
        f"Turkish Wikipedia sample (n={len(pairs)}): "
        f"Vocab={100*in_vocab/len(pairs):.1f}%, CandRec={100*cand_rec/len(pairs):.1f}%"
    )

    weights = (1.0, 4.0, 1.5, 4.0)
    print(f"{'mode':<12} {'top1 %':>8} {'top5 %':>8} {'ms/word':>9}")
    for mode in ("norvig", "keyboard", "phonetic", "unified"):
        res = evaluate(checker, pairs, mode=mode, weights=weights)
        print(f"{mode:<12} {res['top1']:>8.2f} {res['top5']:>8.2f} {res['ms_per_word']:>9.2f}")

    # McNemar unified vs norvig
    b01 = b10 = 0
    for x, t in pairs:
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
