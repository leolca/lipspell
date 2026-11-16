"""Compare Norvig and Unified (cosine) against Linux `aspell` and `hunspell`."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

from phoneticspell.features import DistinctiveFeatures
from phoneticspell.g2p import EnglishG2P
from phoneticspell.keyboard import KeyboardLayout
from phoneticspell.spellchecker import SpellChecker, load_corpus_counts, merge_wordlist
from scripts.benchmark import evaluate, load_testset


def eval_ispell_pipe(cmd: List[str], pairs: List[Tuple[str, str]]) -> Dict[str, float]:
    """Run an ispell-compatible `-a` command in batch mode and compute Top-1 / Top-5."""
    payload = "\n".join(f"^{typo}" for typo, _ in pairs) + "\n"
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        input=payload,
        text=True,
        capture_output=True,
        check=True,
    )
    elapsed = time.perf_counter() - t0

    body_lines = [l for l in proc.stdout.splitlines() if not l.startswith("@")]
    body = "\n".join(body_lines).strip("\n")
    blocks = body.split("\n\n") if body else []

    suggestions: List[List[str]] = []
    for block in blocks:
        blines = [l for l in block.splitlines() if l.strip()]
        if len(blines) == 1 and blines[0].startswith("&"):
            _, rhs = blines[0].split(":", 1)
            cands = [c.strip().lower() for c in rhs.split(",") if c.strip()]
            suggestions.append(cands)
        else:
            suggestions.append([])

    top1 = 0
    top5 = 0
    for (typo, target), cands in zip(pairs, suggestions):
        if not cands:
            cands = [typo]
        if cands[0] == target:
            top1 += 1
        if target in cands[:5]:
            top5 += 1

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
    parser.add_argument(
        "--wordlist",
        type=Path,
        default=Path("/usr/share/dict/american-english"),
    )
    args = parser.parse_args()

    counts = load_corpus_counts(args.data_dir / "big.txt")
    if args.wordlist.exists():
        counts = merge_wordlist(counts, args.wordlist, floor=1)
    features = DistinctiveFeatures(metric="cosine")
    g2p = EnglishG2P(args.data_dir / "cmudict-0.7b")
    kb = KeyboardLayout.from_json(args.data_dir / "qwertyKeymap.json")
    checker = SpellChecker(counts, features=features, g2p=g2p, keyboard=kb, min_freq=1)
    weights = (1.0, 4.0, 1.5, 4.0)

    testsets = [
        ("norvig1", load_testset(args.data_dir / "spell-testset-norvig1.txt")),
        ("norvig2", load_testset(args.data_dir / "spell-testset-norvig2.txt")),
        ("aspell", load_testset(args.data_dir / "spell-testset-aspell.txt")),
        ("wikipedia", load_testset(args.data_dir / "spell-testset-wikipedia.txt", max_items=300)),
        ("holbrook", load_testset(args.data_dir / "spell-testset-holbrook.txt", max_items=300)),
    ]

    for name, pairs in testsets:
        print(f"\n=== {name} (n={len(pairs)}) ===")
        print(f"{'system':<12} {'top1 %':>8} {'top5 %':>8} {'ms/word':>9}")
        for label, fn in (
            ("aspell", lambda p: eval_ispell_pipe(["aspell", "-a", "-l", "en_US"], p)),
            ("hunspell", lambda p: eval_ispell_pipe(["hunspell", "-a", "-d", "en_US"], p)),
            ("norvig", lambda p: evaluate(checker, p, mode="norvig", weights=weights)),
            ("unified", lambda p: evaluate(checker, p, mode="unified", weights=weights)),
        ):
            res = fn(pairs)
            print(
                f"{label:<12} {res['top1']:>8.2f} {res['top5']:>8.2f} {res['ms_per_word']:>9.2f}"
            )


if __name__ == "__main__":
    main()
