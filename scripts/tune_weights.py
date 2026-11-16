"""Tune (w_freq, w_orth, w_kb, w_phon) weights on a held-out dev split of birkbeck."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np

from phoneticspell.features import DistinctiveFeatures
from phoneticspell.g2p import EnglishG2P
from phoneticspell.keyboard import KeyboardLayout
from phoneticspell.spellchecker import SpellChecker, load_corpus_counts


def load_pairs(path: Path) -> List[Tuple[str, str]]:
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
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/leoca/ee/research/spellcheck/data"),
    )
    parser.add_argument("--dev-size", type=int, default=800)
    args = parser.parse_args()

    counts = load_corpus_counts(args.data_dir / "big.txt")
    features = DistinctiveFeatures(
        args.data_dir / "distinctivefeatures_ipa_mhulden.csv",
        metric="salience",
    )
    g2p = EnglishG2P(args.data_dir / "cmudict-0.7b")
    kb = KeyboardLayout.from_json(args.data_dir / "qwertyKeymap.json")
    checker = SpellChecker(counts, features=features, g2p=g2p, keyboard=kb, min_freq=2)

    # Exclude typos present in evaluation testsets
    eval_typos: Set[str] = set()
    for name in (
        "spell-testset-norvig1.txt",
        "spell-testset-norvig2.txt",
        "spell-testset-aspell.txt",
        "spell-testset-wikipedia.txt",
        "spell-testset-holbrook.txt",
    ):
        for typo, _ in load_pairs(args.data_dir / name):
            eval_typos.add(typo)

    dev_pairs = [
        (typo, target)
        for typo, target in load_pairs(args.data_dir / "spell-testset-birkbeck.txt")
        if typo not in eval_typos and target in checker.counts and typo not in checker.counts
    ][: args.dev_size]
    print(f"Tuning on {len(dev_pairs)} birkbeck dev pairs (disjoint from eval sets)...")

    # Pre-extract candidate feature matrices once
    cached: List[Tuple[np.ndarray, int]] = []
    for typo, target in dev_pairs:
        feats = checker.extract_candidate_features(typo, mode="unified")
        cands = [c for c, _ in feats]
        if target not in cands:
            continue
        mat = np.array([f for _, f in feats], dtype=np.float32)
        target_idx = cands.index(target)
        cached.append((mat, target_idx))

    print(f"Dev pairs with target in candidate set: {len(cached)}/{len(dev_pairs)}")

    def score_weights(w: np.ndarray) -> float:
        hits = 0
        for mat, tidx in cached:
            if int(np.argmax(mat @ w)) == tidx:
                hits += 1
        return 100.0 * hits / len(cached)

    # Coarse grid + coordinate refinement (fix w_freq = 1.0 as scale reference)
    best_w = np.array([1.0, 4.0, 3.0, 2.0], dtype=np.float32)
    best_acc = score_weights(best_w)
    for w_orth in (2.0, 4.0, 6.0, 8.0, 10.0):
        for w_kb in (0.0, 1.0, 2.0, 4.0, 6.0):
            for w_phon in (0.0, 0.5, 1.0, 2.0, 4.0, 6.0):
                w = np.array([1.0, w_orth, w_kb, w_phon], dtype=np.float32)
                acc = score_weights(w)
                if acc > best_acc:
                    best_acc = acc
                    best_w = w

    # Coordinate refinement
    for step in (1.0, 0.5, 0.25):
        improved = True
        while improved:
            improved = False
            for dim in (1, 2, 3):
                for delta in (-step, step):
                    cand_w = best_w.copy()
                    cand_w[dim] = max(0.0, cand_w[dim] + delta)
                    acc = score_weights(cand_w)
                    if acc > best_acc:
                        best_acc = acc
                        best_w = cand_w
                        improved = True

    print(
        f"Best dev Top-1: {best_acc:.2f}% at weights = "
        f"({best_w[0]:.2f}, {best_w[1]:.2f}, {best_w[2]:.2f}, {best_w[3]:.2f})"
    )


if __name__ == "__main__":
    main()
