"""Experiments requested in Review.md: component ablation, candidate recall, and dev/eval overlap."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

from phoneticspell.features import DistinctiveFeatures
from phoneticspell.g2p import EnglishG2P, PortugueseG2P
from phoneticspell.keyboard import KeyboardLayout
from phoneticspell.spellchecker import SpellChecker, load_corpus_counts, merge_wordlist
from scripts.benchmark import evaluate, load_testset
from scripts.eval_portuguese import load_pt_counts, load_tsv_pairs
from scripts.eval_toefl_spell import load_toefl_pairs


def candidate_recall_breakdown(checker: SpellChecker, pairs: List[Tuple[str, str]]) -> Dict[str, float]:
    n = max(len(pairs), 1)
    sym_hits = 0
    skel_hits = 0
    both_hits = 0
    cand_sizes: List[int] = []
    for typo, target in pairs:
        c_orth = checker.index.query_orthographic(typo)
        c_phon = checker.index.query_phonetic(typo)
        c_all = c_orth | c_phon
        cand_sizes.append(len(c_all))
        if target in c_orth:
            sym_hits += 1
        if target in c_phon:
            skel_hits += 1
        if target in c_all:
            both_hits += 1
    return {
        "sym_rec": 100.0 * sym_hits / n,
        "skel_rec": 100.0 * skel_hits / n,
        "comb_rec": 100.0 * both_hits / n,
        "avg_cands": float(np.mean(cand_sizes)),
    }


def main() -> None:
    data_dir = Path("/home/leoca/ee/research/spellcheck/data")
    local_data = Path(__file__).resolve().parent.parent / "data"
    features = DistinctiveFeatures(metric="cosine")

    # 1. Verify Birkbeck dev vs Holbrook overlap
    eval_pairs_all: Set[Tuple[str, str]] = set()
    eval_typos_all: Set[str] = set()
    for fname in (
        "spell-testset-norvig1.txt",
        "spell-testset-norvig2.txt",
        "spell-testset-aspell.txt",
        "spell-testset-wikipedia.txt",
        "spell-testset-holbrook.txt",
    ):
        for typo, target in load_testset(data_dir / fname):
            eval_pairs_all.add((typo, target))
            eval_typos_all.add(typo)
    dev_pairs = [
        (typo, target)
        for typo, target in load_testset(data_dir / "spell-testset-birkbeck.txt")
        if typo not in eval_typos_all
    ][:800]
    overlap = set(dev_pairs) & eval_pairs_all
    print(f"Birkbeck dev size={len(dev_pairs)}, exact pair overlap with ALL eval sets={len(overlap)}")

    # 2. English Checker
    en_counts = merge_wordlist(
        load_corpus_counts(data_dir / "big.txt"),
        Path("/usr/share/dict/american-english"),
        floor=1,
    )
    en_checker = SpellChecker(
        en_counts,
        features=features,
        g2p=EnglishG2P(data_dir / "cmudict-0.7b"),
        keyboard=KeyboardLayout.from_json(data_dir / "qwertyKeymap.json"),
        min_freq=1,
    )

    # 3. Portuguese Checker (with palatal ɲ and ʎ)
    pt_counts = load_pt_counts(
        local_data / "enelvo/freq-cgu.txt",
        local_data / "enelvo/unitex-full-clean+enelvo-ja-corrigido.txt",
    )
    pt_checker = SpellChecker(
        pt_counts,
        features=features,
        g2p=PortugueseG2P(),
        keyboard=KeyboardLayout.abnt2(),
        min_freq=1,
    )

    datasets = [
        ("norvig1", en_checker, load_testset(data_dir / "spell-testset-norvig1.txt")),
        ("norvig2", en_checker, load_testset(data_dir / "spell-testset-norvig2.txt")),
        ("aspell", en_checker, load_testset(data_dir / "spell-testset-aspell.txt")),
        ("wikipedia", en_checker, load_testset(data_dir / "spell-testset-wikipedia.txt", max_items=300)),
        ("holbrook", en_checker, load_testset(data_dir / "spell-testset-holbrook.txt", max_items=300)),
        ("toefl", en_checker, load_toefl_pairs(local_data / "toefl_spell_annotations.tsv", max_items=1000)),
        ("pt_wiki", pt_checker, load_tsv_pairs(local_data / "pt_wikipedia_typos.tsv", max_items=500)),
    ]

    print("\n=== Candidate Generation Recall Breakdown ===")
    print(f"{'dataset':<12} {'SymDel%':>8} {'Skel%':>8} {'Comb%':>8} {'AvgCands':>9}")
    for name, chk, pairs in datasets:
        r = candidate_recall_breakdown(chk, pairs)
        print(
            f"{name:<12} {r['sym_rec']:>8.2f} {r['skel_rec']:>8.2f} {r['comb_rec']:>8.2f} {r['avg_cands']:>9.1f}"
        )

    # 4. Component Ablation (A..E)
    configs = [
        ("A_Orth+Freq", (1.0, 4.0, 0.0, 0.0)),
        ("B_Orth+Kb+Freq", (1.0, 4.0, 1.5, 0.0)),
        ("C_Orth+Phon+Freq", (1.0, 4.0, 0.0, 4.0)),
        ("D_Orth+Kb+Phon", (0.0, 4.0, 1.5, 4.0)),
        ("E_Full_Unified", (1.0, 4.0, 1.5, 4.0)),
    ]
    ablation_sets = [d for d in datasets if d[0] in ("norvig1", "aspell", "holbrook", "toefl", "pt_wiki")]
    print("\n=== Component Ablation (Top-1 % / Top-5 %) ===")
    for name, chk, pairs in ablation_sets:
        print(f"\n-- {name} (n={len(pairs)}) --")
        for cfg_name, w in configs:
            res = evaluate(chk, pairs, mode="unified", weights=w)
            print(f"  {cfg_name:<18} T1={res['top1']:6.2f}%  T5={res['top5']:6.2f}%")


if __name__ == "__main__":
    main()
