"""Statistical significance, coverage/recall, sparsity, and error-type analyses for the paper."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest

from phoneticspell.distance import orthographic_dl
from phoneticspell.features import DistinctiveFeatures
from phoneticspell.g2p import EnglishG2P, PortugueseG2P
from phoneticspell.keyboard import KeyboardLayout
from phoneticspell.spellchecker import SpellChecker, load_corpus_counts, merge_wordlist
from scripts.benchmark import load_testset
from scripts.eval_portuguese import load_pt_counts, load_tsv_pairs
from scripts.eval_toefl_spell import load_toefl_pairs


def ispell_top1_flags(cmd: List[str], pairs: List[Tuple[str, str]]) -> List[bool]:
    payload = "\n".join(f"^{typo}" for typo, _ in pairs) + "\n"
    proc = subprocess.run(cmd, input=payload, text=True, capture_output=True, check=True)
    cands_list: List[List[str]] = []
    for line in proc.stdout.splitlines():
        if not line or line.startswith("@"):
            continue
        if line.startswith(("*", "+", "-", "#")):
            cands_list.append([])
        elif line.startswith("&"):
            _, rhs = line.split(":", 1)
            cands_list.append([c.strip().lower() for c in rhs.split(",") if c.strip()])
    flags: List[bool] = []
    for (typo, target), cands in zip(pairs, cands_list):
        top = cands[0] if cands else typo
        flags.append(top == target)
    return flags


def ispell_vocab_coverage(cmd: List[str], pairs: List[Tuple[str, str]]) -> float:
    payload = "\n".join(f"^{target}" for _, target in pairs) + "\n"
    proc = subprocess.run(cmd, input=payload, text=True, capture_output=True, check=True)
    in_voc = 0
    for line in proc.stdout.splitlines():
        if not line or line.startswith("@"):
            continue
        if line.startswith(("*", "+", "-")):
            in_voc += 1
    return 100.0 * in_voc / max(len(pairs), 1)


def get_rss_mb() -> float:
    with open("/proc/self/status", "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return 0.0


def mcnemar_p(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Exact two-sided McNemar p-value (binomial test on discordant pairs)."""
    b01 = sum(1 for x, y in zip(a, b) if x and not y)
    b10 = sum(1 for x, y in zip(a, b) if not x and y)
    n = b01 + b10
    if n == 0:
        return 1.0
    return float(binomtest(min(b01, b10), n=n, p=0.5, alternative="two-sided").pvalue)


def analyze_dataset(
    name: str,
    checker: SpellChecker,
    pairs: List[Tuple[str, str]],
    aspell_cmd: List[str],
    hunspell_cmd: List[str],
    weights: Tuple[float, float, float, float] = (1.0, 4.0, 1.5, 4.0),
) -> None:
    in_vocab = 0
    cand_recall = 0
    orth1_norvig = orth1_unif = orth1_n = 0
    orth2_norvig = orth2_unif = orth2_n = 0
    norvig_flags: List[bool] = []
    unif_flags: List[bool] = []

    for typo, target in pairs:
        if target in checker.counts:
            in_vocab += 1
        orth_cands = checker.index.query_orthographic(typo)
        phon_cands = checker.index.query_phonetic(typo)
        if target in (orth_cands | phon_cands):
            cand_recall += 1

        n_top = checker.correct(typo, mode="norvig", weights=weights) == target
        u_top = checker.correct(typo, mode="unified", weights=weights) == target
        norvig_flags.append(n_top)
        unif_flags.append(u_top)

        d = orthographic_dl(typo, target)
        if d <= 1:
            orth1_n += 1
            orth1_norvig += int(n_top)
            orth1_unif += int(u_top)
        else:
            orth2_n += 1
            orth2_norvig += int(n_top)
            orth2_unif += int(u_top)

    aspell_flags = ispell_top1_flags(aspell_cmd, pairs)
    hunspell_flags = ispell_top1_flags(hunspell_cmd, pairs)
    asp_voc = ispell_vocab_coverage(aspell_cmd, pairs)
    hun_voc = ispell_vocab_coverage(hunspell_cmd, pairs)

    n = len(pairs)
    u_acc = 100.0 * sum(unif_flags) / n
    u_ci = 196.0 * math.sqrt((u_acc / 100.0) * (1.0 - u_acc / 100.0) / n)
    p_norv = mcnemar_p(unif_flags, norvig_flags)
    p_asp = mcnemar_p(unif_flags, aspell_flags)
    p_hun = mcnemar_p(unif_flags, hunspell_flags)

    print(
        f"{name:<12} n={n:<4} | Voc(Ours/Asp/Hun)={100*in_vocab/n:4.1f}/{asp_voc:4.1f}/{hun_voc:4.1f}% CandRec={100*cand_recall/n:4.1f}% | "
        f"Unif={u_acc:5.2f}±{u_ci:4.2f}% Norv={100*sum(norvig_flags)/n:5.2f}% (p={p_norv:.4g}) | "
        f"Asp={100*sum(aspell_flags)/n:5.2f}% (p={p_asp:.4g}) | "
        f"Hun={100*sum(hunspell_flags)/n:5.2f}% (p={p_hun:.4g})"
    )
    print(
        f"   d_orth<=1 (n={orth1_n}): Norv={100*orth1_norvig/max(orth1_n,1):.2f}% Unif={100*orth1_unif/max(orth1_n,1):.2f}% | "
        f"d_orth>=2 (n={orth2_n}): Norv={100*orth2_norvig/max(orth2_n,1):.2f}% Unif={100*orth2_unif/max(orth2_n,1):.2f}%"
    )


def main() -> None:
    data_dir = Path("/home/leoca/ee/research/spellcheck/data")
    local_data = Path(__file__).resolve().parent.parent / "data"

    features = DistinctiveFeatures(metric="cosine")
    syll_idx = features.feature_names.index("syll")
    cons_nnz = [int(np.count_nonzero(v)) for v in features.vectors if v[syll_idx] <= 0]
    vow_nnz = [int(np.count_nonzero(v)) for v in features.vectors if v[syll_idx] > 0]
    print(
        f"=== Feature Sparsity (L0 active features out of {features.vectors.shape[1]}) ===\n"
        f"Consonants (n={len(cons_nnz)}): mean={np.mean(cons_nnz):.2f}, std={np.std(cons_nnz):.2f}\n"
        f"Vowels     (n={len(vow_nnz)}): mean={np.mean(vow_nnz):.2f}, std={np.std(vow_nnz):.2f}\n"
    )

    # English
    rss0 = get_rss_mb()
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
    en_rss = get_rss_mb() - rss0
    print(f"English Index RSS delta: {en_rss:.1f} MiB, delete keys: {len(en_checker.index.delete_index)}, skel keys: {len(en_checker.index.skel_index)}")
    en_sets = [
        ("norvig1", load_testset(data_dir / "spell-testset-norvig1.txt")),
        ("norvig2", load_testset(data_dir / "spell-testset-norvig2.txt")),
        ("aspell", load_testset(data_dir / "spell-testset-aspell.txt")),
        ("wikipedia", load_testset(data_dir / "spell-testset-wikipedia.txt", max_items=300)),
        ("holbrook", load_testset(data_dir / "spell-testset-holbrook.txt", max_items=300)),
        ("toefl", load_toefl_pairs(local_data / "toefl_spell_annotations.tsv", max_items=1000)),
    ]
    print("=== English Coverage, McNemar p-values, and d_orth Breakdown ===")
    for name, pairs in en_sets:
        analyze_dataset(
            name,
            en_checker,
            pairs,
            aspell_cmd=["aspell", "-a", "-l", "en_US"],
            hunspell_cmd=["hunspell", "-a", "-d", "en_US"],
        )

    # Portuguese
    rss_pt0 = get_rss_mb()
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
    pt_rss = get_rss_mb() - rss_pt0
    print(f"\nPortuguese Index RSS delta: {pt_rss:.1f} MiB, delete keys: {len(pt_checker.index.delete_index)}, skel keys: {len(pt_checker.index.skel_index)}")
    pt_pairs = load_tsv_pairs(local_data / "pt_wikipedia_typos.tsv", max_items=500)
    pt_pairs_full = load_tsv_pairs(local_data / "pt_wikipedia_typos.tsv", max_items=None)
    print("\n=== Portuguese Coverage, McNemar p-values, and d_orth Breakdown ===")
    analyze_dataset(
        "pt_wiki_500",
        pt_checker,
        pt_pairs,
        aspell_cmd=["aspell", "-a", "-l", "pt_BR", "--encoding=utf-8"],
        hunspell_cmd=["hunspell", "-a", "-d", "pt_BR", "-i", "utf-8"],
    )
    analyze_dataset(
        "pt_wiki_full",
        pt_checker,
        pt_pairs_full,
        aspell_cmd=["aspell", "-a", "-l", "pt_BR", "--encoding=utf-8"],
        hunspell_cmd=["hunspell", "-a", "-d", "pt_BR", "-i", "utf-8"],
    )


if __name__ == "__main__":
    main()
