"""Evaluate Slavic (Russian) and Semitic (Hebrew) + fixed vs per-language weight tuning."""

from __future__ import annotations

import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.stats import binomtest

from phoneticspell.features import DistinctiveFeatures
from phoneticspell.g2p import HebrewG2P, PortugueseG2P, RussianG2P, TurkishG2P
from phoneticspell.keyboard import KeyboardLayout
from phoneticspell.spellchecker import SpellChecker
from scripts.benchmark import evaluate
from scripts.eval_portuguese import load_pt_counts, load_tsv_pairs
from scripts.eval_turkish import load_tr_counts, load_tr_wiki_sample


def load_freq_words(path: Path, max_words: int = 200_000) -> Counter[str]:
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            w, cnt_s = parts[0].lower(), parts[1]
            if w.isalpha() and cnt_s.isdigit():
                counts[w] += int(cnt_s)
                if len(counts) >= max_words:
                    break
    return counts


def load_ruspellru_pairs(src_path: Path, tgt_path: Path) -> List[Tuple[str, str]]:
    tok_re = re.compile(r"[а-яА-ЯёЁ]+")
    pairs: List[Tuple[str, str]] = []
    seen = set()
    with src_path.open("r", encoding="utf-8") as fs, tgt_path.open("r", encoding="utf-8") as ft:
        for ls, lt in zip(fs, ft):
            ts = [w.lower() for w in tok_re.findall(ls)]
            tt = [w.lower() for w in tok_re.findall(lt)]
            if len(ts) != len(tt):
                continue
            for ws, wt in zip(ts, tt):
                if ws != wt and (ws, wt) not in seen:
                    seen.add((ws, wt))
                    pairs.append((ws, wt))
    return pairs


def tune_on_dev(
    checker: SpellChecker, dev_pairs: List[Tuple[str, str]]
) -> Tuple[float, float, float, float]:
    cached = []
    for x, t in dev_pairs:
        feats = checker.extract_candidate_features(x, mode="unified")
        cands = [c for c, _ in feats]
        if t not in cands:
            continue
        cached.append((np.array([f for _, f in feats], dtype=np.float32), cands.index(t)))
    if not cached:
        return (1.0, 4.0, 1.5, 4.0)

    def acc(w: np.ndarray) -> float:
        return sum(1 for mat, tidx in cached if int(np.argmax(mat @ w)) == tidx) / len(cached)

    best_w = np.array([1.0, 4.0, 1.5, 4.0], dtype=np.float32)
    best_a = acc(best_w)
    for w_o in (2.0, 4.0, 6.0, 8.0):
        for w_k in (0.0, 1.0, 2.0, 4.0):
            for w_p in (0.0, 1.0, 2.0, 4.0, 6.0):
                w = np.array([1.0, w_o, w_k, w_p], dtype=np.float32)
                a = acc(w)
                if a > best_a:
                    best_a = a
                    best_w = w
    return tuple(float(v) for v in best_w)  # type: ignore[return-value]


def run_lang(
    name: str,
    checker: SpellChecker,
    pairs: List[Tuple[str, str]],
    seed: int = 42,
) -> None:
    nonword = [(x, t) for x, t in pairs if x not in checker.counts and t in checker.counts]
    rng = random.Random(seed)
    rng.shuffle(nonword)
    mid = len(nonword) // 2
    dev, test = nonword[:mid], nonword[mid:]

    fixed_w = (1.0, 4.0, 1.5, 4.0)
    tuned_w = tune_on_dev(checker, dev)

    r_norv = evaluate(checker, test, mode="norvig", weights=fixed_w)
    r_fix = evaluate(checker, test, mode="unified", weights=fixed_w)
    r_tun = evaluate(checker, test, mode="unified", weights=tuned_w)

    b01 = b10 = 0
    for x, t in test:
        u = checker.correct(x, mode="unified", weights=fixed_w) == t
        n = checker.correct(x, mode="norvig", weights=fixed_w) == t
        if u and not n:
            b01 += 1
        elif n and not u:
            b10 += 1
    p = float(binomtest(min(b01, b10), n=b01 + b10, p=0.5, alternative="two-sided").pvalue) if (b01 + b10) else 1.0

    print(
        f"\n=== {name} (clean nonword in vocab={len(nonword)}, test={len(test)}) ===\n"
        f"  Norvig          T1={r_norv['top1']:5.2f}%  T5={r_norv['top5']:5.2f}%\n"
        f"  Unified (fixed) T1={r_fix['top1']:5.2f}%  T5={r_fix['top5']:5.2f}%  (vs Norv p={p:.4g})\n"
        f"  Unified (tuned) T1={r_tun['top1']:5.2f}%  T5={r_tun['top5']:5.2f}%  weights={tuple(round(v, 2) for v in tuned_w)}"
    )


def main() -> None:
    data_root = Path("/home/leoca/ee/research/phoneticspellchecker/data")
    features = DistinctiveFeatures(metric="cosine")

    # 1. Russian (Slavic, Cyrillic)
    t0 = time.perf_counter()
    ru_counts = load_freq_words(data_root / "ru/ru_full.txt", max_words=200_000)
    ru_checker = SpellChecker(
        ru_counts, features=features, g2p=RussianG2P(), keyboard=KeyboardLayout.russian_jcuken()
    )
    print(f"Indexed {len(ru_checker.counts)} Russian words in {time.perf_counter() - t0:.2f}s")
    ru_pairs = load_ruspellru_pairs(data_root / "ru/sources.txt", data_root / "ru/corrections.txt")
    run_lang("Russian (RUSpellRU, Slavic/Cyrillic)", ru_checker, ru_pairs)

    # 2. Hebrew (Semitic, Abjad)
    t0 = time.perf_counter()
    he_counts = load_freq_words(data_root / "he/he_full.txt", max_words=200_000)
    he_checker = SpellChecker(
        he_counts, features=features, g2p=HebrewG2P(), keyboard=KeyboardLayout.hebrew()
    )
    print(f"\nIndexed {len(he_checker.counts)} Hebrew words in {time.perf_counter() - t0:.2f}s")
    he_pairs = load_tsv_pairs(data_root / "he/he_wikipedia_typos.tsv")
    run_lang("Hebrew (Wikipedia, Semitic/Abjad)", he_checker, he_pairs)

    # 3. Portuguese & Turkish fixed vs per-language tuned comparison
    pt_counts = load_pt_counts(
        data_root / "enelvo/freq-cgu.txt",
        data_root / "enelvo/unitex-full-clean+enelvo-ja-corrigido.txt",
    )
    pt_checker = SpellChecker(
        pt_counts, features=features, g2p=PortugueseG2P(), keyboard=KeyboardLayout.abnt2()
    )
    pt_pairs = load_tsv_pairs(data_root / "pt_wikipedia_typos.tsv")
    run_lang("Portuguese (PT-Wiki, Romance)", pt_checker, pt_pairs)

    tr_counts = load_tr_counts(data_root / "tr/tr_full.txt", max_words=250_000)
    tr_checker = SpellChecker(
        tr_counts, features=features, g2p=TurkishG2P(), keyboard=KeyboardLayout.turkish_q()
    )
    tr_pairs = load_tr_wiki_sample(data_root / "tr/SpellingMistakes.tsv", sample_size=1000)
    run_lang("Turkish (TR-Wiki, Turkic)", tr_checker, tr_pairs)


if __name__ == "__main__":
    main()
