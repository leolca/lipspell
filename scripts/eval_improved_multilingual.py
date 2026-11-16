"""Evaluate improved Russian, Hebrew, Turkish, and French pipelines."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Set

from phoneticspell.features import DistinctiveFeatures
from phoneticspell.g2p import FrenchG2P, HebrewG2P, RussianG2P, TurkishG2P
from phoneticspell.keyboard import KeyboardLayout
from phoneticspell.spellchecker import SpellChecker
from scripts.eval_portuguese import load_tsv_pairs
from scripts.eval_slavic_semitic_and_tuning import load_freq_words, load_ruspellru_pairs, run_lang
from scripts.eval_turkish import load_tr_counts, load_tr_wiki_sample


def load_dic_words(dic_path: Path, is_turkish: bool = False) -> Set[str]:
    words: Set[str] = set()
    with dic_path.open("r", encoding="utf-8", errors="ignore") as fh:
        next(fh, None)  # skip count line
        for line in fh:
            w = line.strip().split("/", 1)[0].split("\t", 1)[0].strip()
            if is_turkish:
                w = w.replace("İ", "i").replace("I", "ı")
            w = w.lower()
            if w.isalpha():
                words.add(w)
    return words


def merge_clean_lexicon(
    dic_path: Path,
    freq_path: Path,
    max_freq_words: int = 200_000,
    restrict_to_dic: bool = False,
    is_turkish: bool = False,
) -> Counter[str]:
    dic_words = load_dic_words(dic_path, is_turkish=is_turkish)
    counts: Counter[str] = Counter({w: 1 for w in dic_words})
    freq_counts = (
        load_tr_counts(freq_path, max_words=max_freq_words)
        if is_turkish
        else load_freq_words(freq_path, max_words=max_freq_words)
    )
    for w, c in freq_counts.items():
        if restrict_to_dic and w not in dic_words:
            continue
        counts[w] += c
    return counts


def main() -> None:
    data_root = Path(__file__).resolve().parent.parent / "data"
    features = DistinctiveFeatures(metric="cosine")

    # 1. Russian (Slavic/Cyrillic) with final devoicing
    ru_counts = merge_clean_lexicon(data_root / "ru/index.dic", data_root / "ru/ru_full.txt", 250_000)
    ru_checker = SpellChecker(
        ru_counts, features=features, g2p=RussianG2P(), keyboard=KeyboardLayout.russian_jcuken()
    )
    ru_pairs = load_ruspellru_pairs(data_root / "ru/sources.txt", data_root / "ru/corrections.txt")
    run_lang("Russian (Improved G2P + dic+250k)", ru_checker, ru_pairs)

    # 2. Hebrew (Semitic/Abjad) with clean dic + matres lectionis indel overrides
    he_counts = merge_clean_lexicon(
        data_root / "he/index.dic", data_root / "he/he_full.txt", 250_000, restrict_to_dic=True
    )
    he_checker = SpellChecker(
        he_counts,
        features=features,
        g2p=HebrewG2P(),
        keyboard=KeyboardLayout.hebrew(),
        char_indel_overrides={"א": 0.25, "ו": 0.25, "י": 0.25},
    )
    he_pairs = load_tsv_pairs(data_root / "he/he_wikipedia_typos.tsv")
    run_lang("Hebrew (Clean dic + matres lectionis)", he_checker, he_pairs)

    # 3. Turkish (Turkic) with dic + 500k + diacritic tolerance
    tr_counts = merge_clean_lexicon(
        data_root / "tr/index.dic", data_root / "tr/tr_full.txt", 500_000, is_turkish=True
    )
    tr_checker = SpellChecker(
        tr_counts, features=features, g2p=TurkishG2P(), keyboard=KeyboardLayout.turkish_q()
    )
    tr_pairs = load_tr_wiki_sample(data_root / "tr/SpellingMistakes.tsv", sample_size=1000)
    run_lang("Turkish (dic+500k + diacritic tolerance)", tr_checker, tr_pairs)

    # 4. French (Opaque Romance) with clean dic + nasal vowels + diacritic tolerance
    fr_counts = merge_clean_lexicon(
        data_root / "fr/index.dic", data_root / "fr/fr_full.txt", 250_000, restrict_to_dic=True
    )
    fr_checker = SpellChecker(
        fr_counts, features=features, g2p=FrenchG2P(), keyboard=KeyboardLayout.azerty()
    )
    fr_pairs = load_tsv_pairs(data_root / "fr/fr_wikipedia_typos.tsv")
    run_lang("French (Clean dic + nasals + diacritic)", fr_checker, fr_pairs)


if __name__ == "__main__":
    main()
