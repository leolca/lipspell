"""Compare Hunspell, Norvig, LipSpell (Fixed), and LipSpell (Tuned) across all 6 languages."""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

from phoneticspell.features import DistinctiveFeatures
from phoneticspell.g2p import (
    EnglishG2P,
    FrenchG2P,
    HebrewG2P,
    PortugueseG2P,
    RussianG2P,
    TurkishG2P,
)
from phoneticspell.keyboard import KeyboardLayout
from phoneticspell.spellchecker import SpellChecker, load_corpus_counts, merge_wordlist
from scripts.compare_system_spellers import eval_ispell_pipe
from scripts.eval_improved_multilingual import merge_clean_lexicon
from scripts.eval_portuguese import load_pt_counts, load_tsv_pairs
from scripts.eval_slavic_semitic_and_tuning import load_ruspellru_pairs
from scripts.eval_toefl_spell import load_toefl_pairs
from scripts.eval_turkish import load_tr_wiki_sample


def get_50_50_test(checker: SpellChecker, pairs: List[Tuple[str, str]], seed: int = 42) -> List[Tuple[str, str]]:
    nonword = [(x, t) for x, t in pairs if x not in checker.counts and t in checker.counts]
    rng = random.Random(seed)
    rng.shuffle(nonword)
    return nonword[len(nonword) // 2 :]


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    data_root = root / "data"
    en_data = Path("/home/leoca/ee/research/spellcheck/data")
    features = DistinctiveFeatures(metric="cosine")

    # 1. English (TOEFL-Spell 50/50)
    en_counts = merge_wordlist(load_corpus_counts(en_data / "big.txt"), Path("/usr/share/dict/american-english"), 1)
    en_chk = SpellChecker(en_counts, features=features, g2p=EnglishG2P(en_data / "cmudict-0.7b"), keyboard=KeyboardLayout.from_json(en_data / "qwertyKeymap.json"))
    en_test = get_50_50_test(en_chk, load_toefl_pairs(data_root / "toefl_spell_annotations.tsv", 1000))

    # 2. Russian
    ru_counts = merge_clean_lexicon(data_root / "ru/index.dic", data_root / "ru/ru_full.txt", 250_000)
    ru_chk = SpellChecker(ru_counts, features=features, g2p=RussianG2P(), keyboard=KeyboardLayout.russian_jcuken())
    ru_test = get_50_50_test(ru_chk, load_ruspellru_pairs(data_root / "ru/sources.txt", data_root / "ru/corrections.txt"))

    # 3. Hebrew
    he_counts = merge_clean_lexicon(data_root / "he/index.dic", data_root / "he/he_full.txt", 250_000, restrict_to_dic=True)
    he_chk = SpellChecker(he_counts, features=features, g2p=HebrewG2P(), keyboard=KeyboardLayout.hebrew(), char_indel_overrides={"א": 0.25, "ו": 0.25, "י": 0.25})
    he_test = get_50_50_test(he_chk, load_tsv_pairs(data_root / "he/he_wikipedia_typos.tsv"))

    # 4. Portuguese
    pt_counts = load_pt_counts(data_root / "enelvo/freq-cgu.txt", data_root / "enelvo/unitex-full-clean+enelvo-ja-corrigido.txt")
    pt_chk = SpellChecker(pt_counts, features=features, g2p=PortugueseG2P(), keyboard=KeyboardLayout.abnt2())
    pt_test = get_50_50_test(pt_chk, load_tsv_pairs(data_root / "pt_wikipedia_typos.tsv"))

    # 5. Turkish
    tr_counts = merge_clean_lexicon(data_root / "tr/index.dic", data_root / "tr/tr_full.txt", 500_000, is_turkish=True)
    tr_chk = SpellChecker(tr_counts, features=features, g2p=TurkishG2P(), keyboard=KeyboardLayout.turkish_q())
    tr_test = get_50_50_test(tr_chk, load_tr_wiki_sample(data_root / "tr/SpellingMistakes.tsv", 1000))

    # 6. French
    fr_counts = merge_clean_lexicon(data_root / "fr/index.dic", data_root / "fr/fr_full.txt", 250_000, restrict_to_dic=True)
    fr_chk = SpellChecker(fr_counts, features=features, g2p=FrenchG2P(), keyboard=KeyboardLayout.azerty())
    fr_test = get_50_50_test(fr_chk, load_tsv_pairs(data_root / "fr/fr_wikipedia_typos.tsv"))

    print("=== Hunspell Comparison on 50/50 Clean Non-Word Test Splits ===")
    print(f"{'Language':<12} {'n':>5} {'Hunspell T1/T5':>16}")
    for name, cmd, test_pairs in (
        ("English", ["hunspell", "-a", "-d", "en_US"], en_test),
        ("Russian", ["hunspell", "-a", "-d", str(data_root / "ru/index"), "-i", "utf-8"], ru_test),
        ("Hebrew", ["hunspell", "-a", "-d", str(data_root / "he/index"), "-i", "utf-8"], he_test),
        ("Portuguese", ["hunspell", "-a", "-d", "pt_BR", "-i", "utf-8"], pt_test),
        ("Turkish", ["hunspell", "-a", "-d", str(data_root / "tr/index"), "-i", "utf-8"], tr_test),
        ("French", ["hunspell", "-a", "-d", str(data_root / "fr/index"), "-i", "utf-8"], fr_test),
    ):
        res = eval_ispell_pipe(cmd, test_pairs)
        print(f"{name:<12} {len(test_pairs):>5} {res['top1']:>6.2f}% / {res['top5']:>5.2f}%")


if __name__ == "__main__":
    main()
