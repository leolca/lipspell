"""Compare GNU Aspell across all 6 languages on the exact 50/50 clean non-word splits."""

from __future__ import annotations

from pathlib import Path

from phoneticspell.spellchecker import load_corpus_counts, merge_wordlist
from scripts.compare_hunspell_all_langs import get_50_50_test
from scripts.compare_system_spellers import eval_ispell_pipe
from scripts.eval_improved_multilingual import merge_clean_lexicon
from scripts.eval_portuguese import load_pt_counts, load_tsv_pairs
from scripts.eval_slavic_semitic_and_tuning import load_ruspellru_pairs
from scripts.eval_toefl_spell import load_toefl_pairs
from scripts.eval_turkish import load_tr_wiki_sample


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    data_root = root / "data"
    en_data = Path("/home/leoca/ee/research/spellcheck/data")
    aspell_dir = "/tmp/aspell_dicts"

    cases = [
        (
            "English",
            ["aspell", "-a", "-l", "en_US", "--encoding=utf-8"],
            lambda: get_50_50_test(
                merge_wordlist(load_corpus_counts(en_data / "big.txt"), Path("/usr/share/dict/american-english"), 1),
                load_toefl_pairs(data_root / "toefl_spell_annotations.tsv", 1000),
            ),
        ),
        (
            "Russian",
            ["aspell", "-a", f"--data-dir={aspell_dir}", f"--dict-dir={aspell_dir}", "-l", "ru", "--encoding=utf-8"],
            lambda: get_50_50_test(
                merge_clean_lexicon(data_root / "ru/index.dic", data_root / "ru/ru_full.txt", 250_000),
                load_ruspellru_pairs(data_root / "ru/sources.txt", data_root / "ru/corrections.txt"),
            ),
        ),
        (
            "Hebrew",
            ["aspell", "-a", f"--data-dir={aspell_dir}", f"--dict-dir={aspell_dir}", "-l", "he", "--encoding=utf-8"],
            lambda: get_50_50_test(
                merge_clean_lexicon(data_root / "he/index.dic", data_root / "he/he_full.txt", 250_000, restrict_to_dic=True),
                load_tsv_pairs(data_root / "he/he_wikipedia_typos.tsv"),
            ),
        ),
        (
            "Portuguese",
            ["aspell", "-a", "-l", "pt_BR", "--encoding=utf-8"],
            lambda: get_50_50_test(
                load_pt_counts(data_root / "enelvo/freq-cgu.txt", data_root / "enelvo/unitex-full-clean+enelvo-ja-corrigido.txt"),
                load_tsv_pairs(data_root / "pt_wikipedia_typos.tsv"),
            ),
        ),
        (
            "Turkish",
            ["aspell", "-a", f"--data-dir={aspell_dir}", f"--dict-dir={aspell_dir}", "-l", "tr", "--encoding=utf-8"],
            lambda: get_50_50_test(
                merge_clean_lexicon(data_root / "tr/index.dic", data_root / "tr/tr_full.txt", 500_000, is_turkish=True),
                load_tr_wiki_sample(data_root / "tr/SpellingMistakes.tsv", 1000),
            ),
        ),
        (
            "French",
            ["aspell", "-a", f"--data-dir={aspell_dir}", f"--dict-dir={aspell_dir}", "-l", "fr", "--encoding=utf-8"],
            lambda: get_50_50_test(
                merge_clean_lexicon(data_root / "fr/index.dic", data_root / "fr/fr_full.txt", 250_000, restrict_to_dic=True),
                load_tsv_pairs(data_root / "fr/fr_wikipedia_typos.tsv"),
            ),
        ),
    ]

    print("=== GNU Aspell Comparison on 50/50 Clean Non-Word Test Splits ===")
    print(f"{'Language':<12} {'n':>5} {'Aspell T1/T5':>16}")
    for name, cmd, make_pairs in cases:
        test_pairs = make_pairs()
        res = eval_ispell_pipe(cmd, test_pairs)
        print(f"{name:<12} {len(test_pairs):>5} {res['top1']:>6.2f}% / {res['top5']:>5.2f}%", flush=True)


if __name__ == "__main__":
    main()
