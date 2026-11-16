"""Ablate Rule-only G2P vs. Pronunciation Dictionary + Rule fallback across all 6 languages."""

from __future__ import annotations

import gc
import unicodedata
from pathlib import Path
from typing import Dict, Tuple

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
from scripts.benchmark import evaluate
from scripts.compare_hunspell_all_langs import get_50_50_test
from scripts.eval_improved_multilingual import merge_clean_lexicon
from scripts.eval_portuguese import load_pt_counts, load_tsv_pairs
from scripts.eval_slavic_semitic_and_tuning import load_ruspellru_pairs
from scripts.eval_toefl_spell import load_toefl_pairs
from scripts.eval_turkish import load_tr_wiki_sample

_IPA_REMAP: Dict[str, str] = {
    "ɝ": "ɚ",
    "ɫ": "l",
    "r": "ɾ",
    "ʁ": "ɹ",
    "ʀ": "ɾ",
    "œ": "ø",
    "ɥ": "j",
    "ɣ": "ɡ",
    "ɦ": "h",
    "ɕ": "ʃ",
    "ʂ": "ʃ",
    "ʐ": "ʒ",
    "ʑ": "ʒ",
    "ɵ": "o",
    "ʉ": "u",
    "ʋ": "v",
    "ʙ": "b",
    "ʝ": "j",
    "q": "k",
    "ħ": "x",
    "ɢ": "ɡ",
    "ɬ": "l",
    "ɸ": "f",
    "ʕ": "ʔ",
    "β": "v",
    "χ": "x",
    "c": "k",
    "ɟ": "ɡ",
    "ɰ": "ɡ",
    "ʏ": "y",
}


def load_ipa_lexicon(path: str | Path, is_turkish: bool = False) -> Dict[str, str]:
    lex: Dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        w, raw = line.split("\t", 1)
        w = w.strip()
        if is_turkish:
            w = w.replace("İ", "i").replace("I", "ı")
        w = w.lower()
        if not w or " " in w:
            continue
        ipa_raw = raw.split(",")[0].strip().strip("/")
        chars = []
        for ch in unicodedata.normalize("NFD", ipa_raw):
            if unicodedata.category(ch).startswith("M") or ch in (
                "ˈ", "ˌ", ".", "ː", "ˑ", "ʲ", "ˠ", "ʰ", "ʷ", "ˀ", "ˤ", "ʼ",
                " ", "‿", "⁽", "⁾", "˦", "˧", "˨", "˕",
            ):
                continue
            chars.append(_IPA_REMAP.get(ch, ch))
        ipa = "".join(chars)
        if ipa and w not in lex:
            lex[w] = ipa
    return lex


class DictFallbackG2P:
    def __init__(self, lex: Dict[str, str], fallback: object, is_turkish: bool = False) -> None:
        self.lex = lex
        self.fallback = fallback
        self.is_turkish = is_turkish

    def transcribe(self, word: str) -> str:
        w = word.replace("İ", "i").replace("I", "ı").lower() if self.is_turkish else word.lower()
        hit = self.lex.get(w)
        if hit is not None:
            return hit
        return self.fallback.transcribe(word)  # type: ignore[attr-defined]


def run_case(
    name: str,
    counts: Dict[str, int],
    pairs: list[Tuple[str, str]],
    kb: KeyboardLayout,
    rule_g2p: object,
    dict_g2p: object,
    tuned_w: Tuple[float, float, float, float],
    features: DistinctiveFeatures,
    indel: Dict[str, float] | None = None,
) -> None:
    test_pairs = get_50_50_test(counts, pairs)
    fixed_w = (1.0, 4.0, 1.5, 4.0)

    for mode_label, g2p in [("Rule-only", rule_g2p), ("Dict+Rule", dict_g2p)]:
        sc = SpellChecker(
            word_counts=counts,
            features=features,
            g2p=g2p,  # type: ignore[arg-type]
            keyboard=kb,
            char_indel_overrides=indel,
        )
        r_fix = evaluate(sc, test_pairs, mode="unified", weights=fixed_w)
        r_tun = evaluate(sc, test_pairs, mode="unified", weights=tuned_w)
        print(
            f"{name:<12} {len(test_pairs):>4} {mode_label:<10} | "
            f"Fixed: {r_fix['top1']:6.2f}% / {r_fix['top5']:6.2f}% | "
            f"Tuned: {r_tun['top1']:6.2f}% / {r_tun['top5']:6.2f}%",
            flush=True,
        )
        del sc
        gc.collect()


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    data_root = root / "data"
    en_data = Path("/home/leoca/ee/research/spellcheck/data")
    ipa_dir = Path("/tmp/ipa_dict")
    features = DistinctiveFeatures(metric="cosine")

    print("=== G2P Ablation: Rule-only vs. Pronunciation Dict + Rule fallback ===")

    # 1. English (Rule-only vs CMUdict vs ipa-dict en_US)
    en_counts = merge_wordlist(load_corpus_counts(en_data / "big.txt"), Path("/usr/share/dict/american-english"), 1)
    en_pairs = load_toefl_pairs(data_root / "toefl_spell_annotations.tsv", 1000)
    en_rule = EnglishG2P(cmudict_path=None)
    en_cmu = EnglishG2P(cmudict_path=en_data / "cmudict-0.7b")
    run_case("English-CMU", en_counts, en_pairs, KeyboardLayout(), en_rule, en_cmu, (1.0, 4.0, 2.0, 2.0), features)
    en_ipa_lex = load_ipa_lexicon(ipa_dir / "en_US.txt")
    run_case("English-IPA", en_counts, en_pairs, KeyboardLayout(), en_rule, DictFallbackG2P(en_ipa_lex, en_rule), (1.0, 4.0, 2.0, 2.0), features)
    del en_counts, en_ipa_lex
    gc.collect()

    # 2. Russian
    ru_counts = merge_clean_lexicon(data_root / "ru/index.dic", data_root / "ru/ru_full.txt", 250_000)
    ru_pairs = load_ruspellru_pairs(data_root / "ru/sources.txt", data_root / "ru/corrections.txt")
    ru_lex = load_ipa_lexicon(ipa_dir / "ru_wikipron.tsv")
    run_case("Russian", ru_counts, ru_pairs, KeyboardLayout.russian_jcuken(), RussianG2P(), DictFallbackG2P(ru_lex, RussianG2P()), (1.0, 2.0, 4.0, 1.0), features)
    del ru_counts, ru_lex
    gc.collect()

    # 3. Hebrew
    he_counts = merge_clean_lexicon(data_root / "he/index.dic", data_root / "he/he_full.txt", 250_000, restrict_to_dic=True)
    he_pairs = load_tsv_pairs(data_root / "he/he_wikipedia_typos.tsv")
    he_lex = load_ipa_lexicon(ipa_dir / "he_wikipron.tsv")
    run_case("Hebrew", he_counts, he_pairs, KeyboardLayout.hebrew(), HebrewG2P(), DictFallbackG2P(he_lex, HebrewG2P()), (1.0, 4.0, 4.0, 1.0), features, indel={"א": 0.25, "ו": 0.25, "י": 0.25})
    del he_counts, he_lex
    gc.collect()

    # 4. Portuguese
    pt_counts = load_pt_counts(data_root / "enelvo/freq-cgu.txt", data_root / "enelvo/unitex-full-clean+enelvo-ja-corrigido.txt")
    pt_pairs = load_tsv_pairs(data_root / "pt_wikipedia_typos.tsv")
    pt_lex = load_ipa_lexicon(ipa_dir / "pt_BR.txt")
    run_case("Portuguese", pt_counts, pt_pairs, KeyboardLayout.abnt2(), PortugueseG2P(), DictFallbackG2P(pt_lex, PortugueseG2P()), (1.0, 4.0, 1.0, 6.0), features)
    del pt_counts, pt_lex
    gc.collect()

    # 5. Turkish
    tr_counts = merge_clean_lexicon(data_root / "tr/index.dic", data_root / "tr/tr_full.txt", 500_000, is_turkish=True)
    tr_pairs = load_tr_wiki_sample(data_root / "tr/SpellingMistakes.tsv", 1000)
    tr_lex = load_ipa_lexicon(ipa_dir / "tr_wikipron.tsv", is_turkish=True)
    run_case("Turkish", tr_counts, tr_pairs, KeyboardLayout.turkish_q(), TurkishG2P(), DictFallbackG2P(tr_lex, TurkishG2P(), is_turkish=True), (1.0, 8.0, 4.0, 1.0), features)
    del tr_counts, tr_lex
    gc.collect()

    # 6. French
    fr_counts = merge_clean_lexicon(data_root / "fr/index.dic", data_root / "fr/fr_full.txt", 250_000, restrict_to_dic=True)
    fr_pairs = load_tsv_pairs(data_root / "fr/fr_wikipedia_typos.tsv")
    fr_lex = load_ipa_lexicon(ipa_dir / "fr_FR.txt")
    run_case("French", fr_counts, fr_pairs, KeyboardLayout.azerty(), FrenchG2P(), DictFallbackG2P(fr_lex, FrenchG2P()), (1.0, 4.0, 4.0, 6.0), features)


if __name__ == "__main__":
    main()
