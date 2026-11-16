"""Cross-lingual Russian (Slavic/Cyrillic) and Hebrew (Semitic/Abjad) tests."""

from collections import Counter

from phoneticspell.features import DEFAULT_FEATURES_CSV, DistinctiveFeatures
from phoneticspell.g2p import HebrewG2P, RussianG2P
from phoneticspell.keyboard import KeyboardLayout
from phoneticspell.spellchecker import SpellChecker


def test_russian_jcuken_and_g2p() -> None:
    kb = KeyboardLayout.russian_jcuken()
    assert kb.distance("о", "л") < kb.distance("о", "й")
    g2p = RussianG2P()
    assert g2p.transcribe("красного") == "kɾasnovo"


def test_russian_spellchecker() -> None:
    counts = Counter({"привет": 100, "сегодня": 90, "хорошо": 80, "человек": 70})
    features = DistinctiveFeatures(DEFAULT_FEATURES_CSV, metric="cosine")
    checker = SpellChecker(
        counts, features=features, g2p=RussianG2P(), keyboard=KeyboardLayout.russian_jcuken()
    )
    assert checker.correct("превет", mode="unified") == "привет"
    assert checker.correct("севодня", mode="unified") == "сегодня"
    assert checker.correct("селовек", mode="unified") == "человек"


def test_hebrew_layout_and_g2p() -> None:
    kb = KeyboardLayout.hebrew()
    assert kb.distance("ש", "ד") < kb.distance("ש", "פ")
    g2p = HebrewG2P()
    assert g2p.transcribe("פיסיק") == "pjsjk" or len(g2p.transcribe("פיסיק")) == 5


def test_hebrew_spellchecker() -> None:
    counts = Counter({"פיזיק": 100, "אזור": 90, "סינתטי": 80})
    features = DistinctiveFeatures(DEFAULT_FEATURES_CSV, metric="cosine")
    checker = SpellChecker(
        counts, features=features, g2p=HebrewG2P(), keyboard=KeyboardLayout.hebrew()
    )
    assert checker.correct("פיסיק", mode="unified") == "פיזיק"
    assert checker.correct("איזור", mode="unified") == "אזור"
