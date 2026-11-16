"""Cross-lingual Turkish tests reusing the universal IPA distinctive-feature table."""

from collections import Counter

from phoneticspell.features import DEFAULT_FEATURES_CSV, DistinctiveFeatures
from phoneticspell.g2p import TurkishG2P
from phoneticspell.keyboard import KeyboardLayout
from phoneticspell.spellchecker import SpellChecker


def test_turkish_q_and_g2p() -> None:
    kb = KeyboardLayout.turkish_q()
    assert kb.distance("ğ", "ü") < kb.distance("ğ", "z")
    g2p = TurkishG2P()
    assert g2p.transcribe("çiçek") == "tʃitʃek"
    assert g2p.transcribe("güneş") == "ɡyneʃ"


def test_turkish_spellchecker() -> None:
    counts = Counter({"burada": 100, "olarak": 90, "tarafından": 80, "güneş": 70})
    features = DistinctiveFeatures(DEFAULT_FEATURES_CSV, metric="cosine")
    checker = SpellChecker(counts, features=features, g2p=TurkishG2P(), keyboard=KeyboardLayout.turkish_q())
    assert checker.correct("bırada", mode="unified") == "burada"
    assert checker.correct("olark", mode="unified") == "olarak"
    assert checker.correct("tarfından", mode="unified") == "tarafından"
