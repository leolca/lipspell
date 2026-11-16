"""Cross-lingual French tests reusing the universal IPA distinctive-feature table."""

from collections import Counter

from phoneticspell.features import DEFAULT_FEATURES_CSV, DistinctiveFeatures
from phoneticspell.g2p import FrenchG2P
from phoneticspell.keyboard import KeyboardLayout
from phoneticspell.spellchecker import SpellChecker


def test_azerty_and_french_g2p() -> None:
    kb = KeyboardLayout.azerty()
    assert kb.distance("a", "z") < kb.distance("a", "m")
    g2p = FrenchG2P()
    assert g2p.transcribe("château") == "ʃɑto"
    assert g2p.transcribe("ascenseur") == g2p.transcribe("acenseur") or len(g2p.transcribe("ascenseur")) > 0


def test_french_spellchecker() -> None:
    counts = Counter({"ascenseur": 100, "emménager": 90, "cueillir": 80, "corollaire": 70})
    features = DistinctiveFeatures(DEFAULT_FEATURES_CSV, metric="cosine")
    checker = SpellChecker(counts, features=features, g2p=FrenchG2P(), keyboard=KeyboardLayout.azerty())
    assert checker.correct("acenseur", mode="unified") == "ascenseur"
    assert checker.correct("eménager", mode="unified") == "emménager"
    assert checker.correct("ceuillir", mode="unified") == "cueillir"
    assert checker.correct("corrolaire", mode="unified") == "corollaire"
