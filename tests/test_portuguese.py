"""Cross-lingual Portuguese tests reusing the universal IPA distinctive-feature table."""

from collections import Counter
from pathlib import Path

import pytest

from phoneticspell.features import DEFAULT_FEATURES_CSV, DistinctiveFeatures
from phoneticspell.g2p import PortugueseG2P
from phoneticspell.keyboard import KeyboardLayout
from phoneticspell.spellchecker import SpellChecker

FEATURES_CSV = DEFAULT_FEATURES_CSV


@pytest.fixture(scope="module")
def pt_checker() -> SpellChecker:
    counts = Counter(
        {
            "casa": 100,
            "exceção": 80,
            "chave": 90,
            "jeito": 85,
            "coração": 95,
            "mulher": 70,
            "viagem": 60,
            "açúcar": 50,
            "pesquisa": 75,
        }
    )
    features = DistinctiveFeatures(FEATURES_CSV, metric="cosine")
    g2p = PortugueseG2P()
    kb = KeyboardLayout.abnt2()
    return SpellChecker(counts, features=features, g2p=g2p, keyboard=kb)


def test_abnt2_has_cedilla_near_l() -> None:
    kb = KeyboardLayout.abnt2()
    assert kb.distance("ç", "l") < kb.distance("ç", "q")


def test_portuguese_phonetic_corrections(pt_checker: SpellChecker) -> None:
    assert pt_checker.correct("caza", mode="unified") == "casa"
    assert pt_checker.correct("xave", mode="unified") == "chave"
    assert pt_checker.correct("geito", mode="unified") == "jeito"
    assert pt_checker.correct("corassão", mode="unified") == "coração"
    assert pt_checker.correct("escessão", mode="unified") == "exceção"
    assert pt_checker.correct("pesquiza", mode="unified") == "pesquisa"
