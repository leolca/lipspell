"""Unit tests for phoneticspell."""

from collections import Counter
from pathlib import Path

import pytest

from phoneticspell.distance import keyboard_dl, orthographic_dl, phonetic_dl
from phoneticspell.features import DistinctiveFeatures
from phoneticspell.g2p import EnglishG2P
from phoneticspell.keyboard import KeyboardLayout
from phoneticspell.spellchecker import SpellChecker

DATA_DIR = Path("/home/leoca/ee/research/spellcheck/data")
FEATURES_CSV = DATA_DIR / "distinctivefeatures_ipa_mhulden.csv"


@pytest.fixture(scope="module")
def features() -> DistinctiveFeatures:
    return DistinctiveFeatures(FEATURES_CSV, metric="salience")


def test_phonetic_voicing_closer_than_unrelated(features: DistinctiveFeatures) -> None:
    p_id = features.phone_to_id["p"]
    b_id = features.phone_to_id["b"]
    s_id = features.phone_to_id["s"]
    assert features.sub_matrix[p_id, b_id] < features.sub_matrix[p_id, s_id]
    assert features.sub_matrix[p_id, p_id] == 0.0


def test_tokenize_affricate_and_diphthong(features: DistinctiveFeatures) -> None:
    ids = features.tokenize("tʃaɪld")
    phones = [features.phones[i] for i in ids]
    assert phones == ["tʃ", "a", "ɪ", "l", "d"]


def test_keyboard_adjacent_keys_closer_than_distant() -> None:
    kb = KeyboardLayout()
    assert kb.distance("a", "s") < kb.distance("a", "p")
    assert kb.distance("a", "a") == 0.0


def test_orthographic_and_keyboard_dl() -> None:
    kb = KeyboardLayout()
    assert orthographic_dl("teh", "the") == 1
    assert orthographic_dl("accomodate", "accommodate") == 1
    assert keyboard_dl("abouy", "about", kb) < keyboard_dl("aboux", "about", kb)


def test_phonetic_dl_nite_night(features: DistinctiveFeatures) -> None:
    g2p = EnglishG2P()
    p_nite = features.tokenize(g2p.transcribe("nite"))
    p_night = features.tokenize(g2p.transcribe("night"))
    p_neat = features.tokenize(g2p.transcribe("neat"))
    d_night = phonetic_dl(p_nite, p_night, features.sub_matrix, features.indel_costs)
    d_neat = phonetic_dl(p_nite, p_neat, features.sub_matrix, features.indel_costs)
    assert d_night < d_neat


def test_spellchecker_end_to_end(features: DistinctiveFeatures) -> None:
    counts = Counter(
        {
            "night": 100,
            "neat": 80,
            "phone": 90,
            "spelling": 70,
            "accommodate": 50,
            "stephen": 40,
        }
    )
    g2p = EnglishG2P()
    checker = SpellChecker(counts, features=features, g2p=g2p)
    assert checker.correct("speling", mode="unified") == "spelling"
    assert checker.correct("accomodate", mode="unified") == "accommodate"
    assert checker.correct("nite", mode="phonetic") == "night"
    assert checker.correct("foen", mode="phonetic") == "phone"
