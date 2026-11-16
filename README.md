# Phonetic Spell Checker

A fast, cross-lingual spell checker combining:
1. **Distinctive-Feature Phonetic Edit Distance** (precomputed `|P| x |P|` phone substitution & indel tables from IPA articulatory features, supporting `cosine`, `l1`, and ALINE-style `salience` metrics)
2. **Keyboard Layout Distance** (precomputed `|K| x |K|` physical key distances for QWERTY and ABNT2)
3. **Corpus Word Frequency** (Norvig-style unigram log-probability prior)
4. **Two-Stage Indexing** (SymSpell-style symmetric delete index + coarse articulatory skeleton index)

## Structure
- `phoneticspell/features.py`: IPA distinctive-feature loader & precomputed phone distance matrices
- `phoneticspell/keyboard.py`: QWERTY / ABNT2 key-distance matrices
- `phoneticspell/g2p.py`: `EnglishG2P` (CMUdict + rule-based fallback) and `PortugueseG2P`
- `phoneticspell/distance.py`: Weighted Damerau-Levenshtein DP (`orthographic_dl`, `keyboard_dl`, `phonetic_dl`)
- `phoneticspell/candidates.py`: Symmetric-delete + phonetic-skeleton candidate indexing
- `phoneticspell/spellchecker.py`: Unified `SpellChecker`
- `scripts/benchmark.py`, `scripts/tune_weights.py`, `scripts/ablate_metrics.py`, `scripts/compare_system_spellers.py`

## Quick Start
```bash
pytest -v
python3 scripts/benchmark.py --wordlist /usr/share/dict/american-english --min-freq 1
python3 scripts/compare_system_spellers.py
```
