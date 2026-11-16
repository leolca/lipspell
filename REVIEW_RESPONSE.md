# Peer Review Response & Revision Log

This document tracks how each point raised in the peer reviews (`/tmp/paper_review.pdf` and `/tmp/Review.md`) was addressed in the codebase, experiments, and both manuscripts (`paper/main.tex` and `paper/linguamatica/artigo.tex`).

## 1. Essential Corrections (Completed)
1. **Portuguese `nh` and `lh` IPA transcriptions (`nh -> /ɲ/`, `lh -> /ʎ/`):** Added `/ɲ/` and `/ʎ/` to `phoneticspell/data/distinctivefeatures_ipa.csv` and updated `PortugueseG2P` in `phoneticspell/g2p.py`.
2. **Standardized IPA notation:** Replaced mixed ASCII/IPA examples with consistent IPA via `\usepackage[safe]{tipa}` in both papers.
3. **Missing Hulden IPA feature-table & baseline references:** Added citations for Hulden (2012), Norvig (2007), Garbe/SymSpell (2012), Weide/CMUdict (1998), Church & Gale (1991), and Chomsky & Halle (1968) in both `refs.bib` files.
4. **Aspell `pt_BR` evaluation bug (`6.20%` -> `64.00%`):** Fixed `eval_ispell_pipe` in `scripts/compare_system_spellers.py` and `scripts/paper_analyses.py` to handle `*`, `+`, `-`, `#` ispell lines and passed `--encoding=utf-8` / `-i utf-8`. Updated `pt_BR` tables in both papers and `README.md`.
5. **Table 1 vs. Table 2 `aspell` Top-5 discrepancy (`75.78%` vs. `75.97%`):** Made tie-breaking in `SpellChecker.rank_candidates` deterministic (`(-score, candidate)`), yielding `75.97%` across all scripts.
6. **Table 1 improvement range:** Corrected to `+1.33 to +2.59 percentage points` (and consistently used `percentage points` for absolute accuracy differences).
7. **Equation (3) formatting & bounds:** Fixed norm notation (`\lVert \mathbf{v}_u \rVert_2 \, \lVert \mathbf{v}_v \rVert_2`), derived $\frac{1}{2}(1-\cos) \in [0, 1]$ for `{-1, 0, +1}`, and added active-feature sparsity stats (`18.25 ± 1.50` consonants vs. `19.94 ± 0.87` vowels).
8. **Birkbeck dev vs. Holbrook eval independence:** Verified `0` exact pair overlap in `scripts/run_review_experiments.py` and documented it in both papers.
9. **Frequency smoothing & noisy-channel framing:** Specified $\log P(w) = \log(c(w)+0.5) - \log \sum_{v \in V} c(v)$, floor `1`, diacritic preservation, and reframed Eq. (4) as a *noisy-channel-inspired scoring function* (and generalized weighted edit distance).

## 2. Empirical & Methodological Additions (Completed)
1. **5-Configuration Component Ablation (`A..E`):** Added Table 3 (`A: Orth+Freq`, `B: +Kb`, `C: +Phon`, `D: -Freq`, `E: Full Unified`) across Norvig1, Aspell, Holbrook, TOEFL-Spell, and PT-Wiki.
2. **Two-stage candidate-generation recall breakdown:** Added Table 2 separating `SymDelete only`, `Skeleton only`, `Combined recall`, and `Avg. candidates/query`.
3. **Statistical significance & baseline vocabulary coverage:** Added two-sided McNemar $p$-values, 95% confidence intervals, Holm–Bonferroni note, and `Ours / Aspell / Hunspell` target vocabulary coverage.
4. **Full A–Z Portuguese evaluation ($n=1{,}149$):** Evaluated the complete Portuguese Wikipedia list alongside the $n=500$ slice (`72.15%` vs. `66.23%` Norvig, `63.19%` Hunspell, `60.14%` Aspell) to rule out alphabetical prefix bias.
5. **Hardware, latency, and memory footprint:** Reported CPU (`Intel Core i7-13650HX`), RAM (`14 GiB`), OS (`Linux 7.0`), Python 3.14 + NumPy, index build times, warm-cache latencies, and RSS memory (`497.4 MiB` EN, `1,836.6 MiB` PT).
6. **Broader cross-lingual validation (6 languages, 4 families, 3 scripts):** Added **Russian** (Slavic/Cyrillic, `85.31%` vs. `79.38%` Norvig), **Hebrew** (Semitic/Abjad, `55.10%` vs. `53.06%`), **Turkish** (Turkic/Latin, `81.28%` tuned vs. `77.95%`), and **French** (opaque Romance/Latin, `59.32%` vs. `57.63%`), plus fixed vs. per-language weight tuning and summary bar plots (`paper/figures/results_overview.pdf`).

## 3. Deferred to Future Work
* Masked/feature-aware cosine distance on mutually applicable features.
* $1$-to-$2$ phone alignments (ALINE-style) and multiple pronunciations per word $\Phi(w)$.
* Epitran integration for zero-shot G2P across low-resource languages.
* Lightweight $n$-gram context for real-word error correction (`their`/`there`, `mas`/`mais`).
* Neural / transformer re-ranking baselines.
