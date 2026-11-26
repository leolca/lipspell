"""Export compact web-demo data files (IPA features + top-80k PT/EN lexicons) to docs/data/."""

from __future__ import annotations

import json
from pathlib import Path

from phoneticspell.features import DEFAULT_SALIENCE, DistinctiveFeatures
from phoneticspell.spellchecker import load_corpus_counts, merge_wordlist
from scripts.eval_portuguese import load_pt_counts


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "docs" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Distinctive features JSON
    feat = DistinctiveFeatures(metric="cosine")
    feat_payload = {
        "features": feat.feature_names,
        "salience": DEFAULT_SALIENCE,
        "phones": feat.phones,
        "vectors": feat.vectors.astype(int).tolist(),
    }
    (out_dir / "features.json").write_text(
        json.dumps(feat_payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    # 2. Top-80k Portuguese lexicon
    pt_counts = load_pt_counts(
        root / "data/enelvo/freq-cgu.txt",
        root / "data/enelvo/unitex-full-clean+enelvo-ja-corrigido.txt",
    )
    pt_top = pt_counts.most_common(80_000)
    (out_dir / "pt_80k.tsv").write_text(
        "\n".join(f"{w}\t{c}" for w, c in pt_top) + "\n",
        encoding="utf-8",
    )

    # 3. Top-80k English lexicon
    en_data = Path("/home/leoca/ee/research/spellcheck/data")
    en_counts = merge_wordlist(
        load_corpus_counts(en_data / "big.txt"),
        Path("/usr/share/dict/american-english"),
        1,
    )
    en_top = en_counts.most_common(80_000)
    (out_dir / "en_80k.tsv").write_text(
        "\n".join(f"{w}\t{c}" for w, c in en_top) + "\n",
        encoding="utf-8",
    )

    print(
        f"Exported features.json ({(out_dir / 'features.json').stat().st_size} B), "
        f"pt_80k.tsv ({len(pt_top)} words), en_80k.tsv ({len(en_top)} words)"
    )


if __name__ == "__main__":
    main()
