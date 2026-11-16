"""Generate publication-quality grouped bar plots for the papers and README."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0), gridspec_kw={"width_ratios": [1.45, 1.0]})

    # Panel (a): English & Portuguese vs Aspell, Hunspell, Norvig
    datasets = ["Norvig1", "Norvig2", "Aspell", "Wiki-EN", "Holbrook", "TOEFL", "PT-500", "PT-1149"]
    aspell = [75.56, 77.25, 58.72, 83.33, 26.67, 72.80, 64.00, 60.14]
    hunspell = [64.44, 65.50, 53.49, 55.33, 17.33, 66.90, 64.20, 63.19]
    norvig = [74.81, 72.25, 52.71, 77.00, 27.67, 76.80, 66.60, 66.23]
    unified = [79.26, 74.75, 56.40, 81.67, 32.00, 77.10, 73.00, 72.15]

    x1 = np.arange(len(datasets))
    w1 = 0.20
    ax1.bar(x1 - 1.5 * w1, aspell, w1, label="Aspell", color="#9ecae1", edgecolor="#333333", linewidth=0.4)
    ax1.bar(x1 - 0.5 * w1, hunspell, w1, label="Hunspell", color="#fdae6b", edgecolor="#333333", linewidth=0.4)
    ax1.bar(x1 + 0.5 * w1, norvig, w1, label="Norvig", color="#a1d99b", edgecolor="#333333", linewidth=0.4)
    ax1.bar(x1 + 1.5 * w1, unified, w1, label="LipSpell", color="#2b5c8f", edgecolor="#111111", linewidth=0.5)

    ax1.set_ylabel("Top-1 Accuracy (%)")
    ax1.set_title("(a) English & Portuguese vs. Baselines")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(datasets, rotation=22, ha="right")
    ax1.set_ylim(0, 96)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(frameon=False, ncol=2, loc="upper right")

    # Panel (b): Multilingual Cross-Family Transfer (Aspell vs Hunspell vs Norvig vs LipSpell)
    langs = ["English\n(Latin)", "Russian\n(Cyrillic)", "Hebrew\n(Abjad)", "Portuguese\n(Latin)", "Turkish\n(Latin)", "French\n(Latin)"]
    m_asp = [77.51, 71.96, 39.34, 67.13, 4.29, 62.26]
    m_hun = [69.94, 67.60, 33.61, 70.14, 47.14, 82.08]
    m_norv = [79.14, 78.50, 62.30, 76.75, 79.05, 67.92]
    m_fix = [80.37, 85.67, 63.93, 83.57, 79.52, 68.87]
    m_tun = [79.75, 84.74, 74.59, 85.97, 81.90, 73.58]

    x2 = np.arange(len(langs))
    w2 = 0.16
    ba = ax2.bar(x2 - 2.0 * w2, m_asp, w2, label="Aspell", color="#9ecae1", edgecolor="#333333", linewidth=0.4)
    b0 = ax2.bar(x2 - 1.0 * w2, m_hun, w2, label="Hunspell", color="#fdae6b", edgecolor="#333333", linewidth=0.4)
    b1 = ax2.bar(x2, m_norv, w2, label="Norvig", color="#a1d99b", edgecolor="#333333", linewidth=0.4)
    b2 = ax2.bar(x2 + 1.0 * w2, m_fix, w2, label="LipSpell (Fixed)", color="#5b8db8", edgecolor="#111111", linewidth=0.4)
    b3 = ax2.bar(x2 + 2.0 * w2, m_tun, w2, label="LipSpell (Tuned)", color="#2b5c8f", edgecolor="#111111", linewidth=0.5)

    for bars in (ba, b0, b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.8,
                f"{h:.0f}",
                ha="center",
                va="bottom",
                fontsize=5.5,
            )

    ax2.set_ylabel("Top-1 Accuracy (%)")
    ax2.set_title("(b) Cross-Family Transfer (50/50 Non-Word Splits)")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(langs)
    ax2.set_ylim(0, 108)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(frameon=False, ncol=3, loc="upper center", fontsize=7.0)

    plt.tight_layout()
    plt.savefig(out_dir / "results_overview.pdf", bbox_inches="tight")
    plt.savefig(out_dir / "results_overview.png", dpi=200, bbox_inches="tight")
    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
