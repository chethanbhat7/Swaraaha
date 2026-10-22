"""Generate the two bar charts used in paper.tex.

Every value below is transcribed verbatim from the manuscript's own result
tables (Table ``tab:macro`` and Table ``tab:boli_drift``), which were taken
from model/evaluation/reports/comparative_study_report.json. Nothing is
recomputed or estimated here, so the charts cannot drift from the tables.

Run with the project virtualenv, which provides matplotlib:
    .venv/bin/python docs/papers/comparison-study-latex/figures/generate_charts.py
"""

import os
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

W2V = "#2F6690"  # Wav2Vec2 family
CNN = "#D1741F"  # compact CNN family
GREY = "#7A7A7A"

ARMS = ["arm01", "arm02", "arm03", "arm04", "arm05", "arm06", "arm07"]
FAMILY = [W2V, W2V, W2V, CNN, CNN, CNN, CNN]

# Table tab:macro, "Boli F1 (tuned)" column.
BOLI_F1 = [0.292, 0.238, 0.214, 0.332, 0.476, 0.508, 0.345]
# Table tab:macro, "Test F1 (tuned)" column.
TEST_F1 = [0.570, 0.556, 0.378, 0.256, 0.262, 0.249, 0.271]
# Table tab:boli_drift.
BOLI_AUROC = [0.620, 0.621, 0.538, 0.473, 0.497, 0.522, 0.500]
BOLI_PPR = [0.079, 0.102, 0.204, 0.574, 0.615, 0.879, 0.438]

BASE_RATE = 0.408  # macro prevalence on the 53-clip external set
ALL_POSITIVE_F1 = 0.550  # macro F1 of the all-positive baseline
CHANCE_AUROC = 0.500

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times", "Nimbus Roman", "Times New Roman"],
        "font.size": 6.4,
        "axes.labelsize": 6.4,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "legend.fontsize": 5.8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
        # base-14 fonts keep pdfTeX from re-embedding the chart text as Type 3
        "pdf.use14corefonts": True,
        "pdf.compression": 9,
    }
)


def drift(path):
    """RQ3: where the fixed thresholds landed on the external set."""
    fig, ax = plt.subplots(figsize=(2.62, 1.62))
    ax.barh(ARMS, BOLI_PPR, color=FAMILY, height=0.66, zorder=3)
    ax.axvline(BASE_RATE, color="black", ls="--", lw=0.9, zorder=4)
    ax.text(
        BASE_RATE + 0.015,
        -0.62,
        f"base rate {BASE_RATE:.3f}",
        ha="left",
        va="center",
        fontsize=6.0,
    )
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Predicted-positive rate (macro)")
    ax.invert_yaxis()
    ax.grid(axis="x", color="0.85", lw=0.5, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=W2V),
        plt.Rectangle((0, 0), 1, 1, color=CNN),
    ]
    ax.legend(
        handles,
        ["Wav2Vec2", "compact CNN"],
        loc="lower right",
        frameon=False,
        ncol=1,
        handlelength=1.0,
        columnspacing=0.9,
    )
    fig.tight_layout(pad=0.3)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def disagreement(path):
    """RQ2 against RQ4: the two metric families order the models differently."""
    fig, ax = plt.subplots(figsize=(2.62, 1.62))
    x = range(len(ARMS))
    w = 0.38
    ax.bar(
        [i - w / 2 for i in x],
        BOLI_F1,
        width=w,
        color=FAMILY,
        zorder=3,
        label="macro F1 (tuned)",
    )
    ax.bar(
        [i + w / 2 for i in x],
        BOLI_AUROC,
        width=w,
        color=FAMILY,
        alpha=0.45,
        hatch="////",
        edgecolor="white",
        lw=0.0,
        zorder=3,
        label="macro AUROC",
    )
    ax.axhline(ALL_POSITIVE_F1, color="black", ls="--", lw=0.9, zorder=4)
    ax.axhline(CHANCE_AUROC, color=GREY, ls=":", lw=0.9, zorder=4)
    ax.text(6.45, ALL_POSITIVE_F1 + 0.012, "all-positive F1", ha="right", va="bottom", fontsize=5.8)
    ax.text(6.45, CHANCE_AUROC - 0.028, "chance AUROC", ha="right", va="top", fontsize=5.8, color=GREY)
    ax.set_xticks(list(x))
    ax.set_xticklabels(ARMS)
    ax.set_ylim(0, 0.72)
    ax.set_ylabel("Cross-corpus score")
    ax.grid(axis="y", color="0.85", lw=0.5, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=W2V),
        plt.Rectangle((0, 0), 1, 1, color=CNN),
        plt.Rectangle((0, 0), 1, 1, color="0.6", alpha=0.45),
    ]
    ax.legend(
        handles,
        ["Wav2Vec2", "compact CNN", "hatched: AUROC"],
        loc="upper left",
        frameon=False,
        ncol=1,
        handlelength=1.1,
        columnspacing=0.8, handletextpad=0.4,
    )
    fig.tight_layout(pad=0.3)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def to_type1(path):
    """Rewrite chart fonts as Type 1C.

    matplotlib emits Identity-H CID fonts, which pdfTeX can only embed by
    converting them to bitmap Type 3. Passing the file through Ghostscript
    yields Type 1C subsets instead, so the figure text stays vector.
    """
    tmp = path + ".gs"
    subprocess.run(
        [
            "gs", "-q", "-o", tmp,
            "-sDEVICE=pdfwrite",
            "-dPDFSETTINGS=/prepress",
            "-dCompatibilityLevel=1.7",
            "-dSubsetFonts=true",
            "-dEmbedAllFonts=true",
            path,
        ],
        check=True,
    )
    os.replace(tmp, path)


if __name__ == "__main__":
    for name in ("ppr_drift.pdf", "metric_disagreement.pdf"):
        to_type1(name)
    print("wrote ppr_drift.pdf and metric_disagreement.pdf")
