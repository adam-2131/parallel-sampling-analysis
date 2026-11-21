"""Figure for section 5.3 (research questions 2 and 3): marginal accuracy gain per tenfold increase of k.

Reads submission_analysis/marginal_gains.csv and mcnemar_tests.csv (read-only); bars whose paired McNemar
test is significant after Holm correction are marked with an asterisk.
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "scripts"))

from analysis import PLOT_STYLE  # noqa: E402

ANALYSIS = REPO / "submission_analysis"
RUNS = [("aime_4o", "AIME, gpt-4o-mini", "tab:blue"),
        ("aime_41mini", "AIME, gpt-4.1-mini", "tab:orange"),
        ("gsm8k_4o", "GSM8K, gpt-4o-mini", "tab:green")]
STEPS = [(1, 10), (10, 100), (100, 1000)]


def read(name):
    with (ANALYSIS / name).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def main():
    gains = {(r["run"], int(r["from_k"]), int(r["to_k"])): float(r["gain_pp"]) for r in read("marginal_gains.csv")}
    holm = {(r["run"], r["comparison"]): float(r["p_holm"]) for r in read("mcnemar_tests.csv") if r["p_holm"]}

    plt.rcParams.update(PLOT_STYLE)
    fig, ax = plt.subplots()
    width = 0.26
    for i, (run, label, color) in enumerate(RUNS):
        xs = [s + (i - 1) * width for s in range(len(STEPS))]
        values = [gains[(run, a, b)] for a, b in STEPS]
        bars = ax.bar(xs, values, width, color=color, edgecolor="black", linewidth=0.6, label=label)
        for bar, (a, b), v in zip(bars, STEPS, values):
            star = "*" if holm[(run, f"k={a} vs k={b}")] < 0.05 else ""
            ax.annotate(f"{v:+.2f}{star}", (bar.get_x() + bar.get_width() / 2, v), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=10)

    ax.set_xticks(range(len(STEPS)))
    ax.set_xticklabels([f"k = {a} → {b}" for a, b in STEPS])
    ax.set_ylabel("Accuracy gain (percentage points)")
    ax.set_ylim(0, 13.5)
    ax.set_title("Marginal accuracy gain per tenfold increase of k")
    ax.legend(loc="upper right")
    ax.text(0.99, 0.70, "* significant after Holm correction\n   (exact McNemar test, α = 0.05)",
            transform=ax.transAxes, ha="right", va="top", fontsize=10, color="0.25")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()

    for suffix in ("png", "pdf"):
        fig.savefig(HERE / f"marginal_gains.{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
