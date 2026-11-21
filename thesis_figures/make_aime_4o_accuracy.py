"""Figure for section 5.2 (research question 1): gpt-4o-mini accuracy on AIME as a function of k.

Reads the final run in experiments_aime_4o_16k/ (read-only) and writes PNG and PDF next to this script.
"""

import sys
from math import sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "scripts"))

from analysis import PLOT_STYLE, accuracy, load_run, log_xlim  # noqa: E402

RUN_DIR = REPO / "experiments_aime_4o_16k"
COLOR = "tab:blue"


def wilson(successes, n, z=1.959964):
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def main():
    by_k, ids = load_run(RUN_DIR)
    ks = sorted(by_k)
    n = len(ids)
    acc = [100 * accuracy(by_k[k], ids) for k in ks]
    ci = [wilson(round(a * n / 100), n) for a in acc]
    lo = [a - 100 * low for a, (low, _) in zip(acc, ci)]
    hi = [100 * high - a for a, (_, high) in zip(acc, ci)]

    plt.rcParams.update(PLOT_STYLE)
    fig, ax = plt.subplots()
    ax.errorbar(ks, acc, yerr=[lo, hi], marker="o", markersize=8, linewidth=2.5, capsize=5, color=COLOR,
                label="gpt-4o-mini (majority vote)")

    for k, a in zip(ks, acc):
        ax.annotate(f"{a:.2f}%", (k, a), textcoords="offset points", xytext=(9, -17), ha="left",
                    fontsize=11, color=COLOR)

    for (k0, a0), (k1, a1) in zip(zip(ks, acc), zip(ks[1:], acc[1:])):
        x = sqrt(k0 * k1)
        ax.annotate(f"{a1 - a0:+.2f} pp", (x, (a0 + a1) / 2), textcoords="offset points", xytext=(0, 14),
                    ha="center", fontsize=10.5, color="0.3")

    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlim(*log_xlim(ks))
    ax.set_ylim(10, 40)
    ax.set_xlabel("Maximum sampling budget k (log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"AIME accuracy vs. sampling budget (gpt-4o-mini, n = {n})")
    ax.legend(loc="lower right")
    fig.tight_layout()

    for suffix in ("png", "pdf"):
        fig.savefig(HERE / f"aime_4o_mini_accuracy_vs_k.{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)

    for k, a, (low, high) in zip(ks, acc, ci):
        print(f"k={k:<5} accuracy {a:.2f}%  Wilson 95% CI {100 * low:.2f}-{100 * high:.2f}%")


if __name__ == "__main__":
    main()
