"""Figure for section 5.4 (research question 4): samples actually used and reduction vs. the nominal budget.

Reads submission_analysis/summary.csv (read-only).
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

from analysis import PLOT_STYLE, log_xlim  # noqa: E402

RUNS = [("aime_4o", "AIME, gpt-4o-mini", "tab:blue", "o"),
        ("aime_41mini", "AIME, gpt-4.1-mini", "tab:orange", "s"),
        ("gsm8k_4o", "GSM8K, gpt-4o-mini", "tab:green", "^")]
REDUCTION_LABEL_DY = {"aime_4o": -16, "aime_41mini": -4, "gsm8k_4o": 8}


def main():
    with (REPO / "submission_analysis" / "summary.csv").open(encoding="utf-8", newline="") as fh:
        summary = list(csv.DictReader(fh))

    plt.rcParams.update(PLOT_STYLE)
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.2))
    ks = sorted({int(s["k"]) for s in summary})
    left.plot(ks, ks, linestyle="--", color="0.5", linewidth=1.5, label="no early stopping (k_used = k_max)")

    for run, label, color, marker in RUNS:
        rows = sorted((s for s in summary if s["run"] == run), key=lambda s: int(s["k"]))
        x = [int(s["k"]) for s in rows]
        used = [float(s["mean_k_used"]) for s in rows]
        reduction = [float(s["compute_savings_vs_nominal_pct"]) for s in rows]
        left.plot(x, used, marker=marker, markersize=7, linewidth=2.2, color=color, label=label)
        right.plot(x, reduction, marker=marker, markersize=7, linewidth=2.2, color=color, label=label)
        left.annotate(f"{used[-1]:.2f}", (x[-1], used[-1]), textcoords="offset points", xytext=(8, -4),
                      fontsize=10, color=color)
        right.annotate(f"{reduction[-1]:.2f}%", (x[-1], reduction[-1]), textcoords="offset points",
                       xytext=(8, REDUCTION_LABEL_DY[run]), fontsize=10, color=color)

    for ax in (left, right):
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlim(log_xlim(ks)[0], log_xlim(ks)[1] * 1.6)
        ax.set_xlabel("Nominal sampling budget k_max (log scale)")

    left.set_yscale("log")
    left.set_ylabel("Mean samples used per problem, k_used (log scale)")
    left.set_title("(a) Samples actually generated")
    left.legend(loc="upper left", fontsize=10)

    right.set_ylim(-5, 105)
    right.set_ylabel("Reduction vs. nominal budget (%)")
    right.set_title("(b) Reduction: 1 − mean k_used / k_max")
    right.legend(loc="upper left", fontsize=10)

    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(HERE / f"early_stopping_samples.{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
