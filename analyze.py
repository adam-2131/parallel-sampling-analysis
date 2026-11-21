"""Offline statistical analysis of the three final 16k-token runs.

    python submission_analysis/analyze.py

Reads the final per-problem CSVs and the response cache, never writes to
them, and never contacts the OpenAI API: network sockets are disabled before
anything else runs, and a response missing from the cache is an error.

Steps
  1. Validate the 12 CSVs (row count, ids, k_max, k_used, correct, vote counts).
  2. Replay every run from the cache through the live code (eval.py, parse.py)
     and check that it reproduces every CSV row.
  3. Descriptive statistics, marginal gains, compute savings.
  4. Exact McNemar tests with Holm correction (adjacent k; cross-model AIME).
  5. Response-level diagnostics reconstructed from the cache.
  6. Historical comparison: original 2048-token runs, post-hoc corrected
     rescoring, and the final 16k runs, kept separate.
  7. Figures, reports, configuration summary and a SHA-256 manifest.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import socket
import statistics
import subprocess
import sys
from datetime import datetime
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def forbid_network():
    def refuse(*args, **kwargs):
        raise RuntimeError("network access is disabled during the offline analysis")

    socket.socket.connect = refuse
    socket.socket.connect_ex = refuse
    socket.create_connection = refuse


forbid_network()
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import eval as live_eval  # noqa: E402
import utils  # noqa: E402
from data import load_jsonl_dataset  # noqa: E402
from main import COLUMNS, TEMPERATURE, select_subset  # noqa: E402
from parse import extract_answer, normalize_gold  # noqa: E402
from prompting import format_math_prompt  # noqa: E402

utils.CACHE_DIR = Path(os.environ.get("RESPONSE_CACHE_DIR", REPO / ".cache"))

K_VALUES = [1, 10, 100, 1000]
ADJACENT = [(1, 10), (10, 100), (100, 1000)]
N_PROBLEMS = 300
ALPHA = 0.05

# Settings of the final runs, as passed on the command line (console logs).
SETTINGS = {
    "temperature": TEMPERATURE,
    "max_tokens": 16384,
    "seed": 0,
    "seed_tag": "fixed_k",
    "subset_size": 300,
    "subset_seed": 123,
    "k_values": K_VALUES,
    "early_stop_confidence": 0.95,
    "early_stop_stability": 50,
    "early_stop_min_samples": 10,
    "early_stop_min_valid": 10,
}

RUNS = {
    "aime_41mini": {
        "label": "AIME, gpt-4.1-mini", "model": "gpt-4.1-mini", "dataset": "data/aime.jsonl",
        "dir": "experiments_aime_41mini_16k", "max_workers": 64,
        "historical_original": "experiments_aime_41mini", "historical_corrected": "corrected_results/aime_41",
    },
    "aime_4o": {
        "label": "AIME, gpt-4o-mini", "model": "gpt-4o-mini", "dataset": "data/aime.jsonl",
        "dir": "experiments_aime_4o_16k", "max_workers": 48,
        "historical_original": "experiments", "historical_corrected": "corrected_results/aime_4o",
    },
    "gsm8k_4o": {
        "label": "GSM8K, gpt-4o-mini", "model": "gpt-4o-mini", "dataset": "data/gsm8k.jsonl",
        "dir": "experiments_gsm8k_16k", "max_workers": 32,
        "historical_original": "experiments_gsm8k", "historical_corrected": "corrected_results/gsm8k_4o",
    },
}

# Wall-clock times printed by the console at the end of each invocation.
# They are not derivable from the CSVs and are reported as recorded.
WALL_CLOCK = {
    "aime_41mini": [
        {"session": 1, "started_utc": "2026-11-17T20:05:25Z", "ended_utc": "2026-11-17T23:16:38Z",
         "elapsed": "03:11:10", "note": "stopped manually during k=100 (6 problems left)"},
        {"session": 2, "started_utc": "2026-11-18T05:30:10Z", "ended_utc": "2026-11-18T09:55:22Z",
         "elapsed": "04:25:10", "note": "resumed; finished responses were read from the cache"},
    ],
    "aime_4o": [{"session": 1, "started_utc": "2026-11-17T20:05:28Z", "ended_utc": "2026-11-17T23:10:34Z",
                 "elapsed": "03:05:03", "note": "ran concurrently with the other two runs"}],
    "gsm8k_4o": [{"session": 1, "started_utc": "2026-11-17T20:05:30Z", "ended_utc": "2026-11-17T20:39:38Z",
                  "elapsed": "00:34:06", "note": "ran concurrently with the other two runs"}],
}

COLORS = {"aime_41mini": "tab:orange", "aime_4o": "tab:blue", "gsm8k_4o": "tab:green"}
MARKERS = {"aime_41mini": "s", "aime_4o": "o", "gsm8k_4o": "^"}

plt.rcParams.update({
    "figure.figsize": (8, 5), "savefig.dpi": 200, "font.size": 11, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.4,
})


class ValidationError(RuntimeError):
    pass


# ---------------------------------------------------------------- helpers

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def historical_path(relative):
    """Historical results sit in the repository root, or under historical/ in the submission bundle."""
    path = REPO / relative
    return path if path.exists() else REPO / "historical" / relative


def csv_text(rows, columns):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def mcnemar_exact(b, c):
    """Two-sided exact McNemar test: binomial test with p = 0.5 on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(p_values):
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    m = len(p_values)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def wilson(successes, n, z=1.959963984540054):
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def pct(x, digits=2):
    return f"{100 * x:.{digits}f}"


def fmt_p(p):
    return f"{p:.2e}" if p < 1e-4 else f"{p:.4f}"


# ---------------------------------------------------------------- 1. validation

def validate(run, cfg, problems, findings):
    ids = [p["id"] for p in problems]
    if len(ids) != N_PROBLEMS or len(set(ids)) != N_PROBLEMS:
        raise ValidationError(f"{run}: subset does not have {N_PROBLEMS} distinct problems")
    gold = {p["id"]: normalize_gold(p["answer"]) for p in problems}

    rows = {}
    for k in K_VALUES:
        path = REPO / cfg["dir"] / f"k_{k}.csv"
        data = read_csv(path)
        name = f"{cfg['dir']}/k_{k}.csv"
        checks = []

        def check(label, ok, detail=""):
            checks.append((label, ok, detail))
            if not ok:
                raise ValidationError(f"{name}: {label} failed {detail}")

        check("columns match main.py", list(data[0]) == COLUMNS if data else False)
        check("exactly 300 rows", len(data) == N_PROBLEMS, f"({len(data)})")
        row_ids = [r["example_id"] for r in data]
        check("no duplicate ids", len(set(row_ids)) == len(row_ids))
        check("no missing ids", set(row_ids) == set(ids))
        check("same order as the deterministic subset", row_ids == ids)
        check("k_max equals k", all(r["k_max"] == str(k) for r in data))
        k_used = [int(r["k_used"]) for r in data]
        check("1 <= k_used <= k_max", all(1 <= u <= k for u in k_used))
        expected_fixed = min(k, SETTINGS["early_stop_min_samples"])
        check(f"k_used >= {expected_fixed} (minimum samples)", all(u >= expected_fixed for u in k_used))
        check("correct is 0 or 1", all(r["correct"] in {"0", "1"} for r in data))
        check("valid_votes + invalid_votes = k_used",
              all(int(r["valid_votes"]) + int(r["invalid_votes"]) == int(r["k_used"]) for r in data))
        check("0 <= token_limit_votes <= k_used",
              all(0 <= int(r["token_limit_votes"]) <= int(r["k_used"]) for r in data))
        check("empty prediction iff no valid votes",
              all((r["prediction"] == "") == (r["valid_votes"] == "0") for r in data))
        check("correct recomputed from prediction and gold answer",
              all(r["correct"] == str(int(r["prediction"] != "" and r["prediction"] in gold[r["example_id"]]))
                  for r in data))
        check("confidence in [0, 1]", all(0.0 <= float(r["confidence"]) <= 1.0 for r in data))
        check("entropy >= 0", all(float(r["entropy"]) >= 0.0 for r in data))
        check("runtime_seconds present and non-negative", all(float(r["runtime_seconds"]) >= 0 for r in data))
        findings.append({"file": name, "checks": checks})
        rows[k] = data

    # Samples are drawn in the same order for every k, so a larger budget can
    # only extend the sequence of a smaller one.
    for lo, hi in ADJACENT:
        for a, b in zip(rows[lo], rows[hi]):
            ua, ub = int(a["k_used"]), int(b["k_used"])
            if ub < ua:
                raise ValidationError(f"{run} {a['example_id']}: k_used decreases from k={lo} to k={hi}")
            if ua < lo and ub != ua:
                raise ValidationError(f"{run} {a['example_id']}: stopped early at k={lo} but not at the same "
                                      f"point at k={hi}")
            if ua < lo and (a["prediction"], a["correct"]) != (b["prediction"], b["correct"]):
                raise ValidationError(f"{run} {a['example_id']}: early-stopped result differs between k={lo} "
                                      f"and k={hi}")
    findings.append({"file": f"{cfg['dir']} (across k)", "checks": [
        ("k_used never decreases with larger k", True, ""),
        ("problems that stopped before k_max give identical results at larger k", True, ""),
    ]})
    return rows


# ---------------------------------------------------------------- 2. cache replay and diagnostics

def load_entry(model, prompt, seed):
    request = {"model": model, "temperature": SETTINGS["temperature"], "max_tokens": SETTINGS["max_tokens"],
               "prompt": prompt, "seed": seed}
    path = utils._cache_path(request)
    if not path.is_file():
        return None, path
    return json.loads(path.read_text(encoding="utf-8")), path


def replay_from_cache(run, cfg, problems, rows):
    """Run the live eval.py on cached responses only and compare every CSV row."""
    def cache_only(prompt, *, model, temperature, max_tokens, seed=None):
        request = {"model": model, "temperature": temperature, "max_tokens": max_tokens,
                   "prompt": prompt, "seed": seed}
        output = utils._load_cached(request)
        if output is None:
            raise utils.FatalModelError(f"not in cache: {utils._cache_path(request).name}")
        return output

    live_eval.call_model = cache_only
    compared = 0
    for k in K_VALUES:
        results = live_eval.run_fixed_k_majority(
            problems, cfg["model"], SETTINGS["temperature"], SETTINGS["max_tokens"], k_max=k,
            seed=SETTINGS["seed"], early_stop_stability=SETTINGS["early_stop_stability"],
            early_stop_confidence=SETTINGS["early_stop_confidence"],
            early_stop_min_samples=SETTINGS["early_stop_min_samples"],
            early_stop_min_valid=SETTINGS["early_stop_min_valid"], max_workers=1,
        )
        for result, row in zip(results, rows[k]):
            expected = {
                "example_id": result["example_id"], "k_used": str(result["k_used"]),
                "valid_votes": str(result["valid_votes"]), "invalid_votes": str(result["invalid_votes"]),
                "token_limit_votes": str(result["token_limit_votes"]),
                "prediction": "" if result["prediction"] is None else result["prediction"],
                "confidence": f"{result['confidence']:.6f}", "entropy": f"{result['entropy']:.6f}",
                "correct": str(int(result["correct"])), "completion_tokens": str(result["completion_tokens"]),
            }
            for field, value in expected.items():
                if row[field] != value:
                    raise ValidationError(f"{run} k={k} {row['example_id']}: {field} {row[field]!r} "
                                          f"not reproduced from the cache ({value!r})")
            compared += 1
    return compared


def response_diagnostics(run, cfg, problems, rows):
    per_response = []
    for index, problem in enumerate(problems):
        prompt = format_math_prompt(problem["problem"])
        drawn = int(rows[1000][index]["k_used"])
        for call_index in range(drawn):
            seed = live_eval._derive_seed(SETTINGS["seed"], problem_index=index, call_index=call_index,
                                          tag=SETTINGS["seed_tag"])
            entry, path = load_entry(cfg["model"], prompt, seed)
            if entry is None:
                raise ValidationError(f"{run} {problem['id']} call {call_index}: {path.name} missing from cache")
            output = utils.ModelOutput(**entry)
            truncated_live = live_eval.is_truncated(output, SETTINGS["max_tokens"])
            truncated_tokens = (output.completion_tokens or 0) >= SETTINGS["max_tokens"]
            answer = extract_answer(output.text, token_limit=truncated_live)
            answer_by_token_rule = extract_answer(output.text, token_limit=truncated_tokens)
            per_response.append({
                "run": run, "problem_index": index, "example_id": problem["id"], "call_index": call_index,
                "seed": seed, "completion_tokens": output.completion_tokens,
                "prompt_tokens": output.prompt_tokens,
                "finish_reason": "" if output.finish_reason is None else output.finish_reason,
                "truncated": int(truncated_live), "extracted_answer": "" if answer is None else answer,
                "valid": int(answer is not None),
                "_rules_agree": truncated_live == truncated_tokens and answer == answer_by_token_rule,
                "_cache_file": path.name,
                "_written_utc": path.stat().st_mtime,
            })
    return per_response


def summarize_responses(run, per_response, rows):
    by_problem = {}
    for r in per_response:
        by_problem.setdefault(r["problem_index"], []).append(r)

    out = []
    scopes = [("all generated", None)] + [(f"used at k={k}", k) for k in K_VALUES]
    for scope, k in scopes:
        selected = []
        for index, responses in by_problem.items():
            limit = len(responses) if k is None else int(rows[k][index]["k_used"])
            selected.extend(responses[:limit])
        n = len(selected)
        reasons = {}
        for r in selected:
            reasons[r["finish_reason"] or "missing"] = reasons.get(r["finish_reason"] or "missing", 0) + 1
        tokens = [r["completion_tokens"] for r in selected if r["completion_tokens"] is not None]
        valid = sum(r["valid"] for r in selected)
        truncated = sum(r["truncated"] for r in selected)
        out.append({
            "run": run, "scope": scope, "responses": n, "valid_answers": valid, "invalid_responses": n - valid,
            "validity_rate_pct": pct(valid / n), "finish_reason_stop": reasons.get("stop", 0),
            "finish_reason_length": reasons.get("length", 0),
            "finish_reason_other": sum(v for key, v in reasons.items() if key not in {"stop", "length"}),
            "length_rate_pct": pct(reasons.get("length", 0) / n, 3), "truncated_by_live_rule": truncated,
            "truncated_with_valid_answer": sum(r["truncated"] and r["valid"] for r in selected),
            "mean_completion_tokens": f"{statistics.fmean(tokens):.1f}",
            "median_completion_tokens": f"{statistics.median(tokens):.0f}",
            "max_completion_tokens": max(tokens), "total_completion_tokens": sum(tokens),
            "responses_missing_token_count": n - len(tokens),
            "live_and_offline_rules_disagree": sum(not r["_rules_agree"] for r in selected),
        })
    return out


# ---------------------------------------------------------------- 3. statistics

def describe(run, rows):
    out = []
    for k in K_VALUES:
        data = rows[k]
        correct = sum(r["correct"] == "1" for r in data)
        k_used = [int(r["k_used"]) for r in data]
        runtime = [float(r["runtime_seconds"]) for r in data]
        invalid = sum(r["prediction"] == "" for r in data)
        lo, hi = wilson(correct, N_PROBLEMS)
        out.append({
            "run": run, "k": k, "problems": N_PROBLEMS, "correct": correct,
            "accuracy_pct": pct(correct / N_PROBLEMS), "wilson95_low_pct": pct(lo), "wilson95_high_pct": pct(hi),
            "mean_k_used": f"{statistics.fmean(k_used):.2f}", "median_k_used": f"{statistics.median(k_used):g}",
            "max_k_used": max(k_used), "samples_used": sum(k_used),
            "compute_savings_vs_nominal_pct": pct(1 - statistics.fmean(k_used) / k),
            "mean_confidence": f"{statistics.fmean(float(r['confidence']) for r in data):.4f}",
            "mean_entropy_bits": f"{statistics.fmean(float(r['entropy']) for r in data):.4f}",
            "mean_runtime_s": f"{statistics.fmean(runtime):.2f}", "sum_runtime_s": f"{sum(runtime):.1f}",
            "invalid_final_predictions": invalid, "invalid_final_predictions_pct": pct(invalid / N_PROBLEMS),
            "valid_votes": sum(int(r["valid_votes"]) for r in data),
            "invalid_votes": sum(int(r["invalid_votes"]) for r in data),
            "token_limit_votes": sum(int(r["token_limit_votes"]) for r in data),
            "completion_tokens": sum(int(r["completion_tokens"]) for r in data),
        })
    return out


def marginal(run, summary_rows):
    acc = {int(s["k"]): s["correct"] / N_PROBLEMS for s in summary_rows}
    pairs = ADJACENT + [(1, 1000)]
    return [{"run": run, "from_k": lo, "to_k": hi, "accuracy_from_pct": pct(acc[lo]),
             "accuracy_to_pct": pct(acc[hi]), "gain_pp": f"{100 * (acc[hi] - acc[lo]):+.2f}",
             "type": "adjacent (10x)" if (lo, hi) in ADJACENT else "overall"} for lo, hi in pairs]


def adjacent_tests(run, rows):
    tests = []
    for lo, hi in ADJACENT + [(1, 1000)]:
        a, b = rows[lo], rows[hi]
        w2r = sum(x["correct"] == "0" and y["correct"] == "1" for x, y in zip(a, b))
        r2w = sum(x["correct"] == "1" and y["correct"] == "0" for x, y in zip(a, b))
        tests.append({"run": run, "comparison": f"k={lo} vs k={hi}", "problems": N_PROBLEMS,
                      "accuracy_a_pct": pct(sum(x["correct"] == "1" for x in a) / N_PROBLEMS),
                      "accuracy_b_pct": pct(sum(y["correct"] == "1" for y in b) / N_PROBLEMS),
                      "wrong_to_right": w2r, "right_to_wrong": r2w, "discordant": w2r + r2w,
                      "p_raw": mcnemar_exact(w2r, r2w),
                      "family": "adjacent (Holm, m=3)" if (lo, hi) in ADJACENT else "supplementary, not corrected"})
    family = [t for t in tests if t["family"].startswith("adjacent")]
    for t, adj in zip(family, holm([t["p_raw"] for t in family])):
        t["p_holm"] = adj
    for t in tests:
        t.setdefault("p_holm", "")
        t["significant_at_0.05"] = ("yes" if t["p_holm"] < ALPHA else "no") if t["p_holm"] != "" else \
            ("yes (uncorrected)" if t["p_raw"] < ALPHA else "no (uncorrected)")
    return tests


def cross_model(rows_41, rows_4o):
    tests = []
    for k in K_VALUES:
        a, b = rows_4o[k], rows_41[k]
        if [r["example_id"] for r in a] != [r["example_id"] for r in b]:
            raise ValidationError(f"cross-model k={k}: problem ids differ")
        only_41 = sum(x["correct"] == "0" and y["correct"] == "1" for x, y in zip(a, b))
        only_4o = sum(x["correct"] == "1" and y["correct"] == "0" for x, y in zip(a, b))
        both = sum(x["correct"] == "1" and y["correct"] == "1" for x, y in zip(a, b))
        tests.append({"k": k, "problems": N_PROBLEMS,
                      "gpt4o_mini_accuracy_pct": pct(sum(x["correct"] == "1" for x in a) / N_PROBLEMS),
                      "gpt41_mini_accuracy_pct": pct(sum(y["correct"] == "1" for y in b) / N_PROBLEMS),
                      "both_correct": both, "only_gpt41_mini_correct": only_41, "only_gpt4o_mini_correct": only_4o,
                      "both_wrong": N_PROBLEMS - both - only_41 - only_4o,
                      "difference_pp": f"{100 * (only_41 - only_4o) / N_PROBLEMS:+.2f}",
                      "p_raw": mcnemar_exact(only_41, only_4o)})
    for t, adj in zip(tests, holm([t["p_raw"] for t in tests])):
        t["p_holm"] = adj
        t["significant_at_0.05"] = "yes" if adj < ALPHA else "no"
    return tests


# ---------------------------------------------------------------- 6. history

def historical(run, cfg, final_rows):
    out = []
    original = {k: read_csv(historical_path(cfg["historical_original"]) / f"k_{k}.csv") for k in K_VALUES}
    corrected_dir = historical_path(cfg["historical_corrected"])
    corrected = {k: read_csv(corrected_dir / f"k_{k}.csv") for k in K_VALUES} if corrected_dir.is_dir() else None
    for k in K_VALUES:
        def acc(rows):
            return sum(r["correct"].strip().lower() in {"1", "true"} for r in rows) / len(rows)

        def mean_used(rows):
            return statistics.fmean(int(r["k_used"]) for r in rows)

        stages = [("1 original run (2048 tokens, original parser)", original[k]),
                  ("2 post-hoc corrected rescoring (2048-token responses, conservative parser, original k_used)",
                   corrected[k] if corrected else None),
                  ("3 final run (16384 tokens, conservative parser, coverage-aware stopping)", final_rows[k])]
        for stage, rows in stages:
            if rows is None:
                out.append({"run": run, "k": k, "stage": stage, "problems": "", "accuracy_pct": "unavailable",
                            "mean_k_used": "unavailable"})
                continue
            out.append({"run": run, "k": k, "stage": stage, "problems": len(rows), "accuracy_pct": pct(acc(rows)),
                        "mean_k_used": f"{mean_used(rows):.2f}"})
    same_ids = [r["example_id"] for r in original[1]] == [r["example_id"] for r in final_rows[1]]
    return out, same_ids


# ---------------------------------------------------------------- 6b. token usage and AIME years

def token_usage(response_level_path, rows):
    """Token totals of the responses that enter the votes of each k-condition.

    A response with call_index c of problem p is used at k iff c < k_used(p, k), because the samples of one
    problem are drawn in call_index order and sampling stops after k_used of them.
    """
    responses = {}
    for r in read_csv(response_level_path):
        responses.setdefault(r["run"], {}).setdefault(int(r["problem_index"]), []).append(r)
    out = []
    for run in RUNS:
        previous = None
        for k in K_VALUES:
            n_resp = prompt = completion = 0
            for p_idx, row in enumerate(rows[run][k]):
                k_used = int(row["k_used"])
                used = [r for r in responses[run][p_idx] if int(r["call_index"]) < k_used]
                if len(used) != k_used or used[0]["example_id"] != row["example_id"]:
                    raise ValidationError(f"{run} k={k} {row['example_id']}: response_level.csv does not match k_used")
                if sum(int(r["completion_tokens"]) for r in used) != int(row["completion_tokens"]):
                    raise ValidationError(f"{run} k={k} {row['example_id']}: completion tokens do not match the CSV")
                n_resp += k_used
                prompt += sum(int(r["prompt_tokens"]) for r in used)
                completion += sum(int(r["completion_tokens"]) for r in used)
            total = prompt + completion
            entry = {"run": run, "model": RUNS[run]["model"], "k": k, "problems": len(rows[run][k]),
                     "responses": n_resp, "total_prompt_tokens": prompt, "total_completion_tokens": completion,
                     "total_tokens": total, "mean_total_tokens_per_problem": f"{total / len(rows[run][k]):.1f}",
                     "mean_completion_tokens_per_response": f"{completion / n_resp:.1f}",
                     "previous_k": previous["k"] if previous else "",
                     "total_tokens_added_vs_previous_k": total - previous["total_tokens"] if previous else "",
                     "total_tokens_growth_factor_vs_previous_k":
                         f"{total / previous['total_tokens']:.3f}" if previous else "",
                     "responses_growth_factor_vs_previous_k":
                         f"{n_resp / previous['responses']:.3f}" if previous else ""}
            out.append(entry)
            previous = entry
    return out


def aime_years(problems):
    """Competition year of each selected AIME problem, taken from its id (e.g. '2024-II-11' -> 2024)."""
    selected = [int(p["id"].split("-")[0]) for p in problems]
    pool = [int(p["id"].split("-")[0]) for p in load_jsonl_dataset(REPO / RUNS["aime_41mini"]["dataset"])]
    out = []
    for year in sorted(set(pool)):
        out.append({"year": year, "selected_problems": selected.count(year), "problems_in_dataset": pool.count(year),
                    "selected_pct": f"{100 * selected.count(year) / len(selected):.2f}"})
    return out


def token_figure(tokens, outputs, captions):
    fig, ax = plt.subplots()
    for run in RUNS:
        rows = [t for t in tokens if t["run"] == run]
        ax.plot([t["k"] for t in rows], [t["total_tokens"] / 1e6 for t in rows], marker=MARKERS[run],
                color=COLORS[run], linewidth=2, label=RUNS[run]["label"])
    log_k_axis(ax)
    ax.set_yscale("log")
    ax.set_ylabel("Total API tokens (millions, log scale)")
    ax.set_title("API token consumption vs. nominal budget")
    ax.legend(loc="upper left")
    save(fig, "fig9_token_usage.png", outputs)
    captions["fig9_token_usage.png"] = (
        "Total prompt + completion tokens of the responses that enter the votes at each k (300 problems). "
        "Tokens are an API-usage measure, not a FLOP count; tokens of different models are not comparable "
        "as compute.")


# ---------------------------------------------------------------- 7. figures

def save(fig, name, outputs):
    path = HERE / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path)


def log_k_axis(ax, xlabel="k (nominal sample budget, log scale)"):
    ax.set_xscale("log")
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlim(0.6, 1700)
    ax.set_xlabel(xlabel)


def accuracy_plot(ax, run, summary, label_dy):
    rows = [s for s in summary if s["run"] == run]
    ks = [s["k"] for s in rows]
    acc = [100 * s["correct"] / N_PROBLEMS for s in rows]
    lo = [a - float(s["wilson95_low_pct"]) for a, s in zip(acc, rows)]
    hi = [float(s["wilson95_high_pct"]) - a for a, s in zip(acc, rows)]
    ax.errorbar(ks, acc, yerr=[lo, hi], marker=MARKERS[run], color=COLORS[run], capsize=4, linewidth=2,
                markersize=7, label=RUNS[run]["label"])
    for x, y in zip(ks, acc):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(8, label_dy), ha="left",
                    fontsize=9, color=COLORS[run])


def figures(summary, gains, diag_summary, hist_rows):
    outputs = []
    captions = {}

    fig, ax = plt.subplots()
    accuracy_plot(ax, "aime_41mini", summary, 12)
    accuracy_plot(ax, "aime_4o", summary, -16)
    log_k_axis(ax)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("AIME: accuracy vs. sample budget")
    ax.legend(loc="center right")
    save(fig, "fig1_aime_accuracy_vs_k.png", outputs)
    captions["fig1_aime_accuracy_vs_k.png"] = (
        "Majority-vote accuracy on the same 300 AIME problems for gpt-4.1-mini and gpt-4o-mini at k = 1, 10, "
        "100, 1000 (max_tokens 16384). Error bars: Wilson 95 % intervals per point; they do not account for the "
        "pairing across k, which the McNemar tests do.")

    fig, ax = plt.subplots()
    accuracy_plot(ax, "gsm8k_4o", summary, -16)
    log_k_axis(ax)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(85, 100)
    ax.set_title("GSM8K (gpt-4o-mini): accuracy vs. sample budget")
    save(fig, "fig2_gsm8k_accuracy_vs_k.png", outputs)
    captions["fig2_gsm8k_accuracy_vs_k.png"] = (
        "Majority-vote accuracy of gpt-4o-mini on 300 GSM8K problems. The y-axis starts at 85 % to make the "
        "differences visible. Error bars: Wilson 95 % intervals.")

    fig, ax = plt.subplots()
    transitions = [f"{lo}→{hi}" for lo, hi in ADJACENT]
    width = 0.26
    for offset, run in zip((-width, 0, width), RUNS):
        values = [float(g["gain_pp"]) for g in gains if g["run"] == run and g["type"].startswith("adjacent")]
        xs = [i + offset for i in range(len(transitions))]
        bars = ax.bar(xs, values, width, color=COLORS[run], label=RUNS[run]["label"], edgecolor="#333",
                      linewidth=0.5, zorder=3)
        for bar, v in zip(bars, values):
            ax.annotate(f"{v:+.1f}", (bar.get_x() + bar.get_width() / 2, v), textcoords="offset points",
                        xytext=(0, 3 if v >= 0 else -11), ha="center", fontsize=8.5)
    ax.axhline(0, color="#333", linewidth=0.8)
    ax.set_xticks(range(len(transitions)))
    ax.set_xticklabels([f"k = {t}" for t in transitions])
    ax.set_ylabel("Accuracy gain (percentage points)")
    ax.set_title("Marginal accuracy gain per tenfold increase of k")
    ax.legend()
    ax.grid(False, axis="x")
    save(fig, "fig3_marginal_gains.png", outputs)
    captions["fig3_marginal_gains.png"] = (
        "Change in accuracy (percentage points) for each tenfold increase of the nominal budget. Statistical "
        "significance of each step is given in mcnemar_tests.csv, not by the bar height.")

    fig, ax = plt.subplots()
    ax.plot(K_VALUES, K_VALUES, "--", color="grey", label="no early stopping (k_used = k)")
    for run in RUNS:
        rows = [s for s in summary if s["run"] == run]
        ax.plot([s["k"] for s in rows], [float(s["mean_k_used"]) for s in rows], marker=MARKERS[run],
                color=COLORS[run], linewidth=2, label=RUNS[run]["label"])
    log_k_axis(ax, "k_max (nominal sample budget, log scale)")
    ax.set_yscale("log")
    ax.set_ylabel("Mean samples used per problem (log scale)")
    ax.set_title("Samples actually used vs. nominal budget")
    ax.legend(loc="upper left")
    save(fig, "fig4_effective_samples.png", outputs)
    captions["fig4_effective_samples.png"] = (
        "Mean number of samples drawn per problem (k_used) against the nominal budget k_max. The dashed line "
        "is the budget itself.")

    fig, ax = plt.subplots()
    for run in RUNS:
        rows = [s for s in summary if s["run"] == run]
        ax.plot([s["k"] for s in rows], [float(s["compute_savings_vs_nominal_pct"]) for s in rows],
                marker=MARKERS[run], color=COLORS[run], linewidth=2, label=RUNS[run]["label"])
    log_k_axis(ax, "k_max (nominal sample budget, log scale)")
    ax.set_ylabel("Reduction vs. nominal budget (%)")
    ax.set_ylim(-5, 105)
    ax.set_title("Sample reduction from early stopping")
    ax.legend(loc="upper left")
    save(fig, "fig5_compute_savings.png", outputs)
    captions["fig5_compute_savings.png"] = (
        "1 − mean(k_used)/k_max. This is a reduction relative to the nominal budget, not relative to an "
        "experiment run without early stopping; k = 1 and k = 10 cannot stop early (minimum of 10 samples).")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for run in RUNS:
        rows = [s for s in summary if s["run"] == run]
        ks = [s["k"] for s in rows]
        ax1.plot(ks, [float(s["mean_confidence"]) for s in rows], marker=MARKERS[run], color=COLORS[run],
                 linewidth=2, label=RUNS[run]["label"])
        ax2.plot(ks, [float(s["mean_entropy_bits"]) for s in rows], marker=MARKERS[run], color=COLORS[run],
                 linewidth=2, label=RUNS[run]["label"])
    for ax in (ax1, ax2):
        log_k_axis(ax)
    ax1.set_ylabel("Mean vote confidence")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Confidence vs. k")
    ax2.set_ylabel("Mean answer entropy (bits)")
    ax2.set_title("Entropy vs. k")
    ax1.legend(loc="lower left", fontsize=9)
    save(fig, "fig6_confidence_entropy.png", outputs)
    captions["fig6_confidence_entropy.png"] = (
        "Mean vote confidence (share of the winning answer among valid answers) and mean Shannon entropy of "
        "the answer distribution. At k = 1 confidence is 1 and entropy 0 by construction.")

    fig, ax = plt.subplots()
    for run in RUNS:
        rows = [d for d in diag_summary if d["run"] == run and d["scope"].startswith("used at")]
        ks = [int(d["scope"].split("=")[1]) for d in rows]
        ax.plot(ks, [float(d["length_rate_pct"]) for d in rows], marker=MARKERS[run], color=COLORS[run],
                linewidth=2, label=RUNS[run]["label"])
    log_k_axis(ax)
    ax.set_ylabel("Responses with finish_reason = 'length' (%)")
    ax.set_ylim(bottom=0)
    ax.set_title("Truncated responses (16384-token limit)")
    ax.legend()
    save(fig, "fig7_truncation_rates.png", outputs)
    captions["fig7_truncation_rates.png"] = (
        "Share of the responses that enter the votes at each k which ended because they reached the "
        "16384-token limit, reconstructed from the response cache.")

    fig, ax = plt.subplots()
    styles = [("1 original", "original 2048-token run", ":", "v"),
              ("2 post-hoc", "post-hoc corrected scoring (2048-token responses)", "--", "D"),
              ("3 final", "final 16384-token run", "-", "s")]
    for prefix, label, line, marker in styles:
        rows = [h for h in hist_rows if h["run"] == "aime_41mini" and h["stage"].startswith(prefix[0])]
        if any(h["accuracy_pct"] == "unavailable" for h in rows):
            continue
        ax.plot([h["k"] for h in rows], [float(h["accuracy_pct"]) for h in rows], linestyle=line, marker=marker,
                color="tab:orange", linewidth=2, label=label)
    log_k_axis(ax)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("gpt-4.1-mini on AIME: three experimental stages")
    ax.legend(loc="lower right")
    save(fig, "fig8_gpt41mini_stages.png", outputs)
    captions["fig8_gpt41mini_stages.png"] = (
        "gpt-4.1-mini on the same 300 AIME problems in three separate stages: the original run (2048-token "
        "limit, original parser), the post-hoc rescoring of those same responses with the conservative parser "
        "(original early-stopping decisions kept), and the final 16384-token run. These are not equivalent "
        "runs and must not be pooled. The step from the corrected rescoring to the final run changes the token "
        "limit, the early-stopping implementation and the responses themselves at the same time, so the "
        "difference cannot be attributed to the token limit alone.")
    return outputs, captions


# ---------------------------------------------------------------- reports

def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def validation_report(findings, replayed, diag_counts, same_aime, hist_same):
    lines = ["# Data validation report", "",
             "Generated by `submission_analysis/analyze.py`. All checks are performed on the CSV files as they are "
             "on disk; nothing is modified. Network access is disabled while the script runs.", "",
             "## Per-file checks", ""]
    for f in findings:
        lines.append(f"### `{f['file']}`")
        lines.append("")
        for label, ok, detail in f["checks"]:
            lines.append(f"- {'PASS' if ok else 'FAIL'}: {label} {detail}".rstrip())
        lines.append("")
    lines += ["## Cross-experiment checks", "",
              f"- {'PASS' if same_aime else 'FAIL'}: both AIME experiments use exactly the same 300 problem ids in "
              "the same order at every k.",
              f"- {'PASS' if hist_same else 'FAIL'}: the historical 2048-token runs used the same 300 problems in "
              "the same order as the final runs.", "",
              "## Replay from the response cache", "",
              f"Every one of the {replayed} CSV rows (3 experiments × 4 k values × 300 problems) was recomputed by "
              "running the current `eval.py` and `parse.py` on the cached responses only (a missing response "
              "raises an error instead of calling the API). `k_used`, `valid_votes`, `invalid_votes`, "
              "`token_limit_votes`, `prediction`, `confidence`, `entropy`, `correct` and `completion_tokens` "
              "were reproduced exactly for every row. `runtime_seconds` is a wall-clock measurement and is not "
              "reproducible by design.", "",
              f"Cache entries read: {diag_counts['entries']} (one per generated response); missing: 0.", "",
              "## Notes that affect interpretation", "",
              "- `runtime_seconds` is measured per problem and per k-condition. Because all k-conditions share the "
              "same seeds, a later k-condition reads the responses drawn by an earlier one from the cache, so its "
              "runtime does not measure generation time. The gpt-4.1-mini run was also interrupted during k = 100 "
              "and resumed; on resumption k = 1 and k = 10 were read entirely from the cache. Runtimes should not "
              "be used as a measure of compute.",
              "- The model ids are the aliases sent with each request (`gpt-4.1-mini`, `gpt-4o-mini`). The exact "
              "model snapshot returned by the API was not recorded and is unavailable.",
              "- Problems without a valid final prediction are counted as wrong; their confidence and entropy are "
              "recorded as 0.", ""]
    return "\n".join(lines)


def table(rows, columns, headers=None):
    headers = headers or columns
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r[c]) for c in columns) + " |")
    return out


def tokens_at(tokens, run, k):
    return next(t for t in tokens if t["run"] == run and t["k"] == k)


def statistical_report(summary, gains, tests, cross, diag_summary, hist_rows, captions, tokens, years):
    lines = ["# Statistical analysis of the final 16k-token runs", "",
             "Generated by `submission_analysis/analyze.py` from the 12 CSV files of the three final runs "
             "(`experiments_aime_41mini_16k`, `experiments_aime_4o_16k`, `experiments_gsm8k_16k`) and the "
             "response cache. Every value below is computed by the script; nothing is entered by hand.", "",
             "## 1. Accuracy and sample use", "",
             "Accuracy counts a problem as correct only if its majority answer equals the gold answer after "
             "normalization; problems without a valid answer count as wrong. Wilson 95 % intervals are per point "
             "and do not reflect the pairing across k.", ""]
    lines += table([{**s, "label": RUNS[s["run"]]["label"]} for s in summary],
                   ["label", "k", "correct", "accuracy_pct", "wilson95_low_pct", "wilson95_high_pct",
                    "mean_k_used", "median_k_used", "samples_used", "compute_savings_vs_nominal_pct",
                    "mean_confidence", "mean_entropy_bits", "invalid_final_predictions"],
                   ["run", "k", "correct", "acc %", "CI low", "CI high", "mean k_used", "median k_used",
                    "samples", "reduction vs nominal %", "confidence", "entropy (bits)", "no valid prediction"])
    lines += ["", "`samples` is the number of responses that enter the votes in that k-condition. Because all "
              "k-conditions use the same seeds, the responses of a smaller k are the first responses of every "
              "larger k; the number of distinct responses generated per experiment equals the k = 1000 value.", "",
              "**Reduction vs. nominal budget** is 1 − mean(k_used)/k_max. It measures how far below the nominal "
              "budget early stopping ended, not the saving relative to an experiment run without early stopping "
              "(which was not run). At k = 1 and k = 10 no reduction is possible because at least 10 samples are "
              "always drawn.", "",
              "## 2. Marginal gains (percentage points)", ""]
    lines += table([{**g, "label": RUNS[g["run"]]["label"]} for g in gains],
                   ["label", "from_k", "to_k", "accuracy_from_pct", "accuracy_to_pct", "gain_pp"],
                   ["run", "from k", "to k", "acc from %", "acc to %", "gain (pp)"])
    lines += ["", "## 3. Paired significance tests across k (exact McNemar, two-sided)", "",
              "Within each experiment the three adjacent comparisons form one family and are Holm-corrected "
              "(m = 3). The k = 1 vs. k = 1000 comparison is shown for reference only and is not part of the "
              "corrected family.", ""]
    lines += table([{**t, "label": RUNS[t["run"]]["label"], "p": fmt_p(t["p_raw"]),
                     "ph": fmt_p(t["p_holm"]) if t["p_holm"] != "" else "–"} for t in tests],
                   ["label", "comparison", "accuracy_a_pct", "accuracy_b_pct", "wrong_to_right", "right_to_wrong",
                    "p", "ph", "significant_at_0.05"],
                   ["run", "comparison", "acc A %", "acc B %", "wrong→right", "right→wrong", "p (raw)",
                    "p (Holm)", "significant (α = 0.05)"])
    lines += ["", "## 4. gpt-4.1-mini vs. gpt-4o-mini on AIME (paired, same 300 problems)", "",
              "Exact McNemar test at each k; Holm correction across the four comparisons (m = 4).", ""]
    lines += table([{**c, "p": fmt_p(c["p_raw"]), "ph": fmt_p(c["p_holm"])} for c in cross],
                   ["k", "gpt41_mini_accuracy_pct", "gpt4o_mini_accuracy_pct", "difference_pp", "both_correct",
                    "only_gpt41_mini_correct", "only_gpt4o_mini_correct", "both_wrong", "p", "ph",
                    "significant_at_0.05"],
                   ["k", "4.1-mini %", "4o-mini %", "diff (pp)", "both right", "only 4.1-mini", "only 4o-mini",
                    "both wrong", "p (raw)", "p (Holm)", "significant"])
    t41, t4o = tokens_at(tokens, "aime_41mini", 1000), tokens_at(tokens, "aime_4o", 1000)
    lines += ["", "**What this comparison holds constant and what it does not.** Both models were run on the same "
              "300 problems with the same prompt, temperature, token limit, nominal budgets k and stopping policy. "
              "They did not, however, use the same number of samples or the same amount of computation: early "
              "stopping depends on each model's own answer agreement, and the models write responses of very "
              f"different length. At k = 1000, gpt-4.1-mini used {t41['responses']} responses "
              f"({t41['total_tokens'] / 1e6:.1f} M API tokens, {t41['mean_completion_tokens_per_response']} "
              f"completion tokens per response) and gpt-4o-mini used {t4o['responses']} responses "
              f"({t4o['total_tokens'] / 1e6:.1f} M API tokens, {t4o['mean_completion_tokens_per_response']} "
              "completion tokens per response). The comparison is therefore a comparison at equal nominal budget "
              "and policy, not at equal samples, tokens or compute (the model sizes are not disclosed). "
              "See section 7.", "",
              "## 5. Response-level diagnostics (reconstructed from the cache)", "",
              "Each response was located with the exact cache key of `utils.py` and the seed of "
              "`eval._derive_seed`, and re-parsed with `parse.extract_answer`. `all generated` counts every "
              "distinct response of an experiment; `used at k` counts the responses entering the votes of that "
              "k-condition (overlapping sets).", ""]
    lines += table([{**d, "label": RUNS[d["run"]]["label"]} for d in diag_summary],
                   ["label", "scope", "responses", "valid_answers", "validity_rate_pct", "finish_reason_stop",
                    "finish_reason_length", "finish_reason_other", "length_rate_pct", "truncated_with_valid_answer",
                    "mean_completion_tokens", "median_completion_tokens", "max_completion_tokens"],
                   ["run", "scope", "responses", "valid", "valid %", "stop", "length", "other", "length %",
                    "truncated but valid", "mean tokens", "median tokens", "max tokens"])
    disagreements = sum(d["live_and_offline_rules_disagree"] for d in diag_summary if d["scope"] == "all generated")
    lines += ["", f"Live vs. offline truncation rule: the live code marks a response as truncated when "
              "`finish_reason == 'length'`; the offline rescoring script (`scripts/replay_corrected.py`) uses "
              "`completion_tokens >= max_tokens` because the original 2048-token cache entries have no "
              f"`finish_reason`. On the final runs the two rules give a different truncation flag or a different "
              f"extracted answer for {disagreements} response(s). Both use the same `parse.extract_answer`.", "",
              "## 6. Historical stages (kept separate)", "",
              "Three distinct stages exist for every experiment. They differ in token limit, parser and stopping "
              "rule and must not be pooled or presented as replications.", "",
              "1. **Original run**: max_tokens 2048, original parser (with last-integer fallback), original "
              "early-stopping rule. Files: `experiments/`, `experiments_gsm8k/`, `experiments_aime_41mini/`.",
              "2. **Post-hoc corrected rescoring**: the same 2048-token responses, re-scored with the conservative "
              "parser while keeping the original `k_used`; no new responses. Files: `corrected_results/`.",
              "3. **Final run**: max_tokens 16384, conservative parser, coverage-aware early stopping (invalid "
              "responses cannot trigger stopping; at least 10 valid answers required). New responses. Files: "
              "`experiments_*_16k/`.", ""]
    lines += table([{**h, "label": RUNS[h["run"]]["label"]} for h in hist_rows],
                   ["label", "k", "stage", "accuracy_pct", "mean_k_used"],
                   ["run", "k", "stage", "accuracy %", "mean k_used"])
    lines += ["", "**Which variables change between stages.** Stage 1 → 2 changes only the parser (same responses, "
              "same `k_used`), so that difference is attributable to answer extraction. Stage 2 → 3 changes "
              "several things at once: the token limit (2048 → 16384), the early-stopping implementation "
              "(invalid responses no longer count towards confidence or stability, and at least 10 valid answers "
              "are required) and the responses themselves (new samples, so sampling variability also enters). "
              "The difference between stage 2 and stage 3, and a fortiori between stage 1 and stage 3, must "
              "therefore not be attributed to the token limit alone. No experiment varies only the token limit.",
              ""]

    lines += ["## 7. Nominal budget, samples, tokens and compute", "",
              "Four quantities are easily confused and are kept apart in this analysis:", "",
              "- **Nominal sampling budget k** (`k_max`): the maximum number of samples per problem that the "
              "policy may draw (1, 10, 100, 1000). It is the experimental condition and the x-axis of the "
              "figures, but not the amount actually spent.",
              "- **Actual samples k_used**: the number of responses actually drawn for a problem before early "
              "stopping ended sampling (`k_used` in the CSVs; at least 10 for k ≥ 10). Its mean is reported in "
              "section 1; at k = 100 and 1000 it is far below k.",
              "- **API token consumption**: prompt + completion tokens of the used responses, as reported by the "
              "API (`token_usage.csv`, below). It depends on k_used and on response length, which differs strongly "
              "between models and datasets. Prompt tokens are charged for every sample because each sample is a "
              "separate request.",
              "- **Actual computational cost**: the floating-point operations (FLOPs) or energy spent on the "
              "provider's hardware. It is **not measured** in this study and cannot be derived from token counts: "
              "parameter counts and serving details of gpt-4.1-mini and gpt-4o-mini are not public, so equal "
              "token counts of the two models do not imply equal compute. `runtime_seconds` is not a substitute "
              "either (see the validation report).", "",
              "Token usage of the responses that enter the votes in each k-condition (a response is counted at k "
              "if its `call_index` is smaller than the problem's `k_used` at that k; prompt and completion tokens "
              "are taken from `response_level.csv`, which was reconstructed from the response cache). The growth "
              "columns compare with the previous k-condition. Per problem, the completion-token sums match the "
              "`completion_tokens` column of the result CSVs exactly.", ""]
    lines += table([{**t, "label": RUNS[t["run"]]["label"],
                     "added": t["total_tokens_added_vs_previous_k"] if t["previous_k"] != "" else "–",
                     "growth": t["total_tokens_growth_factor_vs_previous_k"] or "–",
                     "rgrowth": t["responses_growth_factor_vs_previous_k"] or "–"} for t in tokens],
                   ["label", "k", "responses", "total_prompt_tokens", "total_completion_tokens", "total_tokens",
                    "mean_total_tokens_per_problem", "mean_completion_tokens_per_response", "rgrowth", "added",
                    "growth"],
                   ["run", "k", "responses", "prompt tokens", "completion tokens", "total tokens",
                    "tokens per problem", "completion tokens per response", "responses × vs prev. k",
                    "tokens added vs prev. k", "tokens × vs prev. k"])

    lines += ["", "## 8. How sampling was executed", "",
              "For each individual problem the samples are generated **sequentially**, one request after another "
              "in `call_index` order (`for call_idx in range(k_max)` in `eval.py`). After every response the vote "
              "is updated and the stopping rules are checked, which is what makes early stopping possible: sample "
              "c + 1 is only requested if the first c samples did not satisfy a stopping rule.", "",
              "Parallelism is across problems, not within a problem: `ThreadPoolExecutor` (64 / 48 / 32 workers "
              "for the three runs) runs the sampling loops of different problems concurrently. The worker count "
              "affects only wall-clock time; replaying every row with a single worker reproduces the results "
              "exactly. In this study, \"parallel sampling\" denotes the method — independent samples of the same "
              "prompt aggregated by majority vote — not concurrent requests for one problem.", ""]

    periods = [(1983, 1999), (2000, 2009), (2010, 2019), (2020, 2024)]
    lines += ["## 9. Year distribution of the AIME subset and possible contamination", "",
              "The 300 AIME problems were drawn from the 933 problems of `data/aime.jsonl` with `subset_seed = 123` "
              "(`main.select_subset`); the year is taken from the problem id. The same subset was used for both "
              "models and for all three historical stages.", ""]
    lines += table([{"period": f"{a}–{b}",
                     "sel": sum(y["selected_problems"] for y in years if a <= y["year"] <= b),
                     "pool": sum(y["problems_in_dataset"] for y in years if a <= y["year"] <= b)} for a, b in periods],
                   ["period", "sel", "pool"], ["period", "selected problems", "problems in dataset"])
    lines += ["", "Per year (full table in `aime_year_distribution.csv`):", ""]
    lines += table(years, ["year", "selected_problems", "problems_in_dataset", "selected_pct"],
                   ["year", "selected", "in dataset", "selected %"])
    lines += ["", "**Possible training-data contamination (unresolved).** All selected problems, together with "
              "their answers and many worked solutions, were publicly available on the internet before either "
              "model was released (gpt-4o-mini in July 2024, gpt-4.1-mini in April 2025). It is therefore "
              "possible that some problems or solutions were part of the models' training data, which could "
              "inflate absolute accuracy, especially for older, widely discussed problems. This study contains "
              "no test for contamination and the providers do not publish their training data, so contamination "
              "is neither established nor excluded. Accuracy differences between competition years would not "
              "settle the question either, because problem difficulty and style also changed over the years. "
              "The within-model comparisons across k are less affected than absolute accuracies, since every k "
              "uses the same problems, but a memorised answer could still make agreement among samples, and "
              "therefore early stopping, more likely.", "",
              "## 10. Figures", ""]
    for name, caption in captions.items():
        lines.append(f"- `{name}`: {caption}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- main

def main():
    findings = []
    problems, rows, summary, gains, tests, diag_summary, hist_rows = {}, {}, [], [], [], [], []
    response_rows = []
    replayed = 0
    hist_same = True

    for run, cfg in RUNS.items():
        print(f"{run}: validating", flush=True)
        problems[run] = select_subset(load_jsonl_dataset(REPO / cfg["dataset"]), SETTINGS["subset_size"],
                                      SETTINGS["subset_seed"])
        rows[run] = validate(run, cfg, problems[run], findings)

    same_aime = all([r["example_id"] for r in rows["aime_41mini"][k]] == [r["example_id"] for r in rows["aime_4o"][k]]
                    for k in K_VALUES)
    if not same_aime:
        raise ValidationError("the two AIME experiments do not use the same problems in the same order")

    for run, cfg in RUNS.items():
        print(f"{run}: replaying from the cache", flush=True)
        replayed += replay_from_cache(run, cfg, problems[run], rows[run])
        print(f"{run}: response diagnostics", flush=True)
        per_response = response_diagnostics(run, cfg, problems[run], rows[run])
        response_rows += per_response
        diag_summary += summarize_responses(run, per_response, rows[run])
        run_summary = describe(run, rows[run])
        summary += run_summary
        gains += marginal(run, run_summary)
        tests += adjacent_tests(run, rows[run])
        h, same = historical(run, cfg, rows[run])
        hist_rows += h
        hist_same = hist_same and same

    cross = cross_model(rows["aime_41mini"], rows["aime_4o"])

    full_run_start = datetime.fromisoformat(WALL_CLOCK["aime_41mini"][0]["started_utc"].replace("Z", "+00:00"))
    reused_from_pilot = sum(r["run"] == "aime_41mini" and r["_written_utc"] < full_run_start.timestamp()
                            for r in response_rows)

    print("writing outputs", flush=True)
    written = []

    def write(name, text):
        path = HERE / name
        path.write_text(text, encoding="utf-8", newline="\n")
        written.append(path)

    write("summary.csv", csv_text(summary, list(summary[0])))
    write("marginal_gains.csv", csv_text(gains, list(gains[0])))
    write("mcnemar_tests.csv", csv_text(
        [{**t, "p_raw": f"{t['p_raw']:.6g}", "p_holm": f"{t['p_holm']:.6g}" if t["p_holm"] != "" else ""}
         for t in tests], list(tests[0])))
    write("cross_model_comparison.csv", csv_text(
        [{**c, "p_raw": f"{c['p_raw']:.6g}", "p_holm": f"{c['p_holm']:.6g}"} for c in cross], list(cross[0])))
    write("response_diagnostics.csv", csv_text(diag_summary, list(diag_summary[0])))
    public = [c for c in response_rows[0] if not c.startswith("_")]
    write("response_level.csv", csv_text([{c: r[c] for c in public} for r in response_rows], public))
    write("historical_comparison.csv", csv_text(hist_rows, list(hist_rows[0])))

    tokens = token_usage(HERE / "response_level.csv", rows)
    write("token_usage.csv", csv_text(tokens, list(tokens[0])))
    years = aime_years(problems["aime_41mini"])
    write("aime_year_distribution.csv", csv_text(years, list(years[0])))

    figure_paths, captions = figures(summary, gains, diag_summary, hist_rows)
    token_figure(tokens, figure_paths, captions)
    written += figure_paths

    code_files = ["main.py", "eval.py", "parse.py", "utils.py", "prompting.py", "data.py",
                  "scripts/replay_corrected.py", "submission_analysis/analyze.py"]
    config = {
        "description": "Actual configuration of the three final runs. Settings come from the command lines and "
                       "the configuration printed at the start of each run; they are confirmed by replaying every "
                       "CSV row from the cache with exactly these settings.",
        "common_settings": SETTINGS,
        "runs": {run: {"model_id_requested": cfg["model"], "model_snapshot": "unavailable (not recorded)",
                       "dataset": cfg["dataset"], "dataset_sha256": sha256(REPO / cfg["dataset"]),
                       "dataset_size": len(load_jsonl_dataset(REPO / cfg["dataset"])),
                       "problems": N_PROBLEMS, "first_problem": problems[run][0]["id"],
                       "last_problem": problems[run][-1]["id"], "results_dir": cfg["dir"],
                       "max_workers": cfg["max_workers"], "wall_clock_sessions": WALL_CLOCK[run]}
                 for run, cfg in RUNS.items()},
        "differences_between_runs": [
            "model: gpt-4.1-mini (aime_41mini) vs. gpt-4o-mini (aime_4o, gsm8k_4o)",
            "dataset: AIME (aime_41mini, aime_4o) vs. GSM8K (gsm8k_4o)",
            "max_workers: 64 / 48 / 32 (affects speed only; results do not depend on it, as the replay with "
            "max_workers=1 reproduces every row)",
            "aime_41mini was interrupted during k=100 and resumed; the replay shows the results are identical to "
            "an uninterrupted run because every response has a fixed seed and is cached",
            f"aime_41mini reused {reused_from_pilot} responses that had been drawn with identical requests "
            "before the full run started (30-problem pilot and a 30-response spread sample; same code and "
            "settings), counted from cache-file timestamps",
        ],
        "request_timeout_s": 600,
        "code_sha256": {f: sha256(REPO / f) for f in code_files},
        "parser_sha256": sha256(REPO / "parse.py"),
        "git_head": git("rev-parse", "HEAD"),
        "git_working_tree_note": "The experiment code was not committed when the runs were made; the SHA-256 "
                                 "fingerprints above identify the exact files. The replay verifies that these "
                                 "files reproduce every CSV row.",
    }
    write("configuration_summary.json", json.dumps(config, indent=2, ensure_ascii=False) + "\n")

    write("data_validation_report.md", validation_report(findings, replayed, {"entries": len(response_rows)},
                                                         same_aime, hist_same))
    write("statistical_analysis_report.md",
          statistical_report(summary, gains, tests, cross, diag_summary, hist_rows, captions, tokens, years))

    inputs = sorted({REPO / cfg["dir"] / f"k_{k}.csv" for cfg in RUNS.values() for k in K_VALUES}
                    | {historical_path(cfg[key]) / f"k_{k}.csv" for cfg in RUNS.values()
                       for key in ("historical_original", "historical_corrected") for k in K_VALUES}
                    | {REPO / cfg["dataset"] for cfg in RUNS.values()})
    manifest = {
        "inputs": {p.relative_to(REPO).as_posix(): sha256(p) for p in inputs},
        "outputs": {p.relative_to(REPO).as_posix(): sha256(p) for p in sorted(written)},
        "code": config["code_sha256"],
    }
    (HERE / "sha256_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
                                               newline="\n")
    print(f"done: {replayed} rows replayed, {len(response_rows)} responses checked, "
          f"{len(written) + 1} files written to {HERE}")


if __name__ == "__main__":
    main()
