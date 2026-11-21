# Statistical analysis package – final 16k-token runs

> **Scope of this repository.** It contains only the analysis outputs (reports, tables, figures, manifest)
> and the analysis script. The experiment code, the per-problem result CSVs and the response cache are not
> included, so `analyze.py` cannot be run from this repository alone; the paths in the "Reproducing" section
> refer to the full project folder.

This folder contains the complete offline analysis of the three final experiments of the Maturaarbeit
"Test-Time Compute Scaling through Parallel Sampling – A Controlled Empirical Study of Saturation in
Self-Consistency". Every number in it is produced by `analyze.py`; nothing was entered by hand.

The analysis made no API calls (the script disables network sockets before importing anything), did not
rerun any experiment and did not modify any experimental CSV file.

## Experiments analysed

| run id | model | dataset | results directory |
|---|---|---|---|
| `aime_41mini` | gpt-4.1-mini | AIME, 300 problems | `experiments_aime_41mini_16k/` |
| `aime_4o` | gpt-4o-mini | AIME, the same 300 problems | `experiments_aime_4o_16k/` |
| `gsm8k_4o` | gpt-4o-mini | GSM8K test, 300 problems | `experiments_gsm8k_16k/` |

Common settings: temperature 0.7, max_tokens 16384, k ∈ {1, 10, 100, 1000}, seed 0, subset of 300 drawn with
subset seed 123, early stopping at confidence ≥ 0.95 or 50 consecutive identical majorities, at least 10
samples and at least 10 valid answers before stopping. Full details: `configuration_summary.json`.

## Files

| file | content |
|---|---|
| `analyze.py` | the analysis script (validation, statistics, cache replay, diagnostics, figures, manifest) |
| `data_validation_report.md` | every data check with PASS/FAIL, the cache replay result and interpretation notes |
| `statistical_analysis_report.md` | all tables in readable form, figure captions, historical stages |
| `summary.csv` | accuracy, Wilson 95 % CI, k_used, samples, reduction vs. nominal budget, confidence, entropy, runtime |
| `marginal_gains.csv` | accuracy change in percentage points for 1→10, 10→100, 100→1000 and 1→1000 |
| `mcnemar_tests.csv` | exact two-sided McNemar tests between k-conditions, Holm-corrected within each experiment (m = 3) |
| `cross_model_comparison.csv` | gpt-4.1-mini vs. gpt-4o-mini on AIME at each k, Holm-corrected across the four k (m = 4) |
| `response_diagnostics.csv` | response-level counts (valid, invalid, finish_reason, truncation, tokens) per run and k |
| `response_level.csv` | one line per generated response (39 841 lines): seed, tokens, finish_reason, extracted answer |
| `token_usage.csv` | prompt, completion and total API tokens of the responses used at each k, with growth vs. the previous k |
| `aime_year_distribution.csv` | competition years of the 300 selected AIME problems and of the full dataset |
| `historical_comparison.csv` | accuracy and k_used of the three historical stages, kept separate |
| `configuration_summary.json` | actual run configuration, wall-clock sessions, code fingerprints, differences between runs |
| `fig1`–`fig9` `*.png` | figures; captions are in `statistical_analysis_report.md`, section 10 |
| `sha256_manifest.json` | SHA-256 of every input file, output file and code file |

## Methods in brief

- **Accuracy**: a problem is correct if its plurality-vote answer equals the gold answer after normalization.
  Problems without any valid answer count as wrong.
- **Marginal gain**: accuracy(k_b) − accuracy(k_a) in percentage points.
- **Reduction vs. nominal budget**: 1 − mean(k_used)/k_max. This is relative to the nominal budget k_max,
  not to an experiment without early stopping (no such experiment was run).
- **McNemar**: exact two-sided binomial test (p = 0.5) on the discordant pairs (wrong→right vs. right→wrong).
  Holm–Bonferroni correction within each family. The k = 1 vs. k = 1000 test is reported for reference
  only and is not part of the corrected family.
- **Response-level diagnostics** are not inferred from the problem-level CSVs. Each response is located in
  the response cache with the exact cache key of `utils.py` (model, temperature, max_tokens, prompt, seed)
  and the seed of `eval._derive_seed`, and re-parsed with `parse.extract_answer`.
- **Budget, samples, tokens, compute**: the nominal budget k (`k_max`) is the experimental condition;
  `k_used` is the number of samples actually drawn; API tokens (`token_usage.csv`) are prompt + completion
  tokens of the used responses; actual computational cost (FLOPs, energy) is not measured and cannot be
  derived from tokens, because the model sizes are not public. A response counts as used at k if its
  `call_index` is smaller than the problem's `k_used` at that k.
- **Execution**: the samples of one problem are drawn sequentially (needed for early stopping);
  `ThreadPoolExecutor` parallelises different problems, not the samples of one problem.
- **Cross-model comparison**: same problems, nominal budgets and stopping policy, but different actual
  sample counts and token consumption; it is not a comparison at equal compute.
- **Replay**: every one of the 3 600 CSV rows is recomputed by running the unchanged `eval.py` on the cached
  responses only (any missing response raises an error instead of calling the API). All fields except the
  wall-clock `runtime_seconds` must match exactly.

## Historical stages

Three stages exist and must not be pooled or presented as replications:

1. **Original runs**: max_tokens 2048, original parser (with last-integer fallback), original early stopping.
2. **Post-hoc corrected rescoring**: the same 2048-token responses re-scored with the conservative parser,
   original `k_used` kept; no new responses.
3. **Final runs**: max_tokens 16384, conservative parser, coverage-aware early stopping, new responses.

Stage 1 → 2 changes only the parser. Stage 2 → 3 changes the token limit, the early-stopping implementation
and the responses at the same time, so the difference must not be attributed to the token limit alone.

Only stage 3 is analysed statistically. Stages 1 and 2 appear only in `historical_comparison.csv` and
`fig8_gpt41mini_stages.png`.

## Reproducing

From the repository root (or the root of the unzipped bundle), with Python ≥ 3.10 and `matplotlib` installed:

```
python submission_analysis/analyze.py
```

The script needs the code files (`main.py`, `eval.py`, `parse.py`, `utils.py`, `prompting.py`, `data.py`),
`data/aime.jsonl`, `data/gsm8k.jsonl`, the three final result directories and the historical results (in the
repository root or under `historical/`).

The cache replay and the response diagnostics additionally need the response cache (`.cache/`, about
40 000 JSON files containing the full model outputs). It is read from `.cache/` next to `main.py`, or from the
directory given in the environment variable `RESPONSE_CACHE_DIR`; it is only read, never written. The cache
is **not** included in the bundle because of its size; without it the script stops at the replay step. `response_level.csv` contains the per-response
information extracted from the cache, so the response-level tables can be checked from it without the cache.
Run time with the cache: about 7 minutes.

## Information that is not available

- The exact model snapshot behind the aliases `gpt-4.1-mini` and `gpt-4o-mini` was not recorded.
- `runtime_seconds` does not measure generation time: later k-conditions read earlier responses from the
  cache, and the gpt-4.1-mini run was interrupted during k = 100 and resumed. Wall-clock times of the whole
  runs are in `configuration_summary.json`, taken from the console logs.
- No run without early stopping exists, so the true accuracy at a full k = 100 or k = 1000 budget is unknown;
  only the accuracy under the early-stopping rule is measured.
- Token counts are available (`token_usage.csv`), but the billed cost and the actual computational cost
  (FLOPs) are not part of this analysis.
- Possible training-data contamination of the AIME problems (1983–2024, all public before both models were
  released) is unresolved: it is neither established nor excluded. The year distribution of the subset is in
  `aime_year_distribution.csv` and section 9 of the statistical report.
- The experiment code was not committed when the runs were made (git HEAD `52a9171`); the exact files are
  identified by the SHA-256 fingerprints in `configuration_summary.json`.
