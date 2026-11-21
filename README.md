# Parallel sampling – code, final results and analysis

Code, final experimental results and offline statistical analysis for the Maturaarbeit
*Test-Time Compute Scaling through Parallel Sampling – A Controlled Empirical Study of Saturation in
Self-Consistency* (Adam Zaremba, 2026). The contents are the same as the submitted analysis bundle.

Layout (`python submission_analysis/analyze.py` runs from the repository root):

- `submission_analysis/` – analysis script, reports, CSV tables, figures, SHA-256 manifest. Start with
  `submission_analysis/README.md`, then `data_validation_report.md` and `statistical_analysis_report.md`.
- `experiments_aime_41mini_16k/`, `experiments_aime_4o_16k/`, `experiments_gsm8k_16k/` – the three FINAL
  runs (max_tokens 16384). Only these are analysed statistically.
- `historical/` – EARLIER STAGES, kept for reference only; not comparable with the final runs:
  - `experiments/` (AIME gpt-4o-mini), `experiments_gsm8k/` (GSM8K gpt-4o-mini),
    `experiments_aime_41mini/` (AIME gpt-4.1-mini): original runs, max_tokens 2048, original parser.
  - `corrected_results/`: post-hoc rescoring of the 2048-token responses with the conservative parser.
  - `experiments_pilot_41mini_16k/`: 30-problem pilot and spread sample used to estimate the cost of the
    final gpt-4.1-mini run (same settings as the final run).
- `main.py`, `eval.py`, `parse.py`, `utils.py`, `prompting.py`, `data.py`, `scripts/`, `tests/`,
  `requirements.txt` – the code exactly as used for the final runs (SHA-256 fingerprints in
  `submission_analysis/configuration_summary.json`).
- `thesis_figures/` – scripts for the figures in sections 5.2–5.4 of the thesis; they read
  `submission_analysis/` and `experiments_aime_4o_16k/`.
- `data/` – the AIME and GSM8K problem files.

## Running

```bash
pip install -r requirements.txt
python submission_analysis/analyze.py   # offline analysis (needs the response cache, see below)
python -m pytest tests                  # automated tests, no API key needed
```

The commands of the three final runs are in Appendix D of the thesis. Running them again requires an
OpenAI API key; the API does not guarantee identical responses at a later date.

Not included: the response cache (`.cache/`, ~40 000 files of raw model outputs), credentials and
environment files. Without the cache, the replay step of `analyze.py` cannot run; `response_level.csv`
contains the per-response information extracted from it.
