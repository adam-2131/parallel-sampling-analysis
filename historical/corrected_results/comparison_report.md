# Corrected rescoring of the original experiments

**This is post-hoc rescoring, not a new run.** For every problem and k the original
`k_used` is kept, so each vote uses exactly the responses the original run drew, in the
original order. Only answer extraction, normalization and scoring were changed
(`parse.py`). The early-stopping decisions are the original ones and were
made with the original extractor; the corrected stopping rule was **not** rerun.

No API calls: network sockets were disabled, and all 50496 cache entries
used were read directly and verified. Every original CSV row was first reproduced exactly
from the cache with the original extractor and stopping rule.

## Accuracy

Accuracy is over all 300 problems; a problem without a valid prediction counts as wrong.
The last column restricts accuracy to problems that have a valid prediction. That subset
differs between k and between runs, so it describes the data and is not a comparable score.

| Run | k | Original | Corrected | Avg. k_used | Problems without valid prediction | Corrected, among valid predictions |
|---|---:|---:|---:|---:|---:|---:|
| aime_4o | 1 | 21.33 % | 21.67 % | 1.00 | 7 | 22.18 % |
| aime_4o | 10 | 26.33 % | 26.33 % | 10.00 | 0 | 26.33 % |
| aime_4o | 100 | 27.00 % | 27.33 % | 67.61 | 0 | 27.33 % |
| aime_4o | 1000 | 26.67 % | 27.00 % | 74.43 | 0 | 27.00 % |
| gsm8k_4o | 1 | 91.33 % | 95.67 % | 1.00 | 0 | 95.67 % |
| gsm8k_4o | 10 | 93.33 % | 96.67 % | 10.00 | 0 | 96.67 % |
| gsm8k_4o | 100 | 93.67 % | 97.00 % | 18.14 | 0 | 97.00 % |
| gsm8k_4o | 1000 | 93.67 % | 97.00 % | 18.41 | 0 | 97.00 % |
| aime_41 | 1 | 24.67 % | 38.67 % | 1.00 | 172 | 90.62 % |
| aime_41 | 10 | 31.00 % | 50.00 % | 10.00 | 116 | 81.52 % |
| aime_41 | 100 | 35.67 % | 57.67 % | 68.38 | 75 | 76.89 % |
| aime_41 | 1000 | 36.00 % | 58.33 % | 75.48 | 74 | 77.43 % |

## Confidence and entropy

| Run | k | Confidence original | Confidence corrected | Entropy original (bits) | Entropy corrected (bits) |
|---|---:|---:|---:|---:|---:|
| aime_4o | 1 | 1.000 | 0.977 | 0.000 | 0.000 |
| aime_4o | 10 | 0.427 | 0.441 | 2.172 | 2.101 |
| aime_4o | 100 | 0.392 | 0.404 | 3.271 | 3.145 |
| aime_4o | 1000 | 0.393 | 0.405 | 3.294 | 3.167 |
| gsm8k_4o | 1 | 1.000 | 1.000 | 0.000 | 0.000 |
| gsm8k_4o | 10 | 0.951 | 0.963 | 0.167 | 0.124 |
| gsm8k_4o | 100 | 0.957 | 0.967 | 0.166 | 0.125 |
| gsm8k_4o | 1000 | 0.957 | 0.967 | 0.166 | 0.125 |
| aime_41 | 1 | 1.000 | 0.427 | 0.000 | 0.000 |
| aime_41 | 10 | 0.420 | 0.574 | 2.137 | 0.120 |
| aime_41 | 100 | 0.372 | 0.672 | 3.026 | 0.306 |
| aime_41 | 1000 | 0.373 | 0.676 | 3.038 | 0.309 |

## Invalid and token-limit responses

Counts are over the responses that enter the votes (the first `k_used` responses of each problem).

| Run | k | Responses | Invalid (original) | Invalid (corrected) | Token-limit | Token-limit rate | Token-limit with valid answer |
|---|---:|---:|---:|---:|---:|---:|---:|
| aime_4o | 1 | 300 | 0 | 7 | 6 | 2.00 % | 0 |
| aime_4o | 10 | 3000 | 0 | 47 | 37 | 1.23 % | 0 |
| aime_4o | 100 | 20284 | 0 | 326 | 250 | 1.23 % | 1 |
| aime_4o | 1000 | 22329 | 0 | 375 | 285 | 1.28 % | 1 |
| gsm8k_4o | 1 | 300 | 0 | 0 | 0 | 0.00 % | 0 |
| gsm8k_4o | 10 | 3000 | 0 | 0 | 0 | 0.00 % | 0 |
| gsm8k_4o | 100 | 5441 | 0 | 5 | 5 | 0.09 % | 0 |
| gsm8k_4o | 1000 | 5522 | 0 | 5 | 5 | 0.09 % | 0 |
| aime_41 | 1 | 300 | 0 | 172 | 172 | 57.33 % | 0 |
| aime_41 | 10 | 3000 | 0 | 1729 | 1733 | 57.77 % | 4 |
| aime_41 | 100 | 20515 | 0 | 12809 | 12838 | 62.58 % | 30 |
| aime_41 | 1000 | 22645 | 0 | 14453 | 14483 | 63.96 % | 32 |

## McNemar tests (paired, exact, two-sided)

| Run | Comparison | Scoring | Wrong → right | Right → wrong | p |
|---|---|---|---:|---:|---:|
| aime_4o | k=1 vs k=10 | original | 22 | 7 | 0.0081 |
| aime_4o | k=1 vs k=10 | corrected | 22 | 8 | 0.0161 |
| aime_4o | k=10 vs k=100 | original | 6 | 4 | 0.7539 |
| aime_4o | k=10 vs k=100 | corrected | 6 | 3 | 0.5078 |
| aime_4o | k=100 vs k=1000 | original | 1 | 2 | 1.0000 |
| aime_4o | k=100 vs k=1000 | corrected | 1 | 2 | 1.0000 |
| gsm8k_4o | k=1 vs k=10 | original | 7 | 1 | 0.0703 |
| gsm8k_4o | k=1 vs k=10 | corrected | 4 | 1 | 0.3750 |
| gsm8k_4o | k=10 vs k=100 | original | 2 | 1 | 1.0000 |
| gsm8k_4o | k=10 vs k=100 | corrected | 1 | 0 | 1.0000 |
| gsm8k_4o | k=100 vs k=1000 | original | 0 | 0 | 1.0000 |
| gsm8k_4o | k=100 vs k=1000 | corrected | 0 | 0 | 1.0000 |
| aime_41 | k=1 vs k=10 | original | 31 | 12 | 0.0054 |
| aime_41 | k=1 vs k=10 | corrected | 35 | 1 | 0.0000 |
| aime_41 | k=10 vs k=100 | original | 20 | 6 | 0.0094 |
| aime_41 | k=10 vs k=100 | corrected | 23 | 0 | 0.0000 |
| aime_41 | k=100 vs k=1000 | original | 1 | 0 | 1.0000 |
| aime_41 | k=100 vs k=1000 | corrected | 2 | 0 | 0.5000 |

## Scoring rules

- An answer counts as correct only if its canonical form equals the canonical gold answer.
- Percent signs, currency signs, degree signs and unit words are kept, so `50%` is not `50`.
  `unit_only_mismatches` in `comparison.csv` counts final predictions that differ from the gold
  answer only by such a marker; they are scored as wrong and are reported, not reinterpreted.
- Responses that hit the 2048-token limit keep an answer only if they contain a complete,
  explicit `Final Answer:`; they are listed individually in `token_limit_responses.csv`.
- Ties in the plurality vote are broken as in the original run (smallest string).
