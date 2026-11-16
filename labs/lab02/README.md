# Lab 02 — Training duration and random-seed variation

Status: **reported**. Experiment script `exp02.py` and complete raw CSVs have not yet been synced.

A training: 3 epochs. B: 3 or 5 epochs. Seeds: 42–46. 20 runs (2 durations × 2 methods × 5 seeds). Model: 784→256→256→10. Replay buffer: 1000 A examples, ~43 appended A samples per B batch of 128 (25% of combined batch).

| B epochs | Method | A forgetting (pp, mean±SD) | B accuracy (%, mean±SD) |
|---|---|---|---|
| 3 | Naive | 9.38 ± 4.25 | 94.76 ± 0.95 |
| 3 | Replay | 0.22 ± 0.56 | 94.52 ± 0.77 |
| 5 | Naive | 9.90 ± 4.82 | 95.60 ± 0.50 |
| 5 | Replay | 0.16 ± 0.50 | 95.53 ± 0.57 |

Source: user-provided `exp02-report.html`, with charts relying on external assets not present in the artifact. No claim of zero-cost Replay or of learning duration having no effect. Full raw data and source should be added in a future commit.
