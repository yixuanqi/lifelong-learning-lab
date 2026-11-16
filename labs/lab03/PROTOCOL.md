# Lab 03 — Long-term loss of plasticity

Status: **in progress**; no results have been received.

**Question**: Does ability to learn *new* tasks degrade after many sequential tasks, as opposed to simply forgetting old ones?

- Tasks: distinct fixed pixel permutations of MNIST; begin with 30 for pipeline checks, extend to 100.
- Continual: one persistent network across tasks. Scratch: fresh network per task, with matched data, optimizer steps and evaluation.
- Hold test samples out; do not provide task ID at inference or future-task information during training.
- At least three seeds; per-task learning curves, not just end-of-task accuracy.
- Log layerwise dormant ReLU fractions, representation effective ranks, weight statistics, seeds, wall time and compute budget.
- Start **without** Replay/EWC/Continual BP, to diagnose plasticity before testing interventions.

Paper's Online Permuted MNIST uses up to 800 tasks, single-pass online updates. Smaller task counts or mini-batching are a *scaled-down study*, not strict replication.

Acceptance: code + no-network test, raw CSV, config, matched Scratch vs Continual curves, diagnostics, dispersion and limitations.
