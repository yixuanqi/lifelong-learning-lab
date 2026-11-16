# Lab 01 — Catastrophic forgetting in sequential Permuted MNIST

Same 10-class MLP learns A (original MNIST) then B (a fixed permutation of the same 784 image pixels). Naive trains B alone; Replay mixes old A samples into B batches, with extra compute.

## Run on Windows

```powershell
cd labs/lab01
python -c "import torch, torchvision; print(torch.cuda.is_available())"
python -m pip install matplotlib
python smoke_test.py
python train.py --device auto
```

First run downloads MNIST. Reuse an existing compatible CUDA torch/torchvision installation rather than overwriting it. CPU: `python train.py --device cpu`.

Exports history.csv, summary.csv, config.json, accuracy.png under results/.
Forgetting is A_after_A minus A_after_B, in percentage points (pp).

Original seed42: Naive A 93.63→90.10% (3.53pp); Replay 93.63→93.70% (-0.07pp). Single-seed observations are not general evidence. Task B changes only input ordering and shares labels, so no direct inference about real robots or LLMs.
