#!/usr/bin/env python3
"""Lab01: sequential Permuted MNIST, Naive vs Replay. Single shared 10-class head."""
from __future__ import annotations
import argparse
import csv
import json
import random
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset

class PermutedDigits(Dataset):
    def __init__(self, mnist, permutation: torch.Tensor):
        self.mnist = mnist
        self.permutation = permutation.long()
        assert self.permutation.shape == (784,)
    def __len__(self):
        return len(self.mnist)
    def __getitem__(self, idx):
        x = self.mnist.data[idx].reshape(-1).float().div_(255.0)
        return x[self.permutation], int(self.mnist.targets[idx])

class DigitMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 10))
    def forward(self, x):
        return self.net(x)

def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def loaders(mnist_train, mnist_test, seed, train_limit, test_limit, batch_size, workers):
    perm_a = torch.arange(784)
    perm_b = torch.randperm(784, generator=torch.Generator().manual_seed(seed + 5001))
    train_idx = torch.randperm(len(mnist_train), generator=torch.Generator().manual_seed(seed + 8001))[:train_limit].tolist()
    test_idx = torch.randperm(len(mnist_test), generator=torch.Generator().manual_seed(seed + 8002))[:test_limit].tolist()
    a_train_ds = Subset(PermutedDigits(mnist_train, perm_a), train_idx)
    b_train_ds = Subset(PermutedDigits(mnist_train, perm_b), train_idx)
    a_test_ds = Subset(PermutedDigits(mnist_test, perm_a), test_idx)
    b_test_ds = Subset(PermutedDigits(mnist_test, perm_b), test_idx)
    def loader(ds, shuffle=False, stream=0):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=workers, pin_memory=False,
                          generator=torch.Generator().manual_seed(seed + stream))
    return (loader(a_train_ds, True, 10), loader(b_train_ds, True, 11),
            loader(a_test_ds, False, 12), loader(b_test_ds, False, 13), a_train_ds)

@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        prediction = model(x.to(device)).argmax(dim=1).cpu()
        correct += int((prediction == y).sum())
        total += y.numel()
    return round(100 * correct / total, 2)

def train_one_epoch(model, loader, optimizer, device, *, replay=None, replay_fraction=0.25):
    model.train()
    total_loss = total_count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if replay is not None:
            buffer_x, buffer_y = replay
            n_old = max(1, round(x.size(0) * replay_fraction / (1 - replay_fraction)))
            indices = torch.randint(len(buffer_y), (n_old,), device=device)
            x = torch.cat((x, buffer_x[indices]), dim=0)
            y = torch.cat((y, buffer_y[indices]), dim=0)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * x.size(0)
        total_count += x.size(0)
    return round(total_loss / total_count, 4)

def make_replay_buffer(a_train_ds, count, seed, device):
    indices = torch.randperm(len(a_train_ds), generator=torch.Generator().manual_seed(seed + 9001))[:count]
    samples = [a_train_ds[i] for i in indices.tolist()]
    return (torch.stack([s[0] for s in samples]).to(device),
            torch.tensor([s[1] for s in samples], dtype=torch.long, device=device))

def experiment(args, mnist_train, mnist_test, seed, method, device):
    seed_everything(seed)
    a_train, b_train, a_test, b_test, a_train_ds = loaders(
        mnist_train, mnist_test, seed, args.train_limit, args.test_limit,
        args.batch_size, args.workers)
    model = DigitMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    def log(stage, epoch, loss=None):
        row = {"seed": seed, "method": method, "stage": stage, "epoch": epoch,
               "A_accuracy": accuracy(model, a_test, device),
               "B_accuracy": accuracy(model, b_test, device), "loss": loss}
        history.append(row)
        print(f"{method:<8} seed={seed} {stage:<8} epoch={epoch:<2} "
              f"A={row['A_accuracy']:>6.2f}% B={row['B_accuracy']:>6.2f}% loss={loss}", flush=True)
    log("initial", 0)
    for epoch in range(1, args.epochs_a + 1):
        log("train_A", epoch, train_one_epoch(model, a_train, optimizer, device))
    old_accuracy = history[-1]["A_accuracy"]
    replay = make_replay_buffer(a_train_ds, args.buffer_size, seed, device) if method == "replay" else None
    for epoch in range(1, args.epochs_b + 1):
        log("train_B", epoch, train_one_epoch(model, b_train, optimizer, device,
            replay=replay, replay_fraction=args.replay_fraction))
    return history, {
        "seed": seed, "method": method, "A_after_A": old_accuracy,
        "A_after_B": history[-1]["A_accuracy"],
        "B_after_B": history[-1]["B_accuracy"],
        "A_forgetting": round(old_accuracy - history[-1]["A_accuracy"], 2),
    }

def write_csv(path, rows):
    if not rows: return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def plot_results(history, output_dir, epochs_a):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; CSV saved; plot skipped.")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for seed, method in sorted({(r["seed"], r["method"]) for r in history}):
        rows = [r for r in history if (r["seed"], r["method"]) == (seed, method)]
        for task in ("A", "B"):
            ax.plot(range(len(rows)), [r[f"{task}_accuracy"] for r in rows],
                    marker="o", label=f"{method} seed={seed} task={task}")
    ax.axvline(x=epochs_a + 0.5, linestyle="--", alpha=0.6)
    ax.set_xlabel("Checkpoint (initial -> A -> B)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy.png", dpi=150)
    plt.close(fig)

def parse_args():
    p = argparse.ArgumentParser(description="Lab01 sequential Permuted MNIST")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--train-limit", type=int, default=12000)
    p.add_argument("--test-limit", type=int, default=3000)
    p.add_argument("--epochs-a", type=int, default=3)
    p.add_argument("--epochs-b", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--buffer-size", type=int, default=1000)
    p.add_argument("--replay-fraction", type=float, default=0.25)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--methods", choices=["naive","replay"], nargs="+", default=["naive","replay"])
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    args = p.parse_args()
    assert 0 < args.train_limit <= 60000
    assert 0 < args.test_limit <= 10000
    assert 0 < args.buffer_size <= args.train_limit
    assert 0 < args.replay_fraction < 1
    assert args.epochs_a >= 1 and args.epochs_b >= 1
    return args

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; use --device cpu.")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    print("torch:", torch.__version__, "device:", device)
    from torchvision.datasets import MNIST
    training = MNIST(root=args.data_dir, train=True, download=True)
    test = MNIST(root=args.data_dir, train=False, download=True)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    histories, summaries = [], []
    for seed in args.seeds:
        for method in args.methods:
            hist, summ = experiment(args, training, test, seed, method, device)
            histories.extend(hist)
            summaries.append(summ)
    write_csv(output_dir / "history.csv", histories)
    write_csv(output_dir / "summary.csv", summaries)
    (output_dir / "config.json").write_text(
        json.dumps(vars(args) | {"torch_version":torch.__version__, "device_used":str(device)},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    plot_results(histories, output_dir, args.epochs_a)
    print("SUMMARY")
    for row in summaries: print(row)
    print("Files:", output_dir.resolve())
