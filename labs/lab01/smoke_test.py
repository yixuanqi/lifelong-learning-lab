"""No-network smoke test covering learning, replay and accuracy."""
import torch
from torch.utils.data import DataLoader
from train import DigitMLP, PermutedDigits, accuracy, make_replay_buffer, train_one_epoch

class FakeMNIST:
    def __init__(self, n=64):
        torch.manual_seed(456)
        self.data = torch.randint(0, 256, (n,28,28), dtype=torch.uint8)
        self.targets = torch.randint(0,10,(n,), dtype=torch.long)
    def __len__(self): return len(self.targets)

fake = FakeMNIST()
a = PermutedDigits(fake, torch.arange(784))
b = PermutedDigits(fake, torch.randperm(784))
assert a[0][0].shape == (784,)
assert b[0][1] == a[0][1]
model = DigitMLP()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
loader_a = DataLoader(a, batch_size=16)
loader_b = DataLoader(b, batch_size=16)
loss_a = train_one_epoch(model, loader_a, opt, torch.device("cpu"))
memory = make_replay_buffer(a, 16, 42, torch.device("cpu"))
loss_b = train_one_epoch(model, loader_b, opt, torch.device("cpu"), replay=memory)
acc = accuracy(model, loader_a, torch.device("cpu"))
assert 0 <= acc <= 100 and loss_a > 0 and loss_b > 0
print("SMOKE TEST PASS")
