# train_add.py — your first training run (CPU, ~10 seconds)
import torch
import torch.nn as nn

torch.manual_seed(0)

# Synthetic dataset: learn a + b
N = 2000
a = torch.randint(0, 50, (N, 1)).float()
b = torch.randint(0, 50, (N, 1)).float()
X = torch.cat([a, b], dim=1)   # (N, 2) inputs
y = a + b                      # (N, 1) targets

model = nn.Sequential(
    nn.Linear(2, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 1),
)
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(2001):
    opt.zero_grad()
    loss = loss_fn(model(X), y)
    loss.backward()
    opt.step()
    if epoch % 500 == 0:
        print(f"epoch {epoch:4d} | loss {loss.item():.3f}")

# Test on numbers it never saw
test = torch.tensor([[37., 47.], [5., 9.]])
print("predictions:", model(test).detach().flatten().tolist())
print("true values: [84.0, 14.0]")