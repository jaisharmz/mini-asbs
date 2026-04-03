import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sde import VESDE
from matcher import AdjointMatcher
from energy import GMM2DEnergy

class MinimalMLP(nn.Module):
    def __init__(self, x_dim=2, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, x_dim)
        )
    def forward(self, t, x):
        inputs = torch.cat([t, x], dim=-1)
        return self.net(inputs)

class ZeroCorrector(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, t, x):
        return torch.zeros_like(x)

def train_minimal_asbs():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sde = VESDE()
    matcher = AdjointMatcher(sde, steps=500)
    energy_fn = GMM2DEnergy(device=device)

    adjoint_model = MinimalMLP().to(device)
    corrector_model = ZeroCorrector().to(device)

    optimizer = optim.Adam(adjoint_model.parameters(), lr=1e-3)

    batch_size = 512
    epochs = 2000
    adjoint_model.train()
    print("[START TRAINING]")
    for epoch in range(epochs):
        x0 = torch.randn(batch_size, 2, device=device)

        loss = matcher.get_loss(adjoint_model, corrector_model, energy_fn, x0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % (epochs // 10) == 0:
            print(f"Epoch: {epoch} | Loss: {loss.item():.4f}")
    print("[END TRAINING]")
    return adjoint_model, sde

@torch.no_grad()
def sample_and_plot(adjoint_model, sde):
    device = next(adjoint_model.parameters()).device
    adjoint_model.eval()
    steps = 100
    dt = 1.0 / steps
    x = torch.randn(1000, 2, device=device)
    for step in range(steps):
        t = torch.full((x.shape[0], 1), step * dt, device=device)
        g_t = sde.diffusion(t)
        learned_drift = (g_t ** 2) * adjoint_model(t, x)
        noise = torch.randn_like(x) * g_t * dt ** 0.5
        x = x + learned_drift + noise
    x_numpy = x.cpu().numpy()
    plt.scatter(x_numpy[:,0], x_numpy[:,1])
    plt.title("Generated Samples (4 Mode GMM)")
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.grid(True)
    plt.savefig("test.png")

if __name__ == "__main__":
    model, sde = train_minimal_asbs()
    sample_and_plot(model, sde)