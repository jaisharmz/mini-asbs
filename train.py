import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sde import VESDE, VPSDE
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

def train_asbs():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sde = VPSDE()
    matcher = AdjointMatcher(sde, steps=500)
    energy_fn = GMM2DEnergy(device=device)

    adjoint_model = MinimalMLP().to(device)
    corrector_model = MinimalMLP().to(device)

    optimizer_adjoint_model = optim.Adam(adjoint_model.parameters(), lr=1e-3)
    optimizer_corrector_model = optim.Adam(corrector_model.parameters(), lr=1e-3)

    batch_size = 512
    epochs = 10
    corrector_iterations = 1000
    adjoint_iterations = 1000
    adjoint_model.train()
    print("[START TRAINING]")
    for epoch in range(epochs):
        for iteration in range(adjoint_iterations):
            x0 = torch.randn(batch_size, 2, device=device)
            loss_adjoint_model = matcher.get_loss(adjoint_model, corrector_model, energy_fn, x0)
            optimizer_adjoint_model.zero_grad()
            loss_adjoint_model.backward()
            optimizer_adjoint_model.step()
            if iteration % max(adjoint_iterations // 2, 1) == 0:
                print(f"Adjoint Model | Epoch: {epoch} | Training Iteration: {epoch * adjoint_iterations + iteration} | Loss: {loss_adjoint_model.item():.4f}")
        for iteration in range(corrector_iterations):
            x0 = torch.randn(batch_size, 2, device=device)
            loss_corrector_model = matcher.get_corrector_loss(corrector_model, x0)
            optimizer_corrector_model.zero_grad()
            loss_corrector_model.backward()
            optimizer_corrector_model.step()
            if iteration % max(corrector_iterations // 2, 1) == 0:
                print(f"Corrector Model | Epoch: {epoch} | Training Iteration: {epoch * adjoint_iterations + iteration} | Loss: {loss_corrector_model.item():.4f}")
        if epoch % max(epochs // 10, 1) == 0:
            print(f"Epoch: {epoch} | Adjoint Model Loss: {loss_adjoint_model.item():.4f} | Corrector Model Loss: {loss_corrector_model.item():.4f}")
    print("[END TRAINING]")
    return adjoint_model, corrector_model, sde

@torch.no_grad()
def sample_and_plot(adjoint_model, sde):
    device = next(adjoint_model.parameters()).device
    adjoint_model.eval()
    steps = 100
    dt = 1.0 / steps
    x = torch.randn(1000, 2, device=device) * 2.0
    for step in range(steps):
        t = torch.full((x.shape[0], 1), step * dt, device=device)
        g_t = sde.diffusion(t)
        base_drift = sde.drift(t, x)
        control_drift = (g_t ** 2) * adjoint_model(t, x)
        drift = base_drift + control_drift
        noise = torch.randn_like(x) * g_t * dt ** 0.5
        x = x + drift * dt + noise
    x_numpy = x.cpu().numpy()
    plt.scatter(x_numpy[:,0], x_numpy[:,1])
    plt.title("Generated Samples (4 Mode GMM)")
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.grid(True)
    plt.savefig("test_vpsde3.png")

if __name__ == "__main__":
    model, corrector_model, sde = train_asbs()
    sample_and_plot(model, sde)