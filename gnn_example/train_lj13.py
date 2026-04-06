import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sde import VESDE, VPSDE
from matcher import AdjointMatcher
from energy import LJ13Energy
from model import MinimalGNN

class MinimalMLP(nn.Module):
    def __init__(self, x_dim=8, hidden_dim=256):
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
        outputs = self.net(inputs)
        # zero center of mass for particles
        outputs = outputs.view(-1, 4, 2)
        outputs = outputs - outputs.mean(dim=1, keepdim=True)
        outputs = outputs.view(-1, 8)
        return outputs

class ZeroCorrector(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, t, x):
        return torch.zeros_like(x)

def zero_center_3d(x):
    B = x.shape[0]
    x = x.view(B, 13, 3)
    x = x - x.mean(dim=1, keepdim=True)
    return x.view(B, 39)

def train_asbs():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sde = VESDE()
    matcher = AdjointMatcher(sde, steps=500, center_fn=zero_center_3d)
    energy_fn = LJ13Energy(device=device)

    adjoint_model = MinimalGNN(num_particles=13).to(device)
    corrector_model = MinimalGNN(num_particles=13).to(device)

    optimizer_adjoint_model = optim.Adam(adjoint_model.parameters(), lr=1e-3)
    optimizer_corrector_model = optim.Adam(corrector_model.parameters(), lr=1e-3)

    batch_size = 512
    epochs = 2000
    corrector_iterations = 1000
    adjoint_iterations = 1000
    adjoint_model.train()
    print("[INITIAL EVAL]")
    sample_and_plot(adjoint_model, sde, f"test_lj13_0.png")
    print("[START TRAINING]")
    for epoch in range(epochs):
        for iteration in range(adjoint_iterations):
            x0 = torch.randn(batch_size, 39, device=device)
            x0 = zero_center_3d(x0)
            loss_adjoint_model = matcher.get_loss(adjoint_model, corrector_model, energy_fn, x0)
            optimizer_adjoint_model.zero_grad()
            loss_adjoint_model.backward()
            optimizer_adjoint_model.step()
            if iteration % max(adjoint_iterations // 2, 1) == 0:
                print(f"Adjoint Model | Epoch: {epoch} | Training Iteration: {epoch * adjoint_iterations + iteration} | Loss: {loss_adjoint_model.item():.4f}")
        for iteration in range(corrector_iterations):
            x0 = torch.randn(batch_size, 39, device=device)
            x0 = zero_center_3d(x0)
            loss_corrector_model = matcher.get_corrector_loss(corrector_model, x0)
            optimizer_corrector_model.zero_grad()
            loss_corrector_model.backward()
            optimizer_corrector_model.step()
            if iteration % max(corrector_iterations // 2, 1) == 0:
                print(f"Corrector Model | Epoch: {epoch} | Training Iteration: {epoch * adjoint_iterations + iteration} | Loss: {loss_corrector_model.item():.4f}")
        if epoch % max(epochs // 10000, 1) == 0:
            print(f"Epoch: {epoch} | Adjoint Model Loss: {loss_adjoint_model.item():.4f} | Corrector Model Loss: {loss_corrector_model.item():.4f}")
            sample_and_plot(adjoint_model, sde, f"test_lj13_{epoch}.png")
    print("[END TRAINING]")
    return adjoint_model, corrector_model, sde

@torch.no_grad()
def sample_and_plot(adjoint_model, sde, filename=None):
    device = next(adjoint_model.parameters()).device
    adjoint_model.eval()
    steps = 100
    dt = 1.0 / steps
    x = torch.randn(1000, 39, device=device)
    x = zero_center_3d(x)
    for step in range(steps):
        t = torch.full((x.shape[0], 1), step * dt, device=device)
        g_t = sde.diffusion(t)
        base_drift = sde.drift(t, x)
        control_drift = (g_t ** 2) * adjoint_model(t, x)
        control_drift = torch.clamp(control_drift, min=-100.0, max=100.0)
        drift = base_drift + control_drift
        noise = torch.randn_like(x)
        noise = zero_center_3d(noise)
        noise = noise * g_t * dt ** 0.5
        x = x + drift * dt + noise
    energy_fn = LJ13Energy(device=device)
    final_energies = energy_fn(x).cpu().numpy()
    print(f"Mean Final Energy: {final_energies.mean():.2f}")
    print(f"Min Final Energy (Best Molecule): {final_energies.min():.2f}") # target: -44.32
    
    x = x.view(-1, 13, 3)
    pairwise_distances = torch.norm(x.unsqueeze(2) - x.unsqueeze(1), dim=-1)
    idx = torch.triu_indices(13, 13, offset=1)
    distances = pairwise_distances[:, idx[0], idx[1]].flatten().cpu().numpy()
    plt.hist(distances, bins=100, density=True)
    plt.title("LJ13 Pairwise Distance Distribution")
    plt.grid(True)
    plt.savefig(filename or "test_lj13.png")
    plt.clf()

if __name__ == "__main__":
    model, corrector_model, sde = train_asbs()
    sample_and_plot(model, sde)