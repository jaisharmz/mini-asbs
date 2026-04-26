"""
Flow networks for ASBS.

FlowPolicy:  u_theta(t, a_t, s)  -- the controlled drift, inputs (obs, action, time)
Corrector:   h_phi(a_1, s)       -- debiases non-memoryless source, inputs (obs, action) at t=1
"""

import torch
import torch.nn as nn
from infrastructure import pytorch_util as ptu


class FlowPolicy(nn.Module):
    """
    Velocity field u_theta(t, a_t, s).

    Takes (observation, noisy_action, time) and outputs a velocity in action space.
    This is the ASBS controlled drift -- at inference time we integrate it from t=0 to t=1.

    Input dim:  ob_dim + ac_dim + 1
    Output dim: ac_dim
    """

    def __init__(self, ob_dim: int, ac_dim: int, hidden_size: int = 256, n_layers: int = 4):
        super().__init__()
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim

        layers = []
        in_dim = ob_dim + ac_dim + 1  # obs + action + scalar time
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.SiLU())
            in_dim = hidden_size
        layers.append(nn.Linear(hidden_size, ac_dim))
        self.net = nn.Sequential(*layers)
        self.to(ptu.device)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs:    (B, ob_dim)
            action: (B, ac_dim)
            t:      (B, 1)   scalar time in [0, 1]
        Returns:
            velocity: (B, ac_dim)
        """
        x = torch.cat([obs, action, t], dim=-1)
        return self.net(x)


class Corrector(nn.Module):
    """
    Corrector h_phi(a_1, s).

    Learns to approximate nabla_{a_1} log phi_hat_1(a_1), which debiases the
    AM objective when the source distribution is non-memoryless (non-Dirac-delta).

    Initialized to output zeros (so the first AM stage just regresses to the
    energy gradient, matching Algorithm 1 in the ASBS paper).

    Input dim:  ob_dim + ac_dim
    Output dim: ac_dim
    """

    def __init__(self, ob_dim: int, ac_dim: int, hidden_size: int = 256, n_layers: int = 4):
        super().__init__()
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim

        layers = []
        in_dim = ob_dim + ac_dim
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.SiLU())
            in_dim = hidden_size
        last_layer = nn.Linear(hidden_size, ac_dim)
        # Zero-initialize the last layer so h_phi starts at 0 (IPF initialization)
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)
        layers.append(last_layer)
        self.net = nn.Sequential(*layers)
        self.to(ptu.device)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs:    (B, ob_dim)
            action: (B, ac_dim)  -- terminal actions a_1
        Returns:
            correction: (B, ac_dim)
        """
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)