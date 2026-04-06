import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

class EGNNLayer(MessagePassing):
    """equivariant message passing layer"""
    def __init__(self, hidden_dim):
        super().__init__(aggr="mean")

        # edge message
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # distance to move closer / farther
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

        # update hidden state of atom
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    def forward(self, h, x, edge_index, t):
        row, col = edge_index
        squared_distance = torch.sum((x[row] - x[col]) ** 2, dim=-1, keepdim=True)
        message = self.edge_mlp(torch.cat([h[row], h[col], squared_distance, t[row]], dim=-1))

        # equivariant
        distance_update = (x[row] - x[col]) * self.coord_mlp(message)
        distance_update = scatter(distance_update, row, dim=0, dim_size=x.size(0), reduce="mean")
        
        # invariant
        invariant_update = scatter(message, row, dim=0, dim_size=h.size(0), reduce="mean")
        invariant_update = h + self.node_mlp(torch.cat([h, invariant_update], dim=-1))

        return invariant_update, distance_update

class MinimalGNN(nn.Module):
    def __init__(self, num_particles=13, hidden_dim=64, num_layers=4):
        super().__init__()
        self.num_particles = num_particles
        self.hidden_dim = hidden_dim
        self.node_embedding = nn.Parameter(torch.randn(1, hidden_dim))
        self.layers = nn.ModuleList([EGNNLayer(hidden_dim) for i in range(num_layers)])
    def forward(self, t, x):
        B = x.shape[0]
        N = self.num_particles
        device = x.device

        x = x.view(B * N, 3)
        t = t.repeat_interleave(N, dim=0)
        h = self.node_embedding.expand(B * N, -1)

        idx = torch.arange(N)
        row, col = torch.combinations(idx, r=2).T
        row, col = torch.cat([row, col]), torch.cat([col, row])
        
        row_batched = torch.cat([row + b * N for b in range(B)]).to(device)
        col_batched = torch.cat([col + b * N for b in range(B)]).to(device)
        edge_index = torch.stack([row_batched, col_batched], dim=0)

        total_drift = torch.zeros_like(x)
        for layer in self.layers:
            h, distance_update = layer(h, x, edge_index, t)
            total_drift = total_drift + distance_update
        total_drift = total_drift.view(B, N, 3)
        total_drift = total_drift - total_drift.mean(dim=1, keepdim=True)
        return total_drift.view(B, N * 3)