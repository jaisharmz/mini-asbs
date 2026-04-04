import torch
import torch.nn as nn

class GMM2DEnergy:
    def __init__(self, device="cpu"):
        self.device = device
        self.means = torch.tensor([
            [-3.0, -3.0],
            [3.0, 3.0],
            [-3.0, 3.0], 
            [3.0, -3.0]
        ]).to(device)
        self.var = 0.5
    def __call__(self, x):
        """return energy of x"""
        x = x.unsqueeze(1)
        means = self.means.unsqueeze(0)
        square_distance = torch.sum((x - means) ** 2, dim=-1)
        logprobs = -square_distance / (2 * self.var)
        energy = -torch.logsumexp(logprobs, dim=1)
        return energy

class DW4Energy:
    def __init__(self, a=0.9, b=-4.0, c=0.0, offset=4.0, device="cpu"):
        self.a = a
        self.b = b
        self.c = c
        self.offset = offset
        self.device = device
    def __call__(self, x):
        B, dim = x.shape
        assert dim == 8
        x = x.view(B, 4, 2)
        pairwise_distances = torch.sqrt(torch.sum((x.unsqueeze(2) - x.unsqueeze(1)) ** 2, dim=-1) + 1e-8)
        idx = torch.triu_indices(4, 4, offset=1)
        distances = pairwise_distances[:, idx[0], idx[1]]
        shift = distances - self.offset
        energy = self.a * shift ** 4 + self.b * shift ** 2 + self.c
        energy = energy.sum(dim=-1)
        return energy
