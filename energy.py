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