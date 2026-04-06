import torch
import torch.nn as nn

class LJ13Energy:
    def __init__(self, epsilon=1.0, sigma=1.0, device="cpu"):
        self.epsilon = epsilon
        self.sigma = sigma
        self.device = device
    def __call__(self, x):
        B = x.shape[0]
        x = x.view(B, 13, 3)
        pairwise_distances = torch.sqrt(torch.sum((x.unsqueeze(2) - x.unsqueeze(1)) ** 2, dim=-1) + 1e-8)
        idx = torch.triu_indices(13, 13, offset=1)
        distances = pairwise_distances[:, idx[0], idx[1]]
        distances = torch.clamp(distances, min=0.1)

        power6 = (self.sigma / distances) ** 6
        power12 = power6 ** 2
        energy = 4 * self.epsilon * (power12 - power6)
        
        return energy.sum(dim=-1)
