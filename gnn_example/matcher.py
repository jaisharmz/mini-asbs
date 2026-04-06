import torch
import torch.nn as nn
from sde import BaseSDE

class AdjointMatcher:
    def __init__(self, sde: BaseSDE, steps=100, center_fn=None):
        self.sde = sde
        self.steps = steps
        self.center_fn = center_fn
    @torch.no_grad()
    def simulate(self, x0):
        dt = 1.0 / self.steps
        x = x0.clone()
        for step in range(self.steps):
            t = torch.full((x0.shape[0], 1), step * dt, device=x0.device)
            drift = self.sde.drift(t, x)
            noise = torch.randn_like(x)
            if self.center_fn is not None:
                noise = self.center_fn(noise)
            diffusion = self.sde.diffusion(t) * dt ** 0.5 * noise
            x = x + drift * dt + diffusion
        return x
    def get_loss(self, adjoint_model, corrector_model, energy_fn, x0):
        x1 = self.simulate(x0)
        x1.requires_grad_(True)
        energy = energy_fn(x1)
        energy_grad = torch.autograd.grad(energy.sum(), x1)[0]
        energy_grad = torch.nan_to_num(energy_grad, nan=0.0, posinf=0.0, neginf=0.0)
        grad_norm = torch.linalg.vector_norm(energy_grad, dim=-1, keepdim=True)
        clip_coeff = torch.clamp(100.0 / (grad_norm + 1e-6), max=1.0)
        energy_grad = energy_grad * clip_coeff

        t1 = torch.ones(x0.shape[0], 1, device=x0.device)
        with torch.no_grad():
            correction = corrector_model(t1, x1)
        
        adjoint1 = energy_grad + correction

        t = torch.rand(x0.shape[0], 1, device=x0.device)
        xt = self.sde.sample_posterior(t, x0, x1)
        scaling = self.sde.adjoint_scaling(t)
        target = -adjoint1 * scaling
        pred = adjoint_model(t, xt)

        loss = torch.mean(torch.sum((pred - target) ** 2, dim=-1))
        return loss
    def get_corrector_loss(self, corrector_model, x0):
        t = torch.rand(x0.shape[0], 1, device=x0.device)
        mean, std = self.sde.marginal_prob(t, x0)
        noise = torch.randn_like(x0)
        if self.center_fn is not None:
            noise = self.center_fn(noise)
        xt = mean + std * noise
        target = -noise # / std
        pred = corrector_model(t, xt)
        loss = torch.mean(torch.sum((std * pred - target) ** 2, dim=-1))
        return loss