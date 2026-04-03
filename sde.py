import torch
import torch.nn as nn
# from abc import ABC, abstractmethod

class BaseSDE(nn.Module):
    def __init__(self):
        super().__init__()
    def drift(self, t, x):
        """f(t, x)"""
        pass
    def diffusion(self, t):
        """g(t)"""
        pass
    def marginal_prob(self, x0, t):
        """mean and std of p(xt | x0)"""
        pass

class VESDE(BaseSDE):
    def __init__(self, sigma_min=0.01, sigma_max=1.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_ratio = sigma_max / sigma_min
        self.total_var = sigma_max ** 2 - sigma_min ** 2
    def _diffsquare_integral(self, t):
        """integral of g^2(t) from 0 to t"""
        return self.sigma_max ** 2 * (1 - (self.sigma_ratio) ** (-2 * t))
    def drift(self, t, x):
        """no drift"""
        return torch.zeros_like(x)
    def diffusion(self, t):
        return self.sigma_min * (self.sigma_ratio ** (1 - t)) * ((2 * torch.log(torch.tensor(self.sigma_ratio))) ** 0.5)
    def marginal_prob(self, x0, t):
        mean = x0
        var = self._diffsquare_integral(t)
        std = torch.sqrt(var.clamp(min=0))
        return mean, std
    def sample_posterior(self, t, x0, x1):
        """sample xt given x0 and x1"""
        t_adjusted = self._diffsquare_integral(t) / self.total_var
        mean = (1 - t_adjusted) * x0 + t_adjusted * x1
        var = self.total_var * t_adjusted * (1 - t_adjusted)
        noise = torch.sqrt((var).clamp(min=0)) * torch.randn_like(x0)
        return mean + noise
    
class VPSDE(BaseSDE):
    def __init__(self, beta_min=0.01, beta_max=20.0, sigma=1.0):
        super().__init__()
        self.beta0 = beta_min
        self.beta1 = beta_max
        self.sigma = sigma
    def _beta(self, t):
        return self.beta0 + t * (self.beta1 - self.beta0)
    def drift(self, t, x):
        return -0.5 * self._beta(t) * x
    def diffusion(self, t):
        return self.sigma * torch.sqrt(self._beta(t))
