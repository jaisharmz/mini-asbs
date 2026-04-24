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
    def marginal_prob(self, t, x0):
        """mean and std of p(xt | x0)"""
        pass
    def adjoint_scaling(self, t):
        return torch.ones_like(t)

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
    def marginal_prob(self, t, x0):
        mean = x0
        var = self._diffsquare_integral(t)
        std = torch.sqrt(var.clamp(min=0))
        return mean, std
    def sample_posterior(self, t, x0, x1):
        """sample xt given x0 and x1"""
        t_adjusted = self._diffsquare_integral(t) / self.total_var
        mean = (1 - t_adjusted) * x0 + t_adjusted * x1
        var = self.total_var * t_adjusted * (1 - t_adjusted)
        noise = torch.sqrt(var.clamp(min=0)) * torch.randn_like(x0)
        return mean + noise
    
class VPSDE(BaseSDE):
    def __init__(self, beta_min=0.01, beta_max=20.0):
        super().__init__()
        self.beta0 = beta_min
        self.beta1 = beta_max
    def _beta(self, t):
        return self.beta0 + t * (self.beta1 - self.beta0)
    def _beta_integral(self, t0, t1):
        return self.beta0 * (t1 - t0) + 0.5 * (self.beta1 - self.beta0) * (t1 ** 2 - t0 ** 2)
    def drift(self, t, x):
        return -0.5 * self._beta(t) * x
    def diffusion(self, t):
        return torch.sqrt(self._beta(t))
    def marginal_prob(self, t, x0):
        decay = -0.5 * self._beta_integral(torch.zeros_like(t), t)
        mean = torch.exp(decay) * x0
        var = 1 - torch.exp(2 * decay)
        std = torch.sqrt(var.clamp(min=0))
        return mean, std
    def sample_posterior(self, t, x0, x1):
        decay_0t = -0.5 * self._beta_integral(torch.zeros_like(t), t)
        decay_t1 = -0.5 * self._beta_integral(t, torch.ones_like(t))
        decay_01 = -0.5 * self._beta_integral(torch.zeros_like(t), torch.ones_like(t))

        alpha_0t = torch.exp(decay_0t)
        alpha_t1 = torch.exp(decay_t1)
        
        var_0t = 1 - torch.exp(2 * decay_0t)
        var_t1 = 1 - torch.exp(2 * decay_t1)
        var_01 = 1 - torch.exp(2 * decay_01)

        mean = (alpha_t1 * var_0t / var_01) * x1 + (alpha_0t * var_t1 / var_01) * x0
        var = (var_0t * var_t1) / var_01
        std = torch.sqrt(var.clamp(min=0))
        noise = std * torch.randn_like(x0)

        return mean + noise
    def adjoint_scaling(self, t):
        return torch.exp(-0.5 * self._beta_integral(t, torch.ones_like(t)))