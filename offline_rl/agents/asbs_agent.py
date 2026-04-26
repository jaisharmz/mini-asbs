"""
Adjoint Schrödinger Bridge Sampler (ASBS) for Offline RL.

Target distribution:
    pi*(a|s) ∝ pi_beta(a|s) * exp(tau * Q(s, a))

We learn this via two alternating matching objectives:

1. Adjoint Matching (AM):
   Regresses the flow policy u_theta to the adjoint target at t=1:
       target = tau * grad_{a1} Q(s, a1) + h_phi(a1, s)
   scaled back through time by adjoint_scaling(t), and evaluated at
   bridge posterior samples a_t ~ p_base(· | a0, a1).

2. Corrector Matching (CM):
   Regresses h_phi to grad_{a1} log p_base(a1 | a0), which for a
   linear interpolant a_t = (1-t)*a0 + t*a1 with noise std sigma is:
       grad_{a1} log p_base(a1 | a0) = -(a1 - a0) / sigma^2
   This is the score of the Gaussian bridge at t=1.

The two objectives are alternated every `corrector_update_every` steps,
implementing the IPF algorithm (Theorem 3.2 in ASBS paper).

Critic is a twin ensemble trained with standard TD backup (identical to FQL).
"""

from typing import Callable, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import infrastructure.pytorch_util as ptu


class ASBSAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        # Network factories (same pattern as FQL / SAC+BC)
        make_flow_actor: Callable,
        make_flow_actor_optimizer: Callable,
        make_corrector: Callable,
        make_corrector_optimizer: Callable,
        make_critic: Callable,
        make_critic_optimizer: Callable,

        # RL hyperparameters
        discount: float = 0.99,
        target_update_rate: float = 0.005,
        tau: float = 1.0,               # inverse temperature: how sharply to peak on Q

        # Flow hyperparameters
        flow_steps: int = 10,           # Euler integration steps at inference
        sigma: float = 0.1,             # noise std of the Gaussian source / bridge

        # AM / CM training schedule
        corrector_update_every: int = 1,  # update corrector every N AM steps
        grad_clip: float = 1.0,           # clip energy/Q gradient norm

        # Replay buffer-style source: how many sim steps to roll out for AM target
        sim_steps: int = 20,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.ob_dim = int(np.prod(observation_shape))

        # Networks
        self.flow_actor = make_flow_actor(observation_shape, action_dim)
        self.corrector = make_corrector(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.flow_actor_optimizer = make_flow_actor_optimizer(self.flow_actor.parameters())
        self.corrector_optimizer = make_corrector_optimizer(self.corrector.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())

        # Hyperparameters
        self.discount = discount
        self.target_update_rate = target_update_rate
        self.tau = tau
        self.flow_steps = flow_steps
        self.sigma = sigma
        self.corrector_update_every = corrector_update_every
        self.grad_clip = grad_clip
        self.sim_steps = sim_steps

        # IPF stage counter (used to decide AM vs CM updates)
        self._step = 0

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Euler integration of the flow from noise to action, conditioned on obs."""
        obs = ptu.from_numpy(np.asarray(observation, dtype=np.float32))[None]  # (1, ob_dim)
        a = torch.randn(1, self.action_dim, device=ptu.device) * self.sigma
        dt = 1.0 / self.flow_steps
        for i in range(self.flow_steps):
            t = torch.full((1, 1), i * dt, device=ptu.device)
            v = self.flow_actor(obs, a, t)
            a = a + dt * v
        a = torch.clamp(a, -1.0, 1.0)
        return ptu.to_numpy(a[0])

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _sample_source(self, B: int) -> torch.Tensor:
        """Sample a0 ~ N(0, sigma^2 I)."""
        return torch.randn(B, self.action_dim, device=ptu.device) * self.sigma

    def _simulate(self, obs: torch.Tensor, a0: torch.Tensor) -> torch.Tensor:
        """
        Run the current flow policy forward (stop_grad) from a0 to get a1.
        Used to get on-policy terminal samples for the AM and CM objectives.
        """
        a = a0.clone()
        dt = 1.0 / self.sim_steps
        with torch.no_grad():
            for i in range(self.sim_steps):
                t = torch.full((a.shape[0], 1), i * dt, device=ptu.device)
                v = self.flow_actor(obs, a, t)
                a = a + dt * v
        return a  # a1

    def _sample_bridge_posterior(
        self, t: torch.Tensor, a0: torch.Tensor, a1: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample a_t ~ p_base(· | a0, a1) for the linear interpolant bridge.

        For the flow model with linear interpolant:
            a_t | a0, a1 ~ N((1-t)*a0 + t*a1,  sigma^2 * t * (1-t) * I)

        This is the standard Brownian bridge posterior with constant diffusion sigma.
        """
        mean = (1.0 - t) * a0 + t * a1
        std = self.sigma * torch.sqrt((t * (1.0 - t)).clamp(min=1e-6))
        noise = torch.randn_like(a0)
        return mean + std * noise

    def _adjoint_scaling(self, t: torch.Tensor) -> torch.Tensor:
        """
        Scale the terminal adjoint target back to time t.

        For the constant-diffusion flow SDE with sigma_t = sigma (constant),
        the adjoint ODE is trivial (no base drift f_t = 0), so the scaling
        from t=1 back to t is just 1.0.  We keep this as a hook for future
        non-trivial schedules.
        """
        return torch.ones_like(t)

    # ------------------------------------------------------------------
    # Critic update  (standard TD, identical to FQL)
    # ------------------------------------------------------------------

    def update_critic(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        with torch.no_grad():
            # Sample next actions from the flow policy (stop-grad)
            a0_next = self._sample_source(observations.shape[0])
            next_actions = self._simulate(next_observations, a0_next).clamp(-1.0, 1.0)
            q1_next, q2_next = self.target_critic(next_observations, next_actions)
            y = rewards + self.discount * (1.0 - dones) * torch.min(q1_next, q2_next)

        q1, q2 = self.critic(observations, actions)
        loss = nn.functional.mse_loss(q1, y) + nn.functional.mse_loss(q2, y)

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "q_loss": loss.item(),
            "q_mean": ((q1 + q2) / 2).mean().item(),
            "q_max": torch.max(q1, q2).max().item(),
            "q_min": torch.min(q1, q2).min().item(),
        }

    # ------------------------------------------------------------------
    # AM: Adjoint Matching update
    # ------------------------------------------------------------------

    def update_flow_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,     # dataset actions (unused for AM target, but useful debug)
    ) -> dict:
        """
        Adjoint Matching loss (Eq. 14 in ASBS paper):

            L_AM(theta) = E_{t, a0, a1, a_t} [
                || u_theta(t, a_t, s) + sigma_t * adjoint_target ||^2
            ]

        where adjoint_target = (tau * grad_{a1} Q(s, a1) + h_phi(a1, s)) * lambda_t
        and lambda_t = adjoint_scaling(t) / sigma.
        """
        B = observations.shape[0]

        # 1. Sample source a0 and simulate to get terminal a1
        a0 = self._sample_source(B)
        a1 = self._simulate(observations, a0)   # (B, ac_dim), stop_grad

        # 2. Compute adjoint target at t=1: tau * grad_Q + corrector
        a1_req = a1.detach().requires_grad_(True)
        q1, q2 = self.critic(observations, a1_req.clamp(-1.0, 1.0))
        q = torch.min(q1, q2).sum()
        q_grad = torch.autograd.grad(q, a1_req)[0]   # (B, ac_dim)

        # Clip gradient norm per sample (prevents exploding targets)
        grad_norm = torch.linalg.vector_norm(q_grad, dim=-1, keepdim=True)
        clip_coeff = torch.clamp(self.grad_clip / (grad_norm + 1e-6), max=1.0)
        q_grad = q_grad * clip_coeff

        with torch.no_grad():
            correction = self.corrector(observations, a1.detach())

        adjoint_at_1 = self.tau * q_grad + correction  # (B, ac_dim)

        # 3. Sample random time and bridge posterior a_t
        t = torch.rand(B, 1, device=ptu.device)
        a_t = self._sample_bridge_posterior(t, a0.detach(), a1.detach())

        # 4. Scale adjoint target back to time t (trivial for constant sigma)
        scaling = self._adjoint_scaling(t)              # (B, 1)
        target = -adjoint_at_1 * scaling                # (B, ac_dim)  [negative: we add to u]

        # 5. AM regression loss
        pred = self.flow_actor(observations, a_t, t)    # (B, ac_dim)
        loss = torch.mean(torch.sum((pred - target) ** 2, dim=-1))

        self.flow_actor_optimizer.zero_grad()
        loss.backward()
        self.flow_actor_optimizer.step()

        return {
            "am_loss": loss.item(),
            "q_grad_norm": grad_norm.mean().item(),
            "correction_norm": torch.linalg.vector_norm(correction, dim=-1).mean().item(),
        }

    # ------------------------------------------------------------------
    # CM: Corrector Matching update
    # ------------------------------------------------------------------

    def update_corrector(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict:
        """
        Corrector Matching loss (Eq. 15 in ASBS paper):

            L_CM(phi) = E_{a0, a1} [
                || h_phi(a1, s) - grad_{a1} log p_base(a1 | a0) ||^2
            ]

        For the linear interpolant bridge with constant noise sigma:
            grad_{a1} log p_base(a1 | a0) at t=1 is the score of the marginal
            p_base(a1) = N(a0, sigma^2 I), so:
            grad_{a1} log p_base(a1 | a0) = -(a1 - a0) / sigma^2

        This is the standard Gaussian score.
        """
        B = observations.shape[0]

        # Sample source and simulate to get on-policy terminal samples
        a0 = self._sample_source(B)
        with torch.no_grad():
            a1 = self._simulate(observations, a0)

        # Target: score of p_base(a1 | a0) = N(a0, sigma^2 I)
        target = -(a1 - a0) / (self.sigma ** 2)    # (B, ac_dim)

        # Clip target norm for stability
        target_norm = torch.linalg.vector_norm(target, dim=-1, keepdim=True)
        clip_coeff = torch.clamp(self.grad_clip / (target_norm + 1e-6), max=1.0)
        target = target * clip_coeff

        pred = self.corrector(observations, a1)     # (B, ac_dim)
        loss = torch.mean(torch.sum((pred - target) ** 2, dim=-1))

        self.corrector_optimizer.zero_grad()
        loss.backward()
        self.corrector_optimizer.step()

        return {
            "cm_loss": loss.item(),
            "corrector_target_norm": target_norm.mean().item(),
        }

    # ------------------------------------------------------------------
    # Target critic update (Polyak)
    # ------------------------------------------------------------------

    def update_target_critic(self) -> None:
        with torch.no_grad():
            for p, p_tgt in zip(self.critic.parameters(), self.target_critic.parameters()):
                p_tgt.data.copy_(
                    self.target_update_rate * p.data
                    + (1.0 - self.target_update_rate) * p_tgt.data
                )

    # ------------------------------------------------------------------
    # Main update (called from run.py training loop)
    # ------------------------------------------------------------------

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ) -> dict:
        self._step = step

        # Critic update every step
        metrics_critic = self.update_critic(observations, actions, rewards, next_observations, dones)

        # AM update every step
        metrics_am = self.update_flow_actor(observations, actions)

        # CM update every `corrector_update_every` steps (IPF alternation)
        metrics_cm = {}
        if step % self.corrector_update_every == 0:
            metrics_cm = self.update_corrector(observations, actions)

        self.update_target_critic()

        metrics = {
            **{f"critic/{k}": v for k, v in metrics_critic.items()},
            **{f"am/{k}": v for k, v in metrics_am.items()},
            **{f"cm/{k}": v for k, v in metrics_cm.items()},
        }
        return metrics