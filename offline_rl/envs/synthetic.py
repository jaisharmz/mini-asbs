"""
Synthetic environments for benchmarking multimodal RL policies.

All environments share a common interface:
    env.reset()  -> (state, info)
    env.step(action) -> (next_state, reward, terminated, truncated, info)

The key property: the optimal policy is MULTIMODAL in all environments.
A Gaussian/deterministic policy structurally cannot represent it.

Environments
------------
1. MultimodalBandit
   - State:  s ~ Uniform([-1,1]^2)
   - Action: a in R^2
   - Reward: logsumexp of 4 Gaussian peaks, centers shift linearly with s
   - Optimal: 4-modal policy, modes shift with state
   - Why hard: modes are 4 units apart, mode-averaging kills reward

2. BimodalObstacle
   - State:  s in R^2 (position toward goal)
   - Action: a in R^2 (velocity)
   - Reward: shaped toward goal, large penalty for hitting central obstacle
   - Optimal: go left OR go right (bimodal)
   - Why hard: Gaussian policy averages through the obstacle

3. MultimodalReacher
   - State:  (target_angle, arm_config) in R^4
   - Action: joint_velocities in R^2
   - Reward: minimize distance to any of 4 symmetric target configurations
   - Optimal: 4 ways to reach each target (arm symmetry)
   - Why hard: deterministic policy commits to one arm configuration

Each env has `env.plot_reward_landscape(ax)` for visualization.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------------------------
# 1. MultimodalBandit  (the primary benchmark)
# ---------------------------------------------------------------------------

class MultimodalBandit(gym.Env):
    """
    Single-step bandit with a 4-modal reward landscape.

    The reward is a log-sum-exp of 4 Gaussians in action space.
    Mode centers shift linearly with the 2D state, so the policy
    must be both multimodal AND state-conditioned.

    Optimal policy: place equal mass on all 4 modes for each state.
    Gaussian policy best case: average of modes -> reward near 0.
    """

    metadata = {"render_modes": []}

    BASE_MODES = np.array([
        [-2.0, -2.0],
        [ 2.0, -2.0],
        [-2.0,  2.0],
        [ 2.0,  2.0],
    ], dtype=np.float32)

    def __init__(self, state_dim=2, action_dim=2, mode_std=0.4, state_shift_scale=0.5):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.mode_std = mode_std
        self.state_shift_scale = state_shift_scale

        # Linear map from state to mode shift: shift = W @ s
        # W is fixed at construction, so modes vary smoothly with state
        np.random.seed(42)
        self.W = np.random.randn(2, state_dim).astype(np.float32) * state_shift_scale

        self.observation_space = spaces.Box(-1.0, 1.0, shape=(state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-4.0, 4.0, shape=(action_dim,), dtype=np.float32)

        self._state = None
        self.max_episode_steps = 1

    def _mode_centers(self, state: np.ndarray) -> np.ndarray:
        """Return (4, 2) mode centers for this state."""
        shift = (self.W @ state)  # (2,)
        return self.BASE_MODES + shift[None, :]  # (4, 2)

    def reward(self, state: np.ndarray, action: np.ndarray) -> float:
        """
        R(s, a) = log sum_k exp(-||a - mu_k(s)||^2 / (2*sigma^2))

        This is the log-probability (up to constant) of a mixture of Gaussians.
        Maximized when a is exactly at any mode center.
        Global max = log(K) = log(4) ≈ 1.386.
        Gaussian policy averages to center -> reward ≈ log(4 * exp(-d^2/(2sig^2)))
        where d is distance from center to mean of modes (= 2*sqrt(2) ≈ 2.83).
        """
        centers = self._mode_centers(state)  # (4, 2)
        dists_sq = np.sum((action[None, :] - centers) ** 2, axis=-1)  # (4,)
        log_probs = -dists_sq / (2.0 * self.mode_std ** 2)
        return float(np.logaddexp.reduce(log_probs))

    def max_reward(self, state: np.ndarray) -> float:
        """Theoretical maximum reward for this state (action exactly on a mode)."""
        return float(np.log(len(self.BASE_MODES)))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._state = self.observation_space.sample()
        return self._state.copy(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        reward = self.reward(self._state, action)
        # Normalize to [0, 1] range for easier comparison across methods
        normalized_reward = reward / self.max_reward(self._state)
        next_state = self.observation_space.sample()  # new random state each step
        self._state = next_state
        return next_state.copy(), normalized_reward, False, True, {"raw_reward": reward}

    def plot_reward_landscape(self, ax, state=None, resolution=100):
        """Plot the 2D reward landscape for a given state."""
        if state is None:
            state = np.zeros(self.state_dim, dtype=np.float32)
        a1 = np.linspace(-4, 4, resolution)
        a2 = np.linspace(-4, 4, resolution)
        A1, A2 = np.meshgrid(a1, a2)
        R = np.zeros((resolution, resolution))
        for i in range(resolution):
            for j in range(resolution):
                R[i, j] = self.reward(state, np.array([A1[i, j], A2[i, j]]))
        im = ax.contourf(A1, A2, R, levels=30, cmap='viridis')
        centers = self._mode_centers(state)
        ax.scatter(centers[:, 0], centers[:, 1], c='red', s=100,
                   marker='*', zorder=5, label='Mode centers')
        ax.set_xlabel('Action dim 0')
        ax.set_ylabel('Action dim 1')
        return im


# ---------------------------------------------------------------------------
# 2. BimodalObstacle  (navigation with a central obstacle)
# ---------------------------------------------------------------------------

class BimodalObstacle(gym.Env):
    """
    Short-horizon navigation where a central obstacle forces bimodal routing.

    State:  (agent_x, agent_y, goal_x, goal_y) in R^4
    Action: velocity (vx, vy) in [-1, 1]^2
    Episode: 10 steps, agent must reach goal while avoiding obstacle at (0, 0)

    The optimal policy is bimodal: go left (negative x) OR right (positive x)
    around the obstacle. The Gaussian policy averages through it and gets penalized.

    Reward:
        - At each step: -0.1 * dist_to_goal  (dense, encourage progress)
        - If hits obstacle (within radius 0.5): -5.0  (large penalty)
        - If reaches goal (within radius 0.3): +10.0  (large bonus)
    """

    metadata = {"render_modes": []}

    OBSTACLE_POS = np.array([0.0, 0.0], dtype=np.float32)
    OBSTACLE_RADIUS = 0.5
    GOAL_RADIUS = 0.3

    def __init__(self, max_steps=10):
        super().__init__()
        self.max_steps = max_steps
        self.max_episode_steps = max_steps

        # State: (agent_x, agent_y, goal_x, goal_y)
        self.observation_space = spaces.Box(-3.0, 3.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self._state = None
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Agent starts at bottom, goal is at top, obstacle is in between
        agent = np.array(
            [self.np_random.uniform(-0.3, 0.3), -2.0], dtype=np.float32
        )
        goal = np.array(
            [self.np_random.uniform(-0.3, 0.3), 2.0], dtype=np.float32
        )
        self._state = np.concatenate([agent, goal])
        self._step_count = 0
        return self._state.copy(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        agent = self._state[:2]
        goal = self._state[2:]

        # Move agent
        new_agent = agent + action * 0.4  # scale velocity
        new_agent = np.clip(new_agent, -3.0, 3.0)

        # Check obstacle
        dist_obstacle = np.linalg.norm(new_agent - self.OBSTACLE_POS)
        hit_obstacle = dist_obstacle < self.OBSTACLE_RADIUS

        # Check goal
        dist_goal = np.linalg.norm(new_agent - goal)
        reached_goal = dist_goal < self.GOAL_RADIUS

        # Reward
        reward = -0.1 * dist_goal
        if hit_obstacle:
            reward -= 5.0
        if reached_goal:
            reward += 10.0

        self._state = np.concatenate([new_agent, goal])
        self._step_count += 1
        terminated = reached_goal or hit_obstacle
        truncated = self._step_count >= self.max_steps

        return (
            self._state.copy(),
            float(reward),
            terminated,
            truncated,
            {"reached_goal": reached_goal, "hit_obstacle": hit_obstacle},
        )


# ---------------------------------------------------------------------------
# 3. MultimodalReacher  (symmetric arm with 4 equivalent solutions)
# ---------------------------------------------------------------------------

class MultimodalReacher(gym.Env):
    """
    2-joint planar arm that must reach a target. Due to arm symmetry,
    there are 4 equivalent joint configurations that solve each task.

    State:  (cos(j1), sin(j1), cos(j2), sin(j2), target_x, target_y) in R^6
    Action: (d_j1, d_j2) joint angle deltas in [-0.3, 0.3]^2
    Episode: 20 steps

    The 4 equivalent solutions come from elbow-up/elbow-down symmetry
    combined with the +-pi periodicity of joint 1.

    Reward: -dist_to_target  (dense, penalize distance of end-effector)
    """

    metadata = {"render_modes": []}

    L1 = 1.0  # upper arm length
    L2 = 0.8  # forearm length

    def __init__(self, max_steps=20):
        super().__init__()
        self.max_steps = max_steps
        self.max_episode_steps = max_steps

        self.observation_space = spaces.Box(-2.0, 2.0, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(-0.3, 0.3, shape=(2,), dtype=np.float32)

        self._j1 = 0.0
        self._j2 = 0.0
        self._target = np.zeros(2, dtype=np.float32)
        self._step_count = 0

    def _end_effector(self, j1, j2):
        x = self.L1 * np.cos(j1) + self.L2 * np.cos(j1 + j2)
        y = self.L1 * np.sin(j1) + self.L2 * np.sin(j1 + j2)
        return np.array([x, y], dtype=np.float32)

    def _obs(self):
        return np.array([
            np.cos(self._j1), np.sin(self._j1),
            np.cos(self._j2), np.sin(self._j2),
            self._target[0], self._target[1],
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._j1 = self.np_random.uniform(-np.pi, np.pi)
        self._j2 = self.np_random.uniform(-np.pi, np.pi)
        # Target reachable by the arm
        r = self.np_random.uniform(0.3, self.L1 + self.L2 - 0.1)
        angle = self.np_random.uniform(-np.pi, np.pi)
        self._target = np.array([r * np.cos(angle), r * np.sin(angle)], dtype=np.float32)
        self._step_count = 0
        return self._obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -0.3, 0.3)
        self._j1 = (self._j1 + action[0]) % (2 * np.pi)
        self._j2 = (self._j2 + action[1]) % (2 * np.pi)

        ee = self._end_effector(self._j1, self._j2)
        dist = np.linalg.norm(ee - self._target)
        reward = -dist

        self._step_count += 1
        terminated = dist < 0.05
        truncated = self._step_count >= self.max_steps

        return self._obs(), float(reward), terminated, truncated, {"dist": dist}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ENVS = {
    "multimodal_bandit":   MultimodalBandit,
    "bimodal_obstacle":    BimodalObstacle,
    "multimodal_reacher":  MultimodalReacher,
}


def make_env(env_name: str, **kwargs) -> gym.Env:
    if env_name not in ENVS:
        raise ValueError(f"Unknown env '{env_name}'. Choose from: {list(ENVS.keys())}")
    return ENVS[env_name](**kwargs)