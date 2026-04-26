"""
Offline dataset collection for synthetic environments.

We collect a mixed behavior dataset:
    - 40% near-optimal (close to a mode, one random mode per episode)
    - 40% random exploration (uniform over action space)
    - 20% suboptimal-but-structured (always picks mode 0 regardless of state)

This mixture ensures:
    1. The dataset contains enough good data for the critic to learn Q accurately
    2. The dataset is multimodal (multiple good actions per state exist)
    3. The dataset is not perfectly optimal (offline RL is nontrivial)
    4. A BC baseline learns a blurry average (proving ASBS is needed)

Usage:
    dataset = collect_dataset("multimodal_bandit", n_transitions=100000)
    # dataset is a dict with keys: observations, actions, rewards, next_observations, dones
"""

import numpy as np
from envs.synthetic import make_env


def collect_dataset(
    env_name: str,
    n_transitions: int = 200_000,
    near_optimal_frac: float = 0.40,
    random_frac: float = 0.40,
    suboptimal_frac: float = 0.20,
    noise_std: float = 0.15,
    seed: int = 0,
    **env_kwargs,
) -> dict:
    """
    Collect an offline dataset from the synthetic environment.

    Returns a dict of numpy arrays, each of shape (n_transitions, ...).
    """
    assert abs(near_optimal_frac + random_frac + suboptimal_frac - 1.0) < 1e-6

    rng = np.random.default_rng(seed)
    env = make_env(env_name, **env_kwargs)

    n_near_opt  = int(n_transitions * near_optimal_frac)
    n_random    = int(n_transitions * random_frac)
    n_subopt    = n_transitions - n_near_opt - n_random

    obs_list, act_list, rew_list, next_obs_list, done_list = [], [], [], [], []

    def run_episode(policy_fn):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        ep_obs, ep_act, ep_rew, ep_next, ep_done = [], [], [], [], []
        for _ in range(env.max_episode_steps):
            action = policy_fn(obs, env)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_obs.append(obs.copy())
            ep_act.append(action.copy())
            ep_rew.append(reward)
            ep_next.append(next_obs.copy())
            ep_done.append(float(done))
            obs = next_obs
            if done:
                break
        return ep_obs, ep_act, ep_rew, ep_next, ep_done

    # ---- Near-optimal: pick a random mode, add Gaussian noise ----
    def near_optimal_policy(obs, env):
        if hasattr(env, '_mode_centers'):
            centers = env._mode_centers(obs)           # (K, ac_dim)
            k = rng.integers(0, len(centers))
            return centers[k] + rng.normal(0, noise_std, size=centers[k].shape)
        elif hasattr(env, '_target'):
            # Reacher: compute IK-like action toward target
            return rng.normal(0, 0.1, size=env.action_space.shape)
        else:
            return env.action_space.sample()

    # ---- Random: uniform ----
    def random_policy(obs, env):
        return env.action_space.sample()

    # ---- Suboptimal: always mode 0 (biased, ignores state) ----
    def suboptimal_policy(obs, env):
        if hasattr(env, '_mode_centers'):
            centers = env._mode_centers(obs)
            return centers[0] + rng.normal(0, noise_std * 2, size=centers[0].shape)
        else:
            return env.action_space.low * 0.5

    policies = [
        (near_optimal_policy,  n_near_opt,  "near_optimal"),
        (random_policy,        n_random,    "random"),
        (suboptimal_policy,    n_subopt,    "suboptimal"),
    ]

    for policy_fn, n_target, name in policies:
        collected = 0
        while collected < n_target:
            ep = run_episode(policy_fn)
            for i in range(len(ep[0])):
                obs_list.append(ep[0][i])
                act_list.append(ep[1][i])
                rew_list.append(ep[2][i])
                next_obs_list.append(ep[3][i])
                done_list.append(ep[4][i])
                collected += 1
                if collected >= n_target:
                    break

    # Shuffle the dataset
    indices = rng.permutation(len(obs_list))

    dataset = {
        "observations":      np.array(obs_list,      dtype=np.float32)[indices],
        "actions":           np.array(act_list,       dtype=np.float32)[indices],
        "rewards":           np.array(rew_list,       dtype=np.float32)[indices],
        "next_observations": np.array(next_obs_list,  dtype=np.float32)[indices],
        "dones":             np.array(done_list,       dtype=np.float32)[indices],
    }

    print(f"[Dataset] {env_name}: {len(indices):,} transitions")
    print(f"  reward  mean={dataset['rewards'].mean():.3f}  "
          f"std={dataset['rewards'].std():.3f}  "
          f"max={dataset['rewards'].max():.3f}")
    print(f"  obs     shape={dataset['observations'].shape}")
    print(f"  actions shape={dataset['actions'].shape}")

    env.close()
    return dataset


class ReplayBuffer:
    """Simple numpy replay buffer, same interface as hw5 version."""

    def __init__(self, capacity: int = 1_000_000):
        self.capacity = capacity
        self.size = 0
        self._ptr = 0
        self.observations      = None
        self.actions           = None
        self.rewards           = None
        self.next_observations = None
        self.dones             = None

    @classmethod
    def from_dataset(cls, dataset: dict) -> "ReplayBuffer":
        n = len(dataset["observations"])
        buf = cls(capacity=n)
        buf.observations      = dataset["observations"].copy()
        buf.actions           = dataset["actions"].copy()
        buf.rewards           = dataset["rewards"].copy()
        buf.next_observations = dataset["next_observations"].copy()
        buf.dones             = dataset["dones"].copy()
        buf.size = n
        return buf

    def sample(self, batch_size: int) -> dict:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "observations":      self.observations[idx],
            "actions":           self.actions[idx],
            "rewards":           self.rewards[idx],
            "next_observations": self.next_observations[idx],
            "dones":             self.dones[idx],
        }

    def insert(self, observation, action, reward, next_observation, done):
        if self.observations is None:
            self.observations      = np.empty((self.capacity, *observation.shape), dtype=np.float32)
            self.actions           = np.empty((self.capacity, *action.shape),      dtype=np.float32)
            self.rewards           = np.empty((self.capacity,),                    dtype=np.float32)
            self.next_observations = np.empty((self.capacity, *next_observation.shape), dtype=np.float32)
            self.dones             = np.empty((self.capacity,),                    dtype=np.float32)
        ptr = self._ptr % self.capacity
        self.observations[ptr]      = observation
        self.actions[ptr]           = action
        self.rewards[ptr]           = reward
        self.next_observations[ptr] = next_observation
        self.dones[ptr]             = float(done)
        self._ptr += 1
        self.size = min(self.size + 1, self.capacity)