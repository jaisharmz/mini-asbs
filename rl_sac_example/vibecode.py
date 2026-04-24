import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import collections
import random

# ==========================================
# 1. THE "TREE IN THE ROAD" ENVIRONMENT
# ==========================================
class TreeEnv:
    def __init__(self):
        self.state = np.array([0.0, 0.0])
        self.goal = np.array([0.0, 10.0])
        self.tree = np.array([0.0, 5.0])
        self.tree_radius = 2.0
        self.step_count = 0
        self.max_steps = 15

    def reset(self):
        self.state = np.array([0.0, 0.0])
        self.step_count = 0
        return self.state.copy()

    def step(self, action):
        self.step_count += 1
        
        # Move agent
        action = np.clip(action, -1.0, 1.0)
        new_state = self.state + action
        
        # Calculate Distances
        dist_to_goal = np.linalg.norm(new_state - self.goal)
        dist_to_tree = np.linalg.norm(new_state - self.tree)
        
        # Reward Function
        reward = -dist_to_goal * 0.1 # Small penalty for being far from goal
        done = False
        
        # Success
        if dist_to_goal < 1.0:
            reward += 100.0
            done = True
            
        # CRASH INTO TREE! (The multimodal trap)
        if dist_to_tree < self.tree_radius:
            reward -= 100.0 
            done = True
            
        # Timeout
        if self.step_count >= self.max_steps:
            done = True
            
        self.state = new_state
        return self.state.copy(), reward, done

# ==========================================
# 2. STANDARD SAC COMPONENTS (Gaussian Policy)
# ==========================================
class GaussianActor(nn.Module):
    def __init__(self, state_dim=2, action_dim=2, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        
        # Reparameterization trick
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()
        action = torch.tanh(z) # Bound between [-1, 1]
        
        # Compute log prob for entropy
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True), mu

class TwinCritic(nn.Module):
    def __init__(self, state_dim=2, action_dim=2, hidden_dim=128):
        super().__init__()
        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_prime, d = map(np.stack, zip(*batch))
        return (torch.FloatTensor(s), torch.FloatTensor(a), 
                torch.FloatTensor(r).unsqueeze(1), torch.FloatTensor(s_prime), 
                torch.FloatTensor(d).unsqueeze(1))

    def __len__(self):
        return len(self.buffer)

# ==========================================
# 3. SAC TRAINING LOOP
# ==========================================
def train_sac():
    env = TreeEnv()
    actor = GaussianActor()
    critic = TwinCritic()
    critic_target = TwinCritic()
    critic_target.load_state_dict(critic.state_dict())
    
    actor_opt = optim.Adam(actor.parameters(), lr=3e-3)
    critic_opt = optim.Adam(critic.parameters(), lr=3e-3)
    
    buffer = ReplayBuffer()
    batch_size = 128
    gamma = 0.99
    tau = 0.005
    alpha = 0.2 # Entropy temperature
    
    episodes = 5000
    rewards_history = []
    
    print("[TRAINING SAC BASELINE]")
    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0
        
        while True:
            # Act
            with torch.no_grad():
                action, _, _ = actor(torch.FloatTensor(state).unsqueeze(0))
            action_np = action.numpy()[0]
            
            next_state, reward, done = env.step(action_np)
            buffer.push(state, action_np, reward, next_state, done)
            state = next_state
            ep_reward += reward
            
            # Train
            if len(buffer) > batch_size:
                s, a, r, s_prime, d = buffer.sample(batch_size)
                
                # --- Critic Update ---
                with torch.no_grad():
                    next_a, next_log_prob, _ = actor(s_prime)
                    target_q1, target_q2 = critic_target(s_prime, next_a)
                    target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
                    target_value = r + (1 - d) * gamma * target_q
                    
                q1, q2 = critic(s, a)
                critic_loss = F.mse_loss(q1, target_value) + F.mse_loss(q2, target_value)
                
                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()
                
                # --- Actor Update ---
                curr_a, curr_log_prob, _ = actor(s)
                q1_curr, q2_curr = critic(s, curr_a)
                q_curr = torch.min(q1_curr, q2_curr)
                
                actor_loss = (alpha * curr_log_prob - q_curr).mean()
                
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()
                
                # Soft update target critic
                for target_param, param in zip(critic_target.parameters(), critic.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
                    
            if done:
                break
                
        rewards_history.append(ep_reward)
        if ep % 50 == 0:
            print(f"Episode {ep} | Reward: {ep_reward:.2f}")

    # ==========================================
    # 4. EVALUATION & PLOTTING
    # ==========================================
    print("[EVALUATING TRAJECTORIES]")
    plt.figure(figsize=(6, 8))
    
    # Plot Tree and Goal
    circle = plt.Circle((0, 5), 2.0, color='red', alpha=0.5, label='Tree (Crash Zone)')
    plt.gca().add_patch(circle)
    plt.scatter([0], [10], color='green', marker='*', s=200, label='Goal')
    plt.scatter([0], [0], color='black', marker='o', s=100, label='Start')
    
    crashes = 0
    for _ in range(10):
        state = env.reset()
        path = [state.copy()]
        for _ in range(env.max_steps):
            with torch.no_grad():
                # Get the mean action from the Gaussian
                _, _, mu = actor(torch.FloatTensor(state).unsqueeze(0))
                action = torch.tanh(mu).numpy()[0] 
            state, r, done = env.step(action)
            path.append(state.copy())
            if r <= -50: crashes += 1
            if done: break
            
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], color='blue', alpha=0.5)

    print(f"Crash Rate: {crashes/10 * 100}%")
    plt.xlim(-5, 5)
    plt.ylim(-1, 11)
    plt.title(f"SAC Baseline Trajectories (Crash Rate: {crashes/10 * 100}%)")
    plt.legend()
    plt.grid(True)
    plt.savefig("sac_failure.png")
    
if __name__ == "__main__":
    train_sac()