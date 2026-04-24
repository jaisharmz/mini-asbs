import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sde import VESDE
from matcher import AdjointMatcher

# ==========================================
# 1. THE Q-FUNCTION (CRITIC) & ENERGY WRAPPER
# ==========================================
class SyntheticQCritic(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        # The two optimal "swerve" actions to avoid the tree
        self.optimal_left = torch.tensor([-2.0, 2.0], device=device)
        self.optimal_right = torch.tensor([2.0, 2.0], device=device)
        
    def forward(self, state, action):
        # We simulate the Q-value landscape at state (0,0)
        dist_left = torch.sum((action - self.optimal_left)**2, dim=-1)
        dist_right = torch.sum((action - self.optimal_right)**2, dim=-1)
        
        # High Q-value (near 1.0) if close to either optimal action. Low Q-value (near 0.0) otherwise.
        q_value = torch.exp(-dist_left / 0.5) + torch.exp(-dist_right / 0.5)
        return q_value

class CriticEnergy:
    def __init__(self, critic, state_context):
        self.critic = critic
        self.state_context = state_context
        
    def __call__(self, action):
        # In ASBS, we MINIMIZE energy. 
        # Therefore, Energy = -Q(s, a). 
        # Minimizing this forces the SDE to maximize the Q-value!
        q_values = self.critic(self.state_context, action)
        return -q_values * 10.0 # Multiply by 10 to steepen the gradients

# ==========================================
# 2. THE STATE-CONDITIONED ASBS ACTOR
# ==========================================
class StateConditionedActor(nn.Module):
    def __init__(self, state_dim=2, action_dim=2, hidden_dim=128):
        super().__init__()
        # Input is Time(1) + Action(2) + State(2) = 5
        self.net = nn.Sequential(
            nn.Linear(1 + action_dim + state_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, t, action, state):
        inputs = torch.cat([t, action, state], dim=-1)
        return self.net(inputs)

# We use a dummy wrapper so AdjointMatcher (which expects f(t,x)) 
# can work with our state-conditioned actor seamlessly.
class ActorWrapper(nn.Module):
    def __init__(self, actor, fixed_state):
        super().__init__()
        self.actor = actor
        self.fixed_state = fixed_state
    def forward(self, t, x):
        return self.actor(t, x, self.fixed_state)

# ==========================================
# 3. TRAINING THE ASBS POLICY
# ==========================================
def train_rl_actor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sde = VESDE()
    matcher = AdjointMatcher(sde, steps=100) # No centering needed for RL actions!
    
    actor = StateConditionedActor().to(device)
    corrector = StateConditionedActor().to(device)
    
    # We are training the policy specifically for the state [0, 0]
    fixed_state = torch.zeros(512, 2, device=device) 
    wrapped_actor = ActorWrapper(actor, fixed_state)
    wrapped_corrector = ActorWrapper(corrector, fixed_state)
    
    critic = SyntheticQCritic(device=device)
    energy_fn = CriticEnergy(critic, fixed_state)

    opt_actor = optim.Adam(actor.parameters(), lr=1e-3)
    opt_corrector = optim.Adam(corrector.parameters(), lr=1e-3)

    print("[TRAINING ASBS MULTIMODAL POLICY]")
    for epoch in range(500):
        # Sample base action noise
        a0 = torch.randn(512, 2, device=device) 
        
        # Train Adjoint
        loss_actor = matcher.get_loss(wrapped_actor, wrapped_corrector, energy_fn, a0)
        opt_actor.zero_grad()
        loss_actor.backward()
        opt_actor.step()
        
        # Train Corrector
        a0 = torch.randn(512, 2, device=device)
        loss_corrector = matcher.get_corrector_loss(wrapped_corrector, a0)
        opt_corrector.zero_grad()
        loss_corrector.backward()
        opt_corrector.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Actor Loss: {loss_actor.item():.4f}")

    return actor, sde

# ==========================================
# 4. EVALUATE AND PLOT ACTIONS
# ==========================================
@torch.no_grad()
def evaluate_policy(actor, sde):
    device = next(actor.parameters()).device
    steps = 100
    dt = 1.0 / steps
    
    # Generate 2000 actions for state [0, 0]
    num_samples = 2000
    a = torch.randn(num_samples, 2, device=device)
    state_context = torch.zeros(num_samples, 2, device=device)
    
    for step in range(steps):
        t = torch.full((num_samples, 1), step * dt, device=device)
        g_t = sde.diffusion(t)
        
        control_drift = (g_t ** 2) * actor(t, a, state_context)
        noise = torch.randn_like(a) * g_t * dt ** 0.5
        
        a = a + control_drift * dt + noise
        
    actions = a.cpu().numpy()
    
    plt.figure(figsize=(6, 6))
    plt.scatter(actions[:, 0], actions[:, 1], alpha=0.5, color='blue', s=10)
    plt.scatter([-2.0, 2.0], [2.0, 2.0], color='red', marker='*', s=200, label='Optimal Q-Values')
    plt.title("ASBS Policy Actions at State (0,0)")
    plt.xlim(-5, 5); plt.ylim(-2, 6)
    plt.legend()
    plt.grid(True)
    plt.savefig("asbs_rl_actions.png")
    print("Saved asbs_rl_actions.png!")

if __name__ == "__main__":
    trained_actor, sde = train_rl_actor()
    evaluate_policy(trained_actor, sde)