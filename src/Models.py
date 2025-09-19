import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math


current_rabbit_pos = torch.zeros(2)
current_wolf_pos = torch.zeros(2)


rabbit_positions = []
wolf_positions = []
noisy_rabbit_positions = []



# Neural network classes with activation tracking
class RabbitNet(nn.Module):
    def __init__(self):
        super(RabbitNet, self).__init__()
        self.fc1 = nn.Linear(5, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 6)
        self.fc5 = nn.Linear(6, 2)
        self.layer_activations = []
        
    def forward(self, x):
        self.layer_activations = []
        
        # Record input as first layer activation
        self.layer_activations.append(x.detach().numpy())
        
        x = F.relu(self.fc1(x))
        self.layer_activations.append(x.detach().numpy())
        
        x = F.relu(self.fc2(x))
        self.layer_activations.append(x.detach().numpy())
        
        x = F.relu(self.fc3(x))
        self.layer_activations.append(x.detach().numpy())
        
        x = F.relu(self.fc4(x))
        self.layer_activations.append(x.detach().numpy())
        
        x = self.fc5(x)
        self.layer_activations.append(x.detach().numpy())
        
        return x

class WolfNet(nn.Module):
    def __init__(self):
        super(WolfNet, self).__init__()
        self.fc1 = nn.Linear(5, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 6)
        self.fc5 = nn.Linear(6, 2)
        self.layer_activations = []
        
    def forward(self, x):
        self.layer_activations = []
        
        # Record input as first layer activation
        self.layer_activations.append(x.detach().numpy())
        
        x = F.relu(self.fc1(x))
        self.layer_activations.append(x.detach().numpy())
        
        x = F.relu(self.fc2(x))
        self.layer_activations.append(x.detach().numpy())
        
        x = F.relu(self.fc3(x))
        self.layer_activations.append(x.detach().numpy())
        
        x = F.relu(self.fc4(x))
        self.layer_activations.append(x.detach().numpy())
        
        x = self.fc5(x)
        self.layer_activations.append(x.detach().numpy())
        
        return x

# Create instances of both networks
rabbit_net = RabbitNet()
wolf_net = WolfNet()

# Create optimizers
opt_rabbit = torch.optim.Adam(rabbit_net.parameters(), lr=1e-3)
opt_wolf = torch.optim.Adam(wolf_net.parameters(), lr=1e-3)


def reset_models():
    global current_rabbit_pos, current_wolf_pos, rabbit_net, wolf_net, opt_rabbit, opt_wolf
    
    # Reset networks
    rabbit_net = RabbitNet()
    wolf_net = WolfNet()
    
    # Reset optimizers
    opt_rabbit = torch.optim.Adam(rabbit_net.parameters(), lr=1e-3)
    opt_wolf = torch.optim.Adam(wolf_net.parameters(), lr=1e-3)
    
    # Reset positions
    current_rabbit_pos = torch.zeros(2)
    current_wolf_pos = torch.zeros(2)
    
    print("Models and positions reset to initial state")


print("Rabbit Network:")
print(rabbit_net)
print("\nWolf Network:")
print(wolf_net)

def get_noisy_observation(rabbit_pos):
    angle = torch.rand(1) * 2 * math.pi
    radius = torch.rand(1).sqrt()  # Uniform in circle
    offset = torch.tensor([torch.cos(angle), torch.sin(angle)]) * radius
    return rabbit_pos + offset

# Training function
def train_step(num_steps=100):
    global rabbit_positions, wolf_positions, noisy_rabbit_positions, current_rabbit_pos, current_wolf_pos
    
    # Reset position tracking
    rabbit_positions = []
    wolf_positions = []
    noisy_rabbit_positions = []

    
    # Zero gradients
    opt_rabbit.zero_grad()
    opt_wolf.zero_grad()
    
    # Initialize positions and history
    rabbit_pos = torch.tensor(current_rabbit_pos, requires_grad=True)
    wolf_pos = torch.tensor(current_wolf_pos, requires_grad=True)
    rabbit_log_probs = []
    wolf_log_probs = []

    for step in range(num_steps):
        # Rabbit's move
        state_rabbit = torch.cat([rabbit_pos, wolf_pos.detach(), torch.tensor([step / 100.0])])
        action_mean = rabbit_net(state_rabbit)
        action_mean = action_mean / torch.norm(action_mean)
        
        # Create distribution with requires_grad=True
        dist = torch.distributions.Normal(action_mean, 0.1)
        action_rabbit = dist.rsample()  # This maintains gradient connection
        action_rabbit = action_rabbit / torch.norm(action_rabbit)
        log_prob_rabbit = dist.log_prob(action_rabbit).sum()  # Calculate proper log prob

        rabbit_pos = rabbit_pos + action_rabbit
        rabbit_log_probs.append(log_prob_rabbit)
        rabbit_positions.append(rabbit_pos.detach().numpy().copy())

        # Wolf's noisy observation - detach rabbit_pos to break gradient connection
        obs_rabbit = get_noisy_observation(rabbit_pos.detach())
        noisy_rabbit_positions.append(obs_rabbit.detach().numpy().copy())

        # Wolf's move
        state_wolf = torch.cat([wolf_pos, obs_rabbit, torch.tensor([step / 100.0])])
        action_mean_wolf = wolf_net(state_wolf)
        action_mean_wolf = action_mean_wolf / torch.norm(action_mean_wolf)
        
        # Create distribution for wolf
        dist_wolf = torch.distributions.Normal(action_mean_wolf, 0.1)
        action_wolf = dist_wolf.rsample()
        action_wolf = action_wolf / torch.norm(action_wolf)
        log_prob_wolf = dist_wolf.log_prob(action_wolf).sum()

        wolf_pos = wolf_pos + action_wolf
        wolf_log_probs.append(log_prob_wolf)
        wolf_positions.append(wolf_pos.detach().numpy().copy())


    # Update current positions for next training step
    current_rabbit_pos = rabbit_pos.detach().clone()
    current_wolf_pos = wolf_pos.detach().clone()

    # Check win condition
    distance = torch.norm(rabbit_pos - wolf_pos)
    rabbit_reward = 1.0 if distance >= 1.0 else 0.0
    wolf_reward = 1.0 - rabbit_reward

    # Policy gradient update for rabbit
    rabbit_loss = -torch.stack(rabbit_log_probs).sum() * rabbit_reward
    rabbit_loss.backward()
    opt_rabbit.step()

    # Policy gradient update for wolf
    wolf_loss = -torch.stack(wolf_log_probs).sum() * wolf_reward
    wolf_loss.backward()
    opt_wolf.step()
    
    print(f"Training step complete. Rabbit reward: {rabbit_reward}, Wolf reward: {wolf_reward}")
    print(f"Current positions | Rabbit: {current_rabbit_pos}, Wolf: {current_wolf_pos}|")