import torch
import torch.nn as nn
import torch.nn.functional as functional
import math
import numpy as np

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
        
        x = functional.relu(self.fc1(x))
        self.layer_activations.append(x.detach().numpy())
        
        x = functional.relu(self.fc2(x))
        self.layer_activations.append(x.detach().numpy())
        
        x = functional.relu(self.fc3(x))
        self.layer_activations.append(x.detach().numpy())
        
        x = functional.relu(self.fc4(x))
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
        
        x = functional.relu(self.fc1(x))
        self.layer_activations.append(x.detach().numpy())
        
        x = functional.relu(self.fc2(x))
        self.layer_activations.append(x.detach().numpy())
        
        x = functional.relu(self.fc3(x))
        self.layer_activations.append(x.detach().numpy())
        
        x = functional.relu(self.fc4(x))
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

# Improved reward system
def calculate_rewards(rabbit_positions, wolf_positions, noisy_rabbit_positions):
    """
    Calculate rewards for both rabbit and wolf based on their performance
    
    Rabbit rewards:
    1. Positive reward for increasing distance from wolf
    2. Negative reward for decreasing distance from wolf
    3. Small penalty for moving too much (to encourage efficiency)
    4. Large penalty for being caught
    
    Wolf rewards:
    1. Positive reward for decreasing distance to true rabbit position
    2. Negative reward for increasing distance to true rabbit position
    3. Small positive reward for following the noisy observation
    4. Large reward for catching the rabbit
    """
    rabbit_rewards = []
    wolf_rewards = []
    
    # Calculate distances at each step
    distances = torch.norm(rabbit_positions - wolf_positions, dim=1)
    
    # Calculate distances to noisy observations (only for steps where we have both)
    min_length = min(len(wolf_positions), len(noisy_rabbit_positions))
    distances_to_noisy = torch.norm(wolf_positions[:min_length] - noisy_rabbit_positions[:min_length], dim=1)
    
    # Calculate movement amounts
    rabbit_movements = torch.norm(rabbit_positions[1:] - rabbit_positions[:-1], dim=1)
    wolf_movements = torch.norm(wolf_positions[1:] - wolf_positions[:-1], dim=1)
    
    # Rabbit rewards
    for i in range(1, len(distances)):
        # Base reward based on distance change
        distance_change = distances[i] - distances[i-1]
        rabbit_reward = distance_change * 0.1  # Scale the distance change
        
        # Small penalty for moving too much (encourage efficiency)
        if i-1 < len(rabbit_movements):
            movement_penalty = rabbit_movements[i-1] * 0.01
            rabbit_reward -= movement_penalty
            
        # Large penalty if caught (distance very small)
        if distances[i] < 1.0:
            rabbit_reward -= 10.0
            
        rabbit_rewards.append(rabbit_reward)
    
    # Wolf rewards
    for i in range(1, min(len(distances), len(distances_to_noisy)+1)):
        # Base reward based on distance change to real rabbit
        distance_change = distances[i-1] - distances[i]  # Opposite of rabbit
        wolf_reward = distance_change * 0.05
        
        # Small reward for following the noisy observation
        if i-1 < len(distances_to_noisy):
            noisy_distance_change = distances_to_noisy[i-2] - distances_to_noisy[i-1] if i > 1 else 0
            wolf_reward += noisy_distance_change * 0.1
            
        # # Small penalty for moving too much
        # if i-1 < len(wolf_movements):
        #     movement_penalty = wolf_movements[i-1] * 0.001
        #     wolf_reward -= movement_penalty
            
        # Large reward for catching the rabbit
        if distances[i] < 1.0:
            wolf_reward += 10.0
            
        wolf_rewards.append(wolf_reward)
    
    # Make sure both reward lists have the same length
    min_reward_length = min(len(rabbit_rewards), len(wolf_rewards))
    return (torch.tensor(rabbit_rewards[:min_reward_length]), 
            torch.tensor(wolf_rewards[:min_reward_length]))

# Training function with improved reward system
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
    
    # Store positions for reward calculation
    all_rabbit_positions = [rabbit_pos.detach().clone()]
    all_wolf_positions = [wolf_pos.detach().clone()]
    all_noisy_positions = []
    
    for step in range(num_steps):
        # Rabbit's move
        state_rabbit = torch.cat([rabbit_pos, wolf_pos.detach(), torch.tensor([step / 1])])
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
        all_rabbit_positions.append(rabbit_pos.detach().clone())

        # Wolf's noisy observation - detach rabbit_pos to break gradient connection
        obs_rabbit = get_noisy_observation(rabbit_pos.detach())
        noisy_rabbit_positions.append(obs_rabbit.detach().numpy().copy())
        all_noisy_positions.append(obs_rabbit.detach().clone())

        # Wolf's move
        state_wolf = torch.cat([wolf_pos, obs_rabbit, torch.tensor([step / 1])])
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
        all_wolf_positions.append(wolf_pos.detach().clone())

    # Update current positions for next training step
    current_rabbit_pos = rabbit_pos.detach().clone()
    current_wolf_pos = wolf_pos.detach().clone()
    
    # Convert to tensors
    rabbit_tensor = torch.stack(all_rabbit_positions)
    wolf_tensor = torch.stack(all_wolf_positions)
    noisy_tensor = torch.stack(all_noisy_positions) if all_noisy_positions else torch.zeros(1, 2)
    
    # Calculate rewards
    rabbit_rewards, wolf_rewards = calculate_rewards(rabbit_tensor, wolf_tensor, noisy_tensor)
    
    # Policy gradient update for rabbit
    if len(rabbit_log_probs) > 0 and len(rabbit_rewards) > 0:
        min_length = min(len(rabbit_log_probs), len(rabbit_rewards))
        rabbit_loss = -torch.stack([log_p * r for log_p, r in 
                                  zip(rabbit_log_probs[:min_length], rabbit_rewards[:min_length])]).sum()
        rabbit_loss.backward()
        opt_rabbit.step()

    # Policy gradient update for wolf
    if len(wolf_log_probs) > 0 and len(wolf_rewards) > 0:
        min_length = min(len(wolf_log_probs), len(wolf_rewards))
        wolf_loss = -torch.stack([log_p * r for log_p, r in 
                                zip(wolf_log_probs[:min_length], wolf_rewards[:min_length])]).sum()
        wolf_loss.backward()
        opt_wolf.step()
    
    # Calculate final distance
    print(rabbit_rewards, "\n", wolf_rewards)
    final_distance = torch.norm(all_rabbit_positions[-1] - all_wolf_positions[-1])
    
    print(f"Training step complete. Final distance: {final_distance:.2f}")
    print(f"Current positions | Rabbit: {current_rabbit_pos}, Wolf: {current_wolf_pos}|")