import time
from math import atan2
from types import NoneType

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

all_rabbit_positions = []
all_wolf_positions = []
all_noisy_positions = []

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
        
        x = functional.leaky_relu(self.fc1(x))
        self.layer_activations.append(x.detach().numpy())
        
        x = functional.leaky_relu(self.fc2(x))
        self.layer_activations.append(x.detach().numpy())
        
        x = functional.leaky_relu(self.fc3(x))
        self.layer_activations.append(x.detach().numpy())
        
        x = functional.leaky_relu(self.fc4(x))
        self.layer_activations.append(x.detach().numpy())
        
        x = self.fc5(x)
        self.layer_activations.append(x.detach().numpy())
        
        return x

class WolfNet(nn.Module):
    def __init__(self):
        super(WolfNet, self).__init__()
        self.fc1 = nn.Linear(7, 15)
        self.fc2 = nn.Linear(15, 15)
        self.fc3 = nn.Linear(15, 15)
        self.fc4 = nn.Linear(15, 7)
        self.fc5 = nn.Linear(7, 2)
        self.layer_activations = []
        
    def forward(self, x):
        self.layer_activations = []
        
        # Record input as first layer activation
        self.layer_activations.append(x.detach().numpy())
        
        x = functional.leaky_relu(self.fc1(x), negative_slope=0.1)
        self.layer_activations.append(x.detach().numpy())
        
        x = functional.leaky_relu(self.fc2(x), negative_slope=0.1)
        self.layer_activations.append(x.detach().numpy())
        
        x = functional.leaky_relu(self.fc3(x), negative_slope=0.1)
        self.layer_activations.append(x.detach().numpy())
        
        x = functional.leaky_relu(self.fc4(x), negative_slope=0.1)
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
    global all_rabbit_positions, all_wolf_positions, all_noisy_positions

    # Reset networks
    rabbit_net = RabbitNet()
    wolf_net = WolfNet()

    # Reset optimizers
    opt_rabbit = torch.optim.Adam(rabbit_net.parameters(), lr=1e-3)
    opt_wolf = torch.optim.Adam(wolf_net.parameters(), lr=1e-3)

    # Reset positions
    current_rabbit_pos = torch.zeros(2)
    current_wolf_pos = torch.zeros(2)

    # Reset position history
    all_rabbit_positions = []
    all_wolf_positions = []
    all_noisy_positions = []

    print("Models and positions reset to initial state")

def get_noisy_observation(rabbit_pos):
    angle = torch.rand(1) * 2 * math.pi
    radius = torch.rand(1).sqrt()  # Uniform in circle
    offset = torch.tensor([torch.cos(angle), torch.sin(angle)]) * radius
    return rabbit_pos + offset

def FindSigmoidOf(x):
    x = max(min(x, 100), -100)
    sig = 1 / (1 + math.exp(-abs(x)))
    return sig if x >= 0 else -sig

# Improved reward system
def calculate_rabbit_rewards(rabbit_positions, wolf_positions, noisy_rabbit_positions):
    """
    Enhanced rabbit rewards with multiple strategic factors:
    1. Distance-based reward (primary)
    2. Survival time bonus
    3. Movement efficiency penalty
    4. Boundary penalty
    5. Capture penalty with danger zones
    6. Strategic positioning bonus
    """
    rabbit_rewards = []

    # Calculate distances at each step
    distances = torch.norm(rabbit_positions - wolf_positions, dim=1)

    # Calculate movements
    rabbit_movements = torch.norm(rabbit_positions[1:] - rabbit_positions[:-1], dim=1)

    # Calculate distance changes
    distance_changes = distances[1:] - distances[:-1]

    # Calculate center of the environment (assume centered at origin)
    center = torch.zeros(2)
    distances_to_center = torch.norm(rabbit_positions - center, dim=1)

    # Calculate velocity alignment (how well rabbit moves away from wolf)
    rabbit_velocities = rabbit_positions[1:] - rabbit_positions[:-1]
    wolf_to_rabbit_vectors = rabbit_positions[:-1] - wolf_positions[:-1]

    for i in range(1, len(distances)):
        total_reward = 0.0

        # 1. Primary distance-based reward (scaled)
        distance_reward = distance_changes[i - 1] * 3.0
        total_reward += distance_reward

        # 2. Absolute distance bonus/penalty
        safe_distance_bonus = torch.sigmoid(distances[i] - 10.0) * 0.5  # Bonus when >10 units away
        danger_penalty = torch.sigmoid(5.0 - distances[i]) * -2.0  # Penalty when <5 units away
        total_reward += safe_distance_bonus + danger_penalty

        # 3. Survival reward (small positive for each step survived)
        survival_reward = 0.1
        total_reward += survival_reward

        # 4. Movement efficiency penalty
        if i - 1 < len(rabbit_movements):
            movement_penalty = rabbit_movements[i - 1] * 0.02
            total_reward -= movement_penalty

            # 5. Strategic movement bonus (reward moving directly away from wolf)
            if i - 1 < len(rabbit_velocities) and i - 1 < len(wolf_to_rabbit_vectors):
                if torch.norm(rabbit_velocities[i - 1]) > 0.1 and torch.norm(wolf_to_rabbit_vectors[i - 1]) > 0.1:
                    velocity_direction = rabbit_velocities[i - 1] / torch.norm(rabbit_velocities[i - 1])
                    escape_direction = wolf_to_rabbit_vectors[i - 1] / torch.norm(wolf_to_rabbit_vectors[i - 1])
                    alignment = torch.dot(velocity_direction, escape_direction)
                    strategic_bonus = max(0.0, alignment) * 0.5  # Bonus for moving away from wolf
                    total_reward += strategic_bonus

        # 6. Boundary penalty (encourage staying in reasonable area)
        boundary_penalty = torch.clamp(distances_to_center[i] - 50.0, min=0) * 0.05
        total_reward -= boundary_penalty

        # 7. Capture penalty (progressive penalty based on proximity)
        if distances[i] < 1.0:
            total_reward -= 20.0  # Large penalty if caught
        elif distances[i] < 5.0:
            total_reward -= 5.0  # Medium penalty when very close
        elif distances[i] < 20.0:
            total_reward -= 1.0  # Small warning penalty

        rabbit_rewards.append(total_reward)

    return torch.tensor(rabbit_rewards, dtype=torch.float32)


def calculate_wolf_rewards(rabbit_positions, wolf_positions, noisy_rabbit_positions):
    """
    Enhanced wolf rewards with multiple strategic factors:
    1. Distance-to-rabbit reward (primary)
    2. Distance-to-observation reward
    3. Movement efficiency penalty
    4. Progressive capture rewards
    5. Persistence bonus
    6. Prediction accuracy bonus
    """
    wolf_rewards = []

    distances = torch.norm(rabbit_positions - wolf_positions, dim=1)

    min_length = min(len(wolf_positions), len(noisy_rabbit_positions))
    distances_to_noisy = torch.norm(wolf_positions[:min_length] - noisy_rabbit_positions[:min_length], dim=1)

    wolf_movements = torch.norm(wolf_positions[1:] - wolf_positions[:-1], dim=1)

    # Calculate distance changes (positive when getting closer)
    distance_changes = distances[:-1] - distances[1:]

    # Calculate prediction accuracy
    prediction_errors = torch.norm(rabbit_positions[:min_length] - noisy_rabbit_positions[:min_length], dim=1)

    # Calculate wolf velocities and directions
    wolf_velocities = wolf_positions[1:] - wolf_positions[:-1]
    wolf_to_rabbit_vectors = rabbit_positions[1:] - wolf_positions[:-1]

    for i in range(1, min(len(distances), len(distances_to_noisy) + 1)):
        total_reward = 0.0

        # 1. Primary distance-based reward (scaled)
        if i - 1 < len(distance_changes):
            distance_reward = distance_changes[i - 1] * 4.0
            total_reward += distance_reward

        # 2. Absolute distance bonus
        proximity_bonus = torch.sigmoid(5.0 - distances[i]) * 2.0  # Bonus when close
        total_reward += proximity_bonus

        # 3. Observation-following reward
        if i - 1 < len(distances_to_noisy):
            if i > 1:
                noisy_distance_change = distances_to_noisy[i - 2] - distances_to_noisy[i - 1]
            else:
                noisy_distance_change = 0.1
            observation_reward = noisy_distance_change * 2.0
            total_reward += observation_reward

        # 4. Movement efficiency penalty
        if i - 1 < len(wolf_movements):
            movement_penalty = wolf_movements[i - 1] * 0.01
            total_reward -= movement_penalty

            # 5. Strategic movement bonus (reward moving directly toward rabbit)
            if i - 1 < len(wolf_velocities) and i - 1 < len(wolf_to_rabbit_vectors):
                if torch.norm(wolf_velocities[i - 1]) > 0.1 and torch.norm(wolf_to_rabbit_vectors[i - 1]) > 0.1:
                    velocity_direction = wolf_velocities[i - 1] / torch.norm(wolf_velocities[i - 1])
                    pursuit_direction = wolf_to_rabbit_vectors[i - 1] / torch.norm(wolf_to_rabbit_vectors[i - 1])
                    alignment = torch.dot(velocity_direction, pursuit_direction)
                    strategic_bonus = max(0.0, alignment) * 0.8  # Bonus for moving toward rabbit
                    total_reward += strategic_bonus

        # 6. Progressive capture rewards
        if distances[i] < 1.0:
            total_reward += 25.0  # Large reward for capture
        elif distances[i] < 5.0:
            total_reward += 8.0  # Medium reward for very close
        elif distances[i] < 20.0:
            total_reward += 2.0  # Small bonus for getting close

        # 7. Persistence bonus (reward for consistent pursuit)
        if i > 2 and i - 2 < len(distance_changes) and i - 1 < len(distance_changes):
            if distance_changes[i - 2] > 0 and distance_changes[i - 1] > 0:
                total_reward += 0.5  # Bonus for consecutive steps getting closer

        # 8. Prediction efficiency penalty
        if i - 1 < len(prediction_errors):
            prediction_penalty = prediction_errors[i - 1] * 0.1
            total_reward -= prediction_penalty

        wolf_rewards.append(total_reward)

    return torch.tensor(wolf_rewards, dtype=torch.float32)

# Training function with improved reward system
def train_step(num_steps=100):
    global rabbit_positions, wolf_positions, noisy_rabbit_positions, current_rabbit_pos, current_wolf_pos, NoisyDistance
    
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

    # if all_rabbit_positions != []:
    #     all_rabbit_positions = [all_rabbit_positions[-1]]

    all_rabbit_positions.append(rabbit_pos.detach().clone())
    all_wolf_positions.append(wolf_pos.detach().clone())

    rabbit_rewards, wolf_rewards = [0], [0]
    
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


        time.sleep(0.1)

        # Wolf's noisy observation - detach rabbit_pos to break gradient connection
        obs_rabbit = get_noisy_observation(rabbit_pos.detach())
        noisy_rabbit_positions.append(obs_rabbit.detach().numpy().copy())
        all_noisy_positions.append(obs_rabbit.detach().clone())

        # Wolf's move
        NowDistance = 0.0
        OldDistance = 0.0

        DPos = rabbit_pos - wolf_pos
        RabbitDPos = [0, 0]
        CaptureAngle = 0.0

        if len(all_wolf_positions) >= 2 and len(all_rabbit_positions) >= 2:
            NowDistance = torch.norm(all_rabbit_positions[-1] - all_wolf_positions[-1])
            OldDistance = torch.norm(all_rabbit_positions[len(all_wolf_positions) - 2] - all_wolf_positions[len(all_wolf_positions) - 2])
            RabbitDPos = all_rabbit_positions[-1] - all_rabbit_positions[-2]
            CaptureAngle = atan2(DPos[1] - RabbitDPos[1], DPos[0] - RabbitDPos[0])
        DPos = rabbit_pos - wolf_pos



        state_wolf = torch.cat([torch.tensor([FindSigmoidOf(DPos[0])]),
                                torch.tensor([FindSigmoidOf(DPos[1])]),
                                torch.tensor([FindSigmoidOf(NowDistance)]),
                                torch.tensor([FindSigmoidOf(OldDistance)]),
                                torch.tensor([FindSigmoidOf(RabbitDPos[0])]),
                                torch.tensor([FindSigmoidOf(RabbitDPos[1])]),
                                torch.tensor([math.sin(CaptureAngle)])])
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

        rabbit_tensor = torch.stack(all_rabbit_positions)
        wolf_tensor = torch.stack(all_wolf_positions)
        noisy_tensor = torch.stack(all_noisy_positions) if all_noisy_positions else torch.zeros(1, 2)

        rabbit_rewards = calculate_rabbit_rewards(rabbit_tensor, wolf_tensor, noisy_tensor)
        wolf_rewards = calculate_wolf_rewards(rabbit_tensor, wolf_tensor, noisy_tensor)


    # Update current positions for next training step
    current_rabbit_pos = rabbit_pos.detach().clone()
    current_wolf_pos = wolf_pos.detach().clone()

    # Convert to tensors
    rabbit_tensor = torch.stack(all_rabbit_positions)
    wolf_tensor = torch.stack(all_wolf_positions)
    noisy_tensor = torch.stack(all_noisy_positions) if all_noisy_positions else torch.zeros(1, 2)

    # Calculate rewards
    
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
    final_distance = torch.norm(all_rabbit_positions[-1] - all_wolf_positions[-1])
    
    print(f"Training step complete. Final distance: {final_distance:.2f}")
    print(f"Current positions | Rabbit: {current_rabbit_pos}, Wolf: {current_wolf_pos}|")
    # После каждого обучения выводи статистику весов волка:
    # print("Wolf network - Weight statistics:")
    # for name, param in wolf_net.named_parameters():
    #     if 'weight' in name:
    #         print(f"{name}: min={param.data.min():.3f}, max={param.data.max():.3f}, mean={param.data.mean():.3f}")
    #