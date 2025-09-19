import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math


# нейросеть и слои
class Net(nn.Module):
   def __init__(self):
       super(Net, self).__init__()
       self.fc1 = nn.Linear(5, 10)
       self.fc2 = nn.Linear(10, 10)
       self.fc3 = nn.Linear(10, 10)
       self.fc4 = nn.Linear(10, 6)
       self.fc5 = nn.Linear(6, 2)

   def forward(self, x):
       x = F.relu(self.fc1(x))
       x = F.relu(self.fc2(x))
       x = self.fc3(x)
       return F.log_softmax(x)

net = Net()
print(net)

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

criterion = nn.NLLLoss()

def get_noisy_observation(rabbit_pos):
    angle = torch.rand(1) * 2 * math.pi
    radius = torch.rand(1).sqrt() * 1.0  # Uniform in circle
    offset = torch.tensor([torch.cos(angle), torch.sin(angle)]) * radius
    return rabbit_pos + offset


# тренировочный цикл:

def train(rabbit_model, wolf_model, episodes=10000, lr=1e-3):
    opt_rabbit = torch.optim.Adam(rabbit_model.parameters(), lr=lr)
    opt_wolf = torch.optim.Adam(wolf_model.parameters(), lr=lr)

    for episode in range(episodes):
        # Initialize positions and history
        rabbit_pos = torch.zeros(2)
        wolf_pos = torch.zeros(2)
        rabbit_log_probs = []
        wolf_log_probs = []

        for step in range(100):
            # Rabbit's move
            state_rabbit = torch.cat([rabbit_pos, wolf_pos, torch.tensor([step / 100.0])])
            action_mean = rabbit_model(state_rabbit)
            action_mean = action_mean / torch.norm(action_mean)
            noise = torch.randn(2) * 0.1
            action_rabbit = action_mean + noise
            action_rabbit = action_rabbit / torch.norm(action_rabbit)
            log_prob_rabbit = -0.5 * torch.sum(noise ** 2)  # Gaussian log-likelihood

            rabbit_pos = rabbit_pos + action_rabbit
            rabbit_log_probs.append(log_prob_rabbit)

            # Wolf's noisy observation
            obs_rabbit = get_noisy_observation(rabbit_pos)

            # Wolf's move
            state_wolf = torch.cat([wolf_pos, obs_rabbit, torch.tensor([step / 100.0])])
            action_mean_wolf = wolf_model(state_wolf)
            action_mean_wolf = action_mean_wolf / torch.norm(action_mean_wolf)
            noise_wolf = torch.randn(2) * 0.1
            action_wolf = action_mean_wolf + noise_wolf
            action_wolf = action_wolf / torch.norm(action_wolf)
            log_prob_wolf = -0.5 * torch.sum(noise_wolf ** 2)

            wolf_pos = wolf_pos + action_wolf
            wolf_log_probs.append(log_prob_wolf)

        # Check win condition
        distance = torch.norm(rabbit_pos - wolf_pos)
        rabbit_reward = 1.0 if distance >= 1.0 else 0.0
        wolf_reward = 1.0 - rabbit_reward

        # Policy gradient update for rabbit
        rabbit_loss = -torch.stack(rabbit_log_probs).sum() * rabbit_reward
        opt_rabbit.zero_grad()
        rabbit_loss.backward()
        opt_rabbit.step()

        # Policy gradient update for wolf
        wolf_loss = -torch.stack(wolf_log_probs).sum() * wolf_reward
        opt_wolf.zero_grad()
        wolf_loss.backward()
        opt_wolf.step()