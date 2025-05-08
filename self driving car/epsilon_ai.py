import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Neural Network for DQN
class Network(nn.Module):
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, nb_action)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

# Experience Replay Buffer
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        sample = zip(*random.sample(self.memory, batch_size))
        return [Variable(torch.cat(x, 0)) for x in sample]

# DQN Agent
class Dqn():
    def __init__(self, input_size, nb_action, gamma, epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.995, target_update_freq=100):
        self.gamma = gamma
        self.nb_action = nb_action
        self.reward_window = []
        self.model = Network(input_size=input_size, nb_action=nb_action)
        self.target_model = Network(input_size=input_size, nb_action=nb_action)
        self.target_model.load_state_dict(self.model.state_dict())  # Initialize target network
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        self.epsilon = epsilon_start  # Initial exploration rate
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        self.epsilon_decay = epsilon_decay  # Decay factor for epsilon
        self.target_update_freq = target_update_freq
        self.steps = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            # Exploration: choose a random action
            action = random.randint(0, self.nb_action - 1)
        else:
            # Exploitation: choose the best action based on the Q-values
            with torch.no_grad():
                q_values = self.model(state)
                action = torch.argmax(q_values).item()
        return action

    def update_epsilon(self):
        # Decay epsilon after every update
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        # Update target model weights periodically
        if self.steps % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def calculate_reward(self, state, map_size):
        # Example of penalizing proximity to the edges
        x, y = state
        edge_penalty = 0

        if x == 0 or x == map_size[0] - 1 or y == 0 or y == map_size[1] - 1:
            edge_penalty = -1  # Negative reward for being near the edge

        return edge_penalty

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        batch_action = batch_action.long()

        # Correcting the index for gather without subtracting 1
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        
        # Next Q-values from the target network
        next_outputs = self.target_model(batch_next_state).detach().max(1)[0]
        
        # Calculate the target
        target = self.gamma * next_outputs + batch_reward
        
        # Compute the TD loss
        td_loss = F.smooth_l1_loss(outputs, target)
        
        # Zero the gradients, backpropagate and update weights
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()
  # Update weights of the optimizer for nn

    def update(self, reward, new_signal, map_size):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        reward += self.calculate_reward(new_signal, map_size)  # Add edge penalty to the reward
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.update_epsilon()
        self.update_target_network()
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        self.steps += 1
        return action

    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1)
    
    def save(self):
        torch.save({"state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    }, "last_brain.pth")
        
    def load(self):
        if os.path.isfile("last_brain.pth"):
            print("loading...")
            checkpoint = torch.load("last_brain.pth")
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            print("Done")
        else:
            print("No checkpoints")

# Example usage of the DQN agent:
if __name__ == "__main__":
    # Example environment parameters
    map_size = (10, 10)  # 10x10 map (for example)
    input_size = 2  # Example: (x, y) coordinates
    nb_action = 4  # Example: up, down, left, right

    # Initialize DQN agent
    dqn_agent = Dqn(input_size=input_size, nb_action=nb_action, gamma=0.99)

    # Dummy environment loop for demonstration
    for episode in range(1000):  # Example number of episodes
        state = [random.randint(0, map_size[0] - 1), random.randint(0, map_size[1] - 1)]  # Random initial state
        for t in range(100):  # Max timesteps per episode
            action = dqn_agent.update(reward=0, new_signal=state, map_size=map_size)  # Dummy reward for now
            # Here, you would interact with the environment to get the next state and reward
            # For this example, we're just moving randomly
            state = [random.randint(0, map_size[0] - 1), random.randint(0, map_size[1] - 1)]
        
        print(f"Episode {episode}: Score = {dqn_agent.score()}")

    # Save trained model
    dqn_agent.save()
