import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd

# let us make the neural network

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
    

# Experience replay
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        # zip *list is interesting *((state,action,reward), (state,action,reward))
        # output = (state,state), (action,action), (reward,reward)
        sample = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), sample)
    
# implementing deep q learning
class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size=input_size, nb_action=nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile=True))*100)  # temperature=7 so more like a car
        #if softmax ([1, 2, 3] * temperature)->[0.0, 0.02, 0.98]  so more sure
        # volatile will help in making gradient graph easily
        action = probs.multinomial(num_samples=1)
        return action.data[0, 0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        # Ensure batch_action is of type int64
        batch_action = batch_action.long()

        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()  # Update weights of the optimizer for nn
    # update weights of the optimizer for nn
    
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory)>100:
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action

    def score(self):
        return sum(self.reward_window) /(len(self.reward_window))+1
    
    def save(self):
        torch.save({"state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict,
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
    



