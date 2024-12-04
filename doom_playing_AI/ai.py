import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import gym
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete

import experience_replay, image_preprocessing


# building the AI
class CNN(nn.Module):

    def count_neurons(self, image_dim): # just re use this code to get number of neurons after convolutions
        x = Variable(torch.rand(1, *image_dim))
        x = F.leaky_relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.leaky_relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.leaky_relu(F.max_pool2d(self.convolution3(x), 3, 2))

        return x.data.view(1, -1).size(1)


    def __init__(self, number_actions):
        super(CNN, self).__init__()
        self.convolution1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.convolution2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5)
        self.convolution3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
        self.fc1 = nn.Linear(in_features= count_neurons((1, 80, 80)), out_features = 64)
        self.fc2 = nn.Linear(in_features=64, out_features=number_actions)

    def forward(self, x):
        x = F.leaky_relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.leaky_relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.leaky_relu(F.max_pool2d(self.convolution3(x), 3, 2))
        x = x.view(x.size(0), -1) # trick to flatten convolutional network
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x # brain making complete

class SoftmaxBody(nn.Module):

    def __init__(self, temperature):
        super(SoftmaxBody, self).__init__()
        self.T = temperature
    
    def forward(self, outputs):
        probs = F.softmax(outputs * self.T)
        actions = probs.multinomial()
        return actions
    
    #body is ready 

class AI:
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body

    def __call__(self, inputs):
        input = Variable(torch.from_numpy(np.array(inputs, dtype=np.float32)))
        output = self.brain(input)
        actions = self.body(output)
        return actions.data.numpy() # it was a torch tensor now its numpy array
    
number_actions=7
cnn = CNN(number_actions)
softmaxbody = SoftmaxBody(T=1.0)
ai = AI(brain = cnn, body = softmaxbody)

n_steps = experience_replay.NStepProgress(env = doom_env, ai=ai, n_step=10)
memory = experience_replay.ReplayMemory(n_steps=n_steps)

def eligibility_trace(batch):
    gamma = 0.99
    inputs = []
    targets = []

    for series in batch:
        input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype=np.float32)))
        output = cnn(input)
        cumul_reward = 0.0 if series[-1].done else output[1].data.max()
        for step in reversed(series[:-1]):
            cumul_reward = step.reward + gamma * cumul_reward
        state = series[0].state
        target = output[0].data
        target[series[0].action] = cumul_reward
        inputs.append(state)
        targets.append(target)
    return torch.from_numpy(np.array(inputs, dtype=np.float32))


class MA: # moving average for 100 steps of the game    
    def __init__(self, size):
        self.list_of_rewards = []
        self.size = size
    def add(self, rewards):
        if isinstance(rewards, list): # if reward is a list
            self.list_of_rewards += rewards
        else: # if reward is a single element
            self.list_of_rewards.append(rewards)
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]
    
    def average(self):
        return np.mean(self.list_of_rewards)

ma = MA(100)


# training the AI
loss = nn.MSELoss()
optimizer = optim.Adam(cnn.parameters(), lr = 0.001)
nb_epochs = 100

for epoch in range(1, nb_epochs+1):
    memory.run_steps(200)
    for batch in memory.sample_batch(128):
        inputs, targets = eligibility_trace(batch=batch)
        inputs, targets = Variable(inputs), Variable(targets)

        predictions = cnn(inputs)
        loss_error = loss(predictions, targets)
        optimizer.zero_grad()
        loss_error.backward()
        optimizer.step()  # weights updated
    rewards_steps = n_steps.rewards_steps()
    ma.add(rewards_steps)
    avg_reward = ma.average()
    print(f"Epoch: {str(epoch)}, reward: {str(avg_reward)}")
