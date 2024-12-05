import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from vizdoom import DoomGame, ScreenResolution, Mode
import cv2
import random
from collections import deque

# Initialize the Doom Environment
def initialize_doom():
    game = DoomGame()
    game.load_config(r"D:\github_ML and DL\ML-and-DL\doom_playing_AI\basic.cfg")  # Use your .cfg file here
    game.set_doom_scenario_path(r"D:\github_ML and DL\ML-and-DL\doom_playing_AI\basic.wad")  # Use your scenario file
    game.set_screen_resolution(ScreenResolution.RES_160X120)
    game.set_window_visible(False)  # Set to True for visualization
    game.set_mode(Mode.PLAYER)
    game.init()
    return game

def preprocess_frame(frame):
    if frame is None:
        raise ValueError("Error: Received an empty frame for preprocessing.")
    frame = cv2.resize(frame, (80, 80))  # Resize to 80x80
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    frame = frame / 255.0  # Normalize pixel values
    return frame

# Replay memory for experience replay
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# CNN Model
class CNN(nn.Module):
    def __init__(self, number_actions):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2)
        self.fc1 = nn.Linear(64 * 9 * 9, 128)  # Adjust based on input shape
        self.fc2 = nn.Linear(128, number_actions)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 3, 2))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Softmax-based action selection
class SoftmaxBody(nn.Module):
    def __init__(self, temperature):
        super(SoftmaxBody, self).__init__()
        self.T = temperature

    def forward(self, outputs):
        probs = F.softmax(outputs * self.T, dim=1)
        actions = probs.multinomial(1)
        return actions

# AI Agent
class AI:
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body

    def __call__(self, state):
        state = Variable(torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        output = self.brain(state)
        action = self.body(output)
        return action.data.numpy()[0][0]

# Initialize Doom and AI
doom_env = initialize_doom()
number_actions = doom_env.get_available_buttons_size()
cnn = CNN(number_actions)
softmax_body = SoftmaxBody(temperature=1.0)
ai = AI(brain=cnn, body=softmax_body)

# Experience replay
memory = ReplayMemory(capacity=10000)
optimizer = optim.Adam(cnn.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Training Loop
nb_epochs = 30
batch_size = 64
gamma = 0.99

for episode in range(10):
    doom_env.new_episode()
    while not doom_env.is_episode_finished():
        state = doom_env.get_state()
        if state and state.screen_buffer is not None:
            frame = preprocess_frame(state.screen_buffer)
        else:
            print("Warning: Invalid state or screen_buffer is None.")
            break


for epoch in range(1, nb_epochs + 1):
    game_score = 0
    doom_env.new_episode()
    state = preprocess_frame(doom_env.get_state().screen_buffer)
    if state and state.screen_buffer is not None:
        frame = state.screen_buffer
        print(f"Frame shape: {frame.shape}")  # Check the shape of the frame
    else:
        print("Error: screen_buffer is None.")

    while not doom_env.is_episode_finished():
        action = ai(state)
        reward = doom_env.make_action([int(action)])
        done = doom_env.is_episode_finished()

        next_state = preprocess_frame(doom_env.get_state().screen_buffer) if not done else None
        memory.push((state, action, reward, next_state))

        state = next_state
        game_score += reward

        if len(memory) >= batch_size:
            batch = memory.sample(batch_size)
            states, actions, rewards, next_states = zip(*batch)

            states = Variable(torch.tensor(states, dtype=torch.float32))
            actions = Variable(torch.tensor(actions, dtype=torch.long))
            rewards = Variable(torch.tensor(rewards, dtype=torch.float32))
            next_states = Variable(torch.tensor([ns for ns in next_states if ns is not None], dtype=torch.float32))

            # Compute targets
            current_q_values = cnn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            max_next_q_values = cnn(next_states).max(1)[0]
            max_next_q_values = torch.cat([max_next_q_values, torch.zeros(batch_size - len(next_states))])
            targets = rewards + (gamma * max_next_q_values)

            loss = loss_fn(current_q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(f"Epoch {epoch}, Score: {game_score}")

doom_env.close()
