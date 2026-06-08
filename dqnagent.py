from collections import deque
import torch.nn as nn
import torch
import random
import numpy as np
from dqnetwork import DQNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPSILON_START = 1.0
EPSILON_MIN = 0.01
MEMORY_SIZE = 100000
TRAIN_START = 1000


class DQNAgent:
    def __init__(
        self,
        input_size,
        output_size,
        hidden_sizes,
        batch_size,
        lr,
        gamma,
        epsilon_decay,
        target_update_steps,
        epsilon_start=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        memory_size=MEMORY_SIZE,
        train_start=TRAIN_START,
    ):
        self.output_size = output_size
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.gamma = gamma
        self.train_start = train_start
        self.target_update_steps = target_update_steps
        self.memory = deque(maxlen=memory_size)
        self.total_steps = 0
        self.model = DQNetwork(input_size, output_size, hidden_sizes).to(device)
        self.target_model = DQNetwork(input_size, output_size, hidden_sizes).to(device)
        self.update_target_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_function = nn.SmoothL1Loss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.output_size)
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()

    def replay(self):
        if len(self.memory) < self.train_start:
            return None
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = (
            torch.FloatTensor(np.array(dones, dtype=np.float32)).unsqueeze(1).to(device)
        )

        q_values = self.model(states).gather(1, actions)
        next_actions = self.model(next_states).argmax(1).unsqueeze(1)
        next_q_values = self.target_model(next_states).gather(1, next_actions).detach()
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_function(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.total_steps += 1
        if self.total_steps % self.target_update_steps == 0:
            self.update_target_model()

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
