from gymnasium.spaces import OneOf
from utils.abstract_trainer import RLTrainer
import gymnasium as gym
from collections import deque
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
            )
    
    def forward(self, x):
        return self.model(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        samples = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.bool_)
        )
    
    def __len__(self):
        return len(self.memory)

class FrozenLakeDeepQAgent:
    def __init__(self, alpha, gamma, epsilon):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.env = gym.make("FrozenLake-v1", map_name="8x8")
        self.state_dim = 64  # 8x8 grid has 64 states
        self.action_dim = 4  # 4 possible actions
        
        # Q-network and target network
        self.q_net = QNetwork(self.state_dim, self.action_dim)
        self.target_net = QNetwork(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        
        # Optimizer and replay memory
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=alpha)
        self.memory = ReplayMemory(capacity=10000)
        
        # Training parameters
        self.batch_size = 64
        self.target_update_freq = 10  # Update target network every 10 episodes
        self.episode_count = 0
        
    def preprocess_state(self, state):
        """Convert state integer to one-hot vector"""
        one_hot = np.zeros(self.state_dim)
        one_hot[state] = 1.0
        return one_hot

    def get_action(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            state_tensor = torch.tensor(
                self.preprocess_state(state), 
                dtype=torch.float32
            ).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            return q_values.argmax().item()

    def update_model(self):
        """Update Q-network using experience replay"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from replay memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)
        
        # Compute current Q-values
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (~dones)
        
        # Compute loss and optimize
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def epoch(self):
        """Run one training episode"""
        state, info = self.env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Select and execute action
            action = self.get_action(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store experience in replay memory
            self.memory.push(
                self.preprocess_state(state),
                action,
                reward,
                self.preprocess_state(next_state),
                done
            )
            
            # Update model
            self.update_model()
            
            state = next_state
            total_reward += reward
        
        # Update target network periodically
        if self.episode_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(0.01, self.epsilon * 0.995)
        self.episode_count += 1
        return total_reward



class DeepQTrainer(RLTrainer):
    def __init__(self):
        super().__init__("deepQ")
        self.agent =  FrozenLakeDeepQAgent(0.8, 0.7, 0.1)
        
    def epoch(self):
        self.agent.epoch()


trainer = DeepQTrainer()
trainer.train(10000)
































