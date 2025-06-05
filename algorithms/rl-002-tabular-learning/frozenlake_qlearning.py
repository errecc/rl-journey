"""
Implementation of Q Learning algorithm for frozenlake enviroment
"""
from utils.abstract_trainer import RLTrainer
import gymnasium as gym
import torch
import numpy as np
import random


class FrozenQAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        """Define single Q learning agent"""
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.done = False
        # Define basic q parameters
        self.env = gym.make("FrozenLake-v1")
        self.env.reset()
        # Define Q Matrix
        self.observation_size = self.env.observation_space.n 
        self.num_actions = self.env.action_space.n
        self.Q = torch.randn([self.observation_size, self.num_actions])

    def epsilon_action(self,action):
        rand = random.random()
        if(rand < self.epsilon):
            act = self.env.action_space.sample()
            observation, reward, done, truncated, info = self.env.step(act)
            return observation, reward, done, truncated, info
        act = action
        observation, reward, done, truncated, info = self.env.step(act)
        return observation, reward, done, truncated, info

    def epoch(self):
        """
        Perform Q Learning basic
        """
        act = self.env.action_space.sample()
        while not self.done:
    




class FrozenLakeQTrainer(RLTrainer):
    def __init__(self):
        """Define the Q-learning trainer for frozenAgent"""
        ...

    def epoch(self):
        """Define single epoch step for training rl agent"""
        ...


