"""
Implementation of Q Learning algorithm for frozenlake enviroment
"""
from utils.abstract_trainer import RLTrainer
import gymnasium as gym
import torch
import numpy as np 
import random


class FrozenQAgent:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=1):
        """Define single Q learning agent"""
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.done = False
        # Define basic q parameters
        self.env = gym.make("FrozenLake-v1")
        # Define Q Matrix
        self.observation_size = self.env.observation_space.n 
        self.num_actions = self.env.action_space.n
        self.Q = np.zeros([self.observation_size, self.num_actions])

    def epoch(self):
        """
        Perform Q Learning basic
        """
        rewards = []
        for _ in range(1000):
            state, reward = self.env.reset()
            act = self.env.action_space.sample() 
            total_reward = 0 
            done = False
            while not done:
                rand = random.random()
                # perform epsilon exploration
                if(rand < self.epsilon):
                    act = self.env.action_space.sample()
                else:
                    act = np.argmax(self.Q[state, :])
                # perform action and update q values
                next_state, reward, done, truncated, info = self.env.step(act)
                current_q = self.Q[state, act]
                next_q_max = np.max(self.Q[next_state, :])
                new_q = current_q  + self.alpha * (reward + self.gamma * next_q_max - current_q)
                self.Q[state, act] = new_q
                total_reward += reward
                # update state
                state = next_state
            self.epsilon = max(0.001, self.epsilon * 0.99)
            rewards.append(total_reward)
        print(np.mean(rewards))
        return np.mean(rewards)


class FrozenLakeQTrainer(RLTrainer):
    def __init__(self, collection):
        """Define the Q-learning trainer for frozenAgent"""
        super().__init__(collection)
        self.agent = FrozenQAgent()

    def epoch(self):
        """Define single epoch step for training rl agent"""
        reward = self.agent.epoch()
        return reward


trainer = FrozenLakeQTrainer("qlearning_frozenlake")
trainer.train(100_000)
