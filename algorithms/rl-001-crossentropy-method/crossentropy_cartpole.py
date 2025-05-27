from utils.abstract_trainer import RLTrainer
import gymnasium as gym
import torch
import numpy as np


class CrossEntropyNetwork(torch.nn.Module):
    def __init__(self, obs_size, hidden_size, num_actions):
        super().__init__()
        self.model = torch.nn.Sequential(
                torch.nn.Linear(obs_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, num_actions)
                )

    def forward(self, x):
        return self.model(x)


def eval_policy(policy, env):
    """
    Evaluate the reward of a single policy or network
    """
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        tensor_state = torch.FloatTensor(state)
        logits = policy(tensor_state)
        action = np.argmax(logits.detach().numpy())
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if(truncated):
            break
        state = next_state
    return total_reward


class CrossEntropyCartpoleTrainer(RLTrainer):
    def __init__(self, collection, hidden_size, population_size, elite_prop = 0.1, noise = 0.1):
        super().__init__(collection)
        self.noise = noise
        self.hidden_size = hidden_size
        self.population_size = population_size
        self.elite_prop = elite_prop
        self.env = gym.make("CartPole-v1")
        self.num_actions = self.env.action_space.n 
        self.obs_size = len(self.env.observation_space.sample())
        self.policy = CrossEntropyNetwork(self.obs_size, self.hidden_size, self.num_actions)
        self.best_reward = -float("inf")

    def epoch(self):
        population = []
        for _ in range(self.population_size):
            perturbed = CrossEntropyNetwork(self.obs_size, self.hidden_size, self.num_actions)
            perturbed.load_state_dict(self.policy.state_dict())
            # Add Gaussian noise to parameters
            with torch.no_grad():
                for param in perturbed.parameters():
                    param.add_(torch.randn_like(param) * self.noise)
            population.append(perturbed)
        rewards = []
        for policy in population:
            reward = eval_policy(policy, self.env)
            rewards.append(reward)
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_policy = policy
        elite_count = int(self.population_size * self.elite_prop)
        elite_indices = np.argsort(rewards)[-elite_count:]
        elites = [population[i] for i in elite_indices]
        new_state_dict = {}
        for key in self.policy.state_dict():
            new_state_dict[key] = torch.stack(
                [elite.state_dict()[key] for elite in elites]
            ).mean(dim=0)
        self.policy.load_state_dict(new_state_dict)
        print(np.max(rewards))
        return np.max(rewards)

    def get_best_policy(self):
        return self.best_policy

    def close(self):
        self.env.close()
