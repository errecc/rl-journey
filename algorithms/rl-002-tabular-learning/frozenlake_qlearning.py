"""
Implementation of Q Learning algorithm for frozenlake enviroment
"""
from utils.abstract_trainer import RLTrainer


class FrozenQAgent:
    def __init__(self):
        """Define single Q learning agent"""
        ...

class FrozenLakeQTrainer(RLTrainer):
    def __init__(self):
        """Define the Q-learning trainer for frozenAgent"""
        ...

    def epoch(self):
        """Define single epoch step for training rl agent"""
        ...
