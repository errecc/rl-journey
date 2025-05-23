from abc import ABC, abstractmethod
from database.database import DatabaseManager
import time


class RLTrainer(ABC):
    def __init__(self, collection: str):
        self.db = DatabaseManager(collection)
        self.rewards = []
        self.start_time = time.time()

    @abstractmethod
    def step(self):
        ...

    @abstractmethod
    def train(self, epochs):
        ...

    @abstractmethod
    def plot_performance(self):
        ...

    def finish(self):
        final_t = time.time()
        total_time = final_t - self.start_time
        print(f"whole process took {total_time/3600} hours")
        return total_time
