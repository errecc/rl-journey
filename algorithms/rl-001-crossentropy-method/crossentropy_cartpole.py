from utils.abstract_trainer import RLTrainer
import gymnasium as gym


class CrossEntropyCartpole(RLTrainer):
    def __init__(self, collection):
        super().__init__(collection)

    def epoch(self):
        ...


carpole_trainer = CrossEntropyCartpole("cartpole-crossentropy")
