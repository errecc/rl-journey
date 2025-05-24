from utils.abstract_trainer import RLTrainer


class CrossEntropyCartpole(RLTrainer):
    def __init__(self, collection):
        super().__init__(collection)

    def step(self):
        ...

    def train(self):
        ...

    def plot_performance(self):
        ...


carpole_trainer = CrossEntropyCartpole("cartpole-crossentropy")
