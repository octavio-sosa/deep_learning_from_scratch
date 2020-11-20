import numpy as np
from components.loss import Loss
from components.layers import Layer

class NeuralNetwork():
    def __init__(self, layers: List[Layer], loss: Loss, seed: int=1):
        self.layers = layers
        self.loss = loss
        self.seed = seed
        if seed:
            for layer in self.layers:
                setattr(layer, 'seed', self.seed)
