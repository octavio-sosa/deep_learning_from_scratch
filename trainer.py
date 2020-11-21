import numpy as np
from copy import deepcopy
from typing import Tuple
from nn import NeuralNetwork
from components.optimizer import Optimizer, SGD

class Trainer():
    def __init__(self, net: NeuralNetwork, optim: Optimizer):
        self.net = net
        self.optim = optim
        self.best_loss = 1e9
        setattr(self.optim, 'net', self.net)
