import numpy as np
from typing import List

class Layer():
    '''
    Base layer
    '''
    def __init__(self, n_neurons: int):
    self.n_neurons = n_neurons
    self.first = True
    self.params: List[np.ndarray] = []
    self.param_grads: List[np.ndarray] = []
    self.operations: List[Operation] = []
