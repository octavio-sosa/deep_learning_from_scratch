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

    def yield_batches(self, X_train: np.ndarray, Y_train: np.ndarray,
                         batch_size: int = 32)\
                         -> Tuple[np.ndarray]:
        assert X_train.shape[0] == Y_train.shape[0],\
        '''
        observation and target data must have the same number of rows, instead
        observation data has {0} and target data has {1}
        '''.format(X_train.shape[0], y_train.shape[0]) 

        n_observations = X_train.shape[0]

        for i in range(0, n_observations, batch_size):
            X_batch, Y_batch = X_train[i:i+batch_size], Y_train[i:i+batch_size]
            yield X_batch, Y_batch
