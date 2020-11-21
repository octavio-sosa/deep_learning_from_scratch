import numpy as np
from copy import deepcopy
from typing import Tuple
from nn import NeuralNetwork
from components.optimizer import Optimizer, SGD

def permute_data(X, Y):
    perm = np.random.permutation(X.shape[0])
    return X[perm], Y[perm]

class Trainer():
    def __init__(self, net: NeuralNetwork, optim: Optimizer):
        self.net = net
        self.optim = optim
        self.least_loss = 1e9
        setattr(self.optim, 'net', self.net)

    def yield_batches(self, X_train: np.ndarray, Y_train: np.ndarray,
                         batch_size: int = 32)\
                         -> Tuple[np.ndarray]:
        assert X_train.shape[0] == Y_train.shape[0],\
        '''
        observation and target data must have the same number of rows, instead
        observation data has {0} and target data has {1}
        '''.format(X_train.shape[0], Y_train.shape[0]) 

        n_observations = X_train.shape[0]

        for i in range(0, n_observations, batch_size):
            X_batch, Y_batch = X_train[i:i+batch_size], Y_train[i:i+batch_size]
            yield X_batch, Y_batch

    def train(self, X_train: np.ndarray, Y_train: np.ndarray,
              X_test: np.ndarray, Y_test: np.ndarray,
              epochs: int=100, eval_period: int=10,
              batch_size: int=32, seed: int=1,
              reset: bool=True):

        np.random.seed(seed)
        if reset:
            for layer in self.net.layers:
                layer.setup = True
            self.least_loss = 1e9 #reset 

        for epoch in range(epochs):
            if (epoch+1) % eval_period==0:
                #save model
                latest_model = deepcopy(self.net)

            X_train, Y_train = permute_data(X_train, Y_train)
            batches_yielded = self.yield_batches(X_train, Y_train, batch_size)

            for _ , (X_batch, Y_batch) in enumerate(batches_yielded):
                self.net.train_batch(X_batch, Y_batch)
                self.optim.update_params()

            if (epoch+1) % eval_period==0:
                test_preds = self.net.forward(X_test)
                loss = self.net.loss.forward(test_preds, Y_test)

                if loss < self.least_loss:
                    print(f'Validation loss after {epoch+1} epochs is {loss:.3f}')
                    self.least_loss = loss
                else:
                    print(f'\nLoss increased after {epoch+1} epochs.')
                    print(f'The model from epoch {epoch+1-eval_period} gave the least loss.')
                    # update optimizer with latest NN model
                    self.net = latest_model
                    setattr(self.optim, 'net', self.net)
                    break # halt training

                
        print() 
