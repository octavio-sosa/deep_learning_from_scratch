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

    def forward(self, x_batch: np.ndarray) -> np.ndarray:
        x = x_batch
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def backward(self, loss_grad: np.ndarray):
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train_batch(self, x_batch: np.ndarray, y_batch: np.ndarray) -> float:
        '''
        Forward and backward propagation
        '''

        preds = self.forward(x_batch)
        loss = self.loss.forward(preds, y_batch)
        dLdP = self.loss.backward()
        self.backward(dLdP)

        return loss

    def yield_params(self):
        '''
        Yields params from each Layer cache
        '''
        for layer in self.layers:
            yield from layer.params

    def yield_param_grads(self):
        '''
        Yields param gradients from each Layer cache
        '''
        for layer in self.layers:
            yield from layer.param_grads
