import numpy as np
from typing import List
import operations as op

class Layer():
    '''
    Base layer
    '''
    def __init__(self, n_neurons: int):
    self.n_neurons = n_neurons
    self.not_setup = True
    self.params: List[np.ndarray] = []
    self.param_grads: List[np.ndarray] = []
    self.operations: List[Operation] = []

    def _setup_layer(self, input_: np.ndarray)
        '''
        Interface
        '''
        raise NotImplementedError()

    def forward(self, input_: np.ndarray) -> np.ndarray:
    '''
    feed forward through all operations of layer
    '''
    if self.not_setup:
        self._setup_layer(input_)
        self.not_setup = False

    self.input_ = input_

    for operation in self.operations:
        input_ = operation.forward(input_)

    self.output = input_

    return self.output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
    '''
    Applies chain rule from output-layer towards input-layer
    '''
    op.assert_same_shape(self.output, output_grad)
    
    for operation in reversed(self.operations):
        output_grad = operation.backward(output_grad)

    input_grad = output_grad
    self._cache_param_grads()

    return input_grad
