import numpy as np
from typing import List
from components.operations import Operation, ParamOperation,\
                                  WeightTransform,\
                                  BiasAdd, Sigmoid
from components.assertions import assert_same_shape

class Layer():
    '''
    Base layer
    '''
    def __init__(self, n_neurons: int):
        self.n_neurons = n_neurons
        self.setup = True
        self.params: List[np.ndarray] = []
        self.param_grads: List[np.ndarray] = []
        self.operations: List[Operation] = []

    def _setup_layer(self, input_: np.ndarray):
        '''
        Interface
        '''
        raise NotImplementedError()

    def forward(self, input_: np.ndarray) -> np.ndarray:
        '''
        feed forward through all operations of layer
        '''
        if self.setup:
            self._setup_layer(input_)
            self.setup = False

        self.input_ = input_

        for operation in self.operations:
            input_ = operation.forward(input_)

        self.output = input_

        return self.output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        '''
        Applies chain rule from output-layer towards input-layer
        '''
        assert_same_shape(self.output, output_grad)
        
        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)

        input_grad = output_grad
        self._cache_param_grads()

        return input_grad

    def _cache_param_grads(self):
        '''
        Cache the parameter gradients in data member
        '''
        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _cache_params(self):
        '''
        Cache the updated parameters
        '''
        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)

class Dense(Layer):
    '''
    Fully connected layer
    '''
    def __init__(self, n_neurons: int, activation: Operation = Sigmoid()):
        super().__init__(n_neurons)
        self.activation = activation

    def _setup_layer(self, input_: np.ndarray):
        if self.seed:
            np.random.seed(self.seed)

        self.params = []

        # weights
        self.params.append(np.random.randn(input_.shape[1], self.n_neurons))

        # bias
        self.params.append(np.random.randn(1, self.n_neurons))

        self.operations = [WeightTransform(self.params[0]),
                           BiasAdd(self.params[1]),
                           self.activation]
