import numpy as np
from assert import assert_same_shape

class Loss():
    def __init__(self):
        pass

    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        assert_same_shape(prediction, target)

        self.prediction = prediction
        self.target = target

        loss_val = self._f()
        return loss_val

    def backward(self) -> np.ndarray:
        '''
        Computes dLdP
        '''
        self.input_grad = self._get_input_grad()
        assert_same_shape(self.prediction, self.input_grad)
        return self.input_grad

    def 
