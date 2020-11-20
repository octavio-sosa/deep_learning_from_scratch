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

    def _f(self) -> float:
        '''
        Interface for loss function
        '''
        raise NotImplementedError()

    def _get_input_grad(self) -> np.ndarray:
        '''
        Interface for computing partial derivative of loss function w.r.t. predictions
        '''
        raise NotImplementedError()

class MeanSquaredError(Loss):
    def __init__(self):
        super().__init__() #pass

    def _f(self) -> float:
        loss = np.sum(np.power(self.prediction - self.target, 2))
                / self.prediction.shape[0]
        return loss

    def _get_input_grad(self) -> np.ndarray:
        return 2.0*(self.prediction - self.target) / self.prediction.shape[0]
