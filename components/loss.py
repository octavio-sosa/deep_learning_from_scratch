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



