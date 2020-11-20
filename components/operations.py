class Operation():
    '''
    Base operation
    '''

    def __init__(self):
        pass

    def forward(self, input_: np.ndarray):
    '''
    '''
    self.input_ = input_
    self.output = self.f()

    return self.output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
    assert_same_shape(self.output, output_grad)



    
