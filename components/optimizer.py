class Optimizer():
    '''
    Base optimizer
    '''
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
    
    def update_params(self):
        '''
        Interface for updating params
        '''
        raise NotImplementedError()

class SGD(Optimizer):
    '''
    Stochastic Gradient Descent
    '''
    def __init__(self, learning_rate: float = 0.01):
        super().__init__(learning_rate)

    def update_params(self):
    '''
    Network params are updated directly (yielded by reference)
    Self.net attribute is set in Trainer init
    '''
        for (param, param_grad) in zip(self.net.yield_params(),
                                       self.net.yield_param_grads()):
            param -= self.learning_rate*param_grad
