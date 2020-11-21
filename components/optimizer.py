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

