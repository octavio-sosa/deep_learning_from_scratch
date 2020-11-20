import numpy as np

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
        self.input_grad = self.get_input_grad(output_grad) 
        assert_same_shape(self.input_, self.input_grad)
        return self.input_grad
    
    
    def f(self) -> np.ndarray:
        '''
        Interface for computing forward func
        '''
        raise NotImplementedError()

    def get_input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        '''
        Interface for computing partial derivative w.r.t. input
        '''
        raise NotImplementedError()

class ParamOperation(Operation):
    '''
    Operation with model params as argument
    '''

    def __init__(self, param: np.ndarray):
        super().__init__() #inherit from super-class (Operation)
        self.param = param

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        assert_same_shape(self.output, output_grad)

        self.input_grad = self.get_input_grad(output_grad)
        self.param_grad = self.get_param_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)
        assert_same_shape(self.param, self.param_grad)

        return self.input_grad

    def get_param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        '''
        Interface for computing partial derivatives w.r.t. params
        '''
        raise NotImplementedError()

class WeightTransform(ParamOperation):
    '''
    Transformation of inputs using weights
    '''

    def __init__(self, W: np.ndarray):
        super().__init__(W)

    def f(self) -> np.ndarray:
        return self.input_.dot(self.param)

    def get_input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        # dLdOut . dOutdIn
        output_grad.dot(self.param.T)

    def get_param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        # dOutdP . dLdOut
        return (self.input_.T).dot(output_grad)

