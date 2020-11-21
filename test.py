import numpy as np
import data_processor as dp
from nn import NeuralNetwork
from components.layers import Dense 
from components.operations import Sigmoid, Linear
from components.loss import MeanSquaredError
from trainer import Trainer
from components.optimizer import Optimizer, SGD
import matplotlib.pyplot as plt

def data():
    X_train, X_test, Y_train, Y_test = dp.get_data()
    data = {'X_train': X_train, 'X_test': X_test,\
            'Y_train': Y_train, 'Y_test': Y_test}

    for key in data.keys():
        print(f'{key}: {data[key].shape}')

def train():
    X_train, X_test, Y_train, Y_test = dp.get_data()

    deep_net = NeuralNetwork(layers=[Dense(n_neurons=13, activation=Sigmoid()),
                                     Dense(n_neurons=13, activation=Sigmoid()),
                                     Dense(n_neurons=1, activation=Linear())],
                             loss=MeanSquaredError(), seed=80718)

    trainer = Trainer(deep_net, SGD(learning_rate=0.01))
    trainer.train(X_train, Y_train, X_test, Y_test,
                  epochs=1_000, eval_period=100, batch_size=23,
                  seed=80718)

def test():
    #data()
    print()
    train()

test()
