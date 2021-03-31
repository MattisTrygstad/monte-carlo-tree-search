from torch import optim
import torch.nn as nn


def instantiate_activation_func(value: str) -> object:
    if value.lower() == 'linear':
        return None
    elif value.lower() == 'sigmoid':
        return nn.Sigmoid()
    elif value.lower() == 'tanh':
        return nn.Tanh()
    elif value.lower() == 'relu':
        return nn.ReLU()
    else:
        raise Exception('Invalid activation function')


def instantiate_optimizer(value: str, params: list, lr: float) -> object:
    if value.lower() == 'sgd':
        return optim.SGD(params, lr)
    elif value.lower() == 'adam':
        return optim.Adam(params, lr)
    elif value.lower() == 'adagrad':
        return optim.Adagrad(params, lr)
    elif value.lower() == 'rmsprop':
        return optim.RMSprop(params, lr)
    else:
        raise Exception('Invalid optimizer')
