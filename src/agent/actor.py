
from copy import deepcopy
from math import sqrt
from random import choice
import sys
import numpy as np
import torch
import torch.nn as nn
from environment.hexagonal_grid import HexagonalGrid
from environment.universal_action import UniversalAction
from environment.universal_state import UniversalState
from utils.instantiate import instantiate_activation_func, instantiate_optimizer


class Actor(nn.Module):
    def __init__(self, epochs: int, learning_rate: float, save_interval: int, board_size: int, nn_dimensions: list, activation_functions: list, optimizer: str) -> None:
        super(Actor, self).__init__()
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.save_interval = save_interval

        # Input layer
        network_config = [nn.Linear(board_size**2 + 1, nn_dimensions[0])]
        # network_config.append(nn.Dropout(0.5))
        activation_function = instantiate_activation_func(activation_functions[0])
        network_config.append(activation_function) if activation_function else None

        # Hidden layers
        for layer_index in range(len(nn_dimensions) - 1):
            network_config.append(nn.Linear(nn_dimensions[layer_index], nn_dimensions[layer_index + 1]))

            activation_function = instantiate_activation_func(activation_functions[layer_index + 1])
            network_config.append(activation_function) if activation_function else None

        # Output layer
        network_config.append(nn.Linear(nn_dimensions[-1], board_size**2))
        network_config.append(nn.Softmax(-1))
        self.model = nn.Sequential(*network_config)
        self.model.apply(self.init_weights)
        self.optimizer = instantiate_optimizer(optimizer, list(self.model.parameters()), self.learning_rate)
        self.loss_function = nn.BCELoss(reduction='mean')

        # print(self.model)

    def generate_action(self, state: UniversalState, legal_actions: list) -> UniversalAction:
        input = Actor.__to_tensor(state.to_numpy())

        prediction = self.inference(input).data.numpy()

        all_nodes = state.nodes.keys()

        nodes = [1 if node in legal_actions else 0 for node in all_nodes]

        for node_index in range(len(nodes)):
            nodes[node_index] *= prediction[node_index]

        nodes /= sum(nodes)

        greedy_index = np.argmax(nodes)

        size = sqrt(len(all_nodes))
        action_coordinates = (greedy_index // size, greedy_index % size)

        assert action_coordinates in legal_actions
        action = UniversalAction(action_coordinates)

        return action

    def generate_probabilistic_action(self, state: UniversalState, legal_actions: list) -> UniversalAction:
        input = Actor.__to_tensor(state.to_numpy())

        prediction = self.inference(input).data.numpy()

        all_nodes = state.nodes.keys()

        nodes: list = [1 if node in legal_actions else 0 for node in all_nodes]

        for node_index in range(len(nodes)):
            nodes[node_index] *= prediction[node_index]

        nodes /= sum(nodes)

        index = np.random.choice([x for x in range(len(nodes))], p=nodes)

        size = sqrt(len(all_nodes))
        action_coordinates = (index // size, index % size)

        assert action_coordinates in legal_actions
        action = UniversalAction(action_coordinates)

        return action

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> tuple:
        x_train = Actor.__to_tensor(x_train)
        y_train = Actor.__to_tensor(y_train)

        for epoch_index in range(self.epochs):
            prediction = self.model(x_train)
            loss = self.loss_function(prediction, y_train)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        accuracy = prediction.argmax(dim=1).eq(y_train.argmax(dim=1)).sum().numpy() / len(y_train)

        return loss.item(), accuracy

    def inference(self, input):
        with torch.no_grad():
            return self.model(input)

    def save_model(self, iterations: int) -> None:
        print(f'\nSaved model ANET_{iterations}')
        torch.save(self.state_dict(), f'../models/ANET_{iterations}')

    def load_model(self, iterations: int):
        self.iterations = str(iterations)
        print(f'Loaded model ANET_{iterations}')
        self.load_state_dict(torch.load(f'../models/ANET_{iterations}'))

    @ staticmethod
    def __to_tensor(data):
        return torch.FloatTensor(data)

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
