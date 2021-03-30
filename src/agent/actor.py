
from math import sqrt
from random import choice
import sys
import numpy as np
import torch
import torch.nn as nn
from environment.hexagonal_grid import HexagonalGrid
from environment.universal_action import UniversalAction
from environment.universal_state import UniversalState


class Actor(nn.Module):
    def __init__(self, epochs: int, learning_rate: float, save_interval: int, input_neurons: int, nn_dimensions: list, activation_functions: list) -> None:
        super(Actor, self).__init__()
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.save_interval = save_interval

        # Input layer
        network_config = [nn.Linear(input_neurons, nn_dimensions[0])]
        print(input_neurons, nn_dimensions[0])
        network_config.append(nn.Dropout(0.5))
        network_config.append(nn.ReLU())

        # Hidden layers
        for layer_index in range(len(nn_dimensions) - 2):
            print(nn_dimensions[layer_index], nn_dimensions[layer_index + 1])
            network_config.append(nn.Linear(nn_dimensions[layer_index], nn_dimensions[layer_index + 1]))

        # Output layer
        print(nn_dimensions[-2], nn_dimensions[-1])
        network_config.append(nn.Linear(nn_dimensions[-2], nn_dimensions[-1]))
        network_config.append(nn.Softmax(-1))
        self.model = nn.Sequential(*network_config)
        self.optimizer = torch.optim.Adagrad(list(self.model.parameters()), lr=self.learning_rate)
        self.loss_function = nn.BCELoss(reduction='mean')

    def generate_action(self, state: UniversalState, legal_actions: list) -> UniversalAction:
        input = Actor.__to_tensor(state.to_numpy())

        prediction = self.inference(input).data.numpy()

        all_nodes = state.nodes.keys()

        nodes = [1 if node in legal_actions else 0 for node in all_nodes]

        for node_index in range(len(nodes)):
            nodes[node_index] *= prediction[node_index]

        nodes /= sum(nodes)

        #random_index = choice(list(enumerate(nodes)))[0]
        #random_action = UniversalAction((random_index // size, random_index % size))

        greedy_index = np.argmax(nodes)

        size = sqrt(len(all_nodes))
        action = UniversalAction((greedy_index // size, greedy_index % size))

        return action

    def train(self, input: np.ndarray, target: np.ndarray) -> tuple:
        x_train = Actor.__to_tensor(input)
        y_train = Actor.__to_tensor(target)

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
        print(f'Saved model ANET_{iterations}')
        torch.save(self.state_dict(), f'../models/ANET_{iterations}')

    def load_model(self, iterations: int):
        print(f'Loaded model ANET_{iterations}')
        self.load_state_dict(torch.load(f'../models/ANET_{iterations}'))

    @staticmethod
    def __to_tensor(data):
        return torch.FloatTensor(data)
