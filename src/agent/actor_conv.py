from collections import OrderedDict
from math import sqrt
from random import choice
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from environment.state_manager import StateManager
from environment.universal_action import UniversalAction
from environment.universal_state import UniversalState
from utils.bridges import generate_bridge_neighbors
from utils.config_parser import Config
from utils.instantiate import instantiate_activation_func, instantiate_optimizer


class Actor(nn.Module):
    def __init__(self, epochs: int, learning_rate: float, save_interval: int, board_size: int, nn_dimensions: list, activation_functions: list, optimizer: str) -> None:
        super(Actor, self).__init__()
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.save_interval = save_interval
        self.board_size = board_size

        self.env = StateManager(visual=True)

        edges = self.env.state.edges

        neighbors = {}  # Key: (row, col), value: [(row, col), ...]
        for (t, f) in edges:
            # TO
            if t in neighbors:
                if f not in neighbors[t]:
                    lis = neighbors[t]
                    lis.append(f)
                    neighbors[t] = lis
            else:
                neighbors[t] = [f]

            # FROM
            if f in neighbors:
                if t not in neighbors[f]:
                    lis = neighbors[f]
                    lis.append(t)
                    neighbors[f] = lis
            else:
                neighbors[f] = [t]
        self.neighbors = neighbors
        self.bridge_neighbors = generate_bridge_neighbors(self.board_size, neighbors)
        print(self.bridge_neighbors)

        layers = OrderedDict([
            ('0', nn.ZeroPad2d(2)),
            ('1', nn.Conv2d(9, nn_dimensions[0], 3)),
            ('2', instantiate_activation_func(activation_functions[0]))])
        for i in range(len(nn_dimensions) - 1):
            layers[str(len(layers))] = nn.ZeroPad2d(1)
            layers[str(len(layers))] = nn.Conv2d(nn_dimensions[i], nn_dimensions[i + 1], 3)
            layers[str(len(layers))] = instantiate_activation_func(activation_functions[i + 1])

        layers[str(len(layers))] = nn.Conv2d(nn_dimensions[-1], 1, 3)
        #layers[str(len(layers))] = instantiate_activation_func(activation_functions[-1])
        layers[str(len(layers))] = nn.Conv2d(1, 1, 1)

        self.model = nn.Sequential(layers)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = instantiate_optimizer(optimizer, list(self.model.parameters()), self.learning_rate)
        # print(self.model)

    def generate_action(self, state: UniversalState, legal_actions: list) -> UniversalAction:
        input = state.generate_actor_input()

        prediction = self.inference_softmax([input]).data.numpy()[0]

        all_nodes = state.nodes.keys()
        nodes = [1 if node in legal_actions else 0 for node in all_nodes]

        for node_index in range(len(nodes)):
            nodes[node_index] *= prediction[node_index]
        nodes /= sum(nodes)

        greedy_index = np.argmax(nodes)

        size = sqrt(len(all_nodes))
        action = UniversalAction((greedy_index // size, greedy_index % size))

        return action

    def generate_probabilistic_action(self, state: UniversalState, legal_actions: list) -> UniversalAction:
        input = state.generate_actor_input()

        prediction = self.inference_softmax([input]).data.numpy()[0]

        all_nodes = state.ordered_keys()

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

    def fit(self, input: np.ndarray, target: np.ndarray) -> tuple:
        target = torch.FloatTensor(target)
        for i in range(self.epochs):
            pred_y = self.inference_linear(input, training=True)
            loss = self.loss_function(pred_y, target.argmax(dim=1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        acc = pred_y.argmax(dim=1).eq(target.argmax(dim=1)).sum().numpy() / len(target)
        return loss.item(), acc

    def inference_softmax(self, input: np.ndarray, training=False):
        self.train(training)
        x = self.process_data(input)
        x = self.model(x)
        x = x.reshape(-1, self.board_size**2)
        return F.softmax(x, dim=1)

    def inference_linear(self, input: np.ndarray, training=False):
        self.train(training)
        x = self.process_data(input)
        x = self.model(x)
        return x.reshape(-1, self.board_size**2)

    def process_data(self, input: np.ndarray):
        out = []
        for x in input:
            player = x[0]
            x = x[1:].reshape(self.board_size, self.board_size)
            planes = np.zeros(9 * self.board_size**2).reshape(9, self.board_size, self.board_size)

            planes[player + 3] += 1  # plane 3/4
            for r in range(self.board_size):
                for c in range(self.board_size):
                    piece = x[r][c]
                    planes[piece][r][c] = 1  # plane 0-2

                    if (r, c) in self.bridge_neighbors:
                        for (rb, cb) in self.bridge_neighbors[(r, c)]:
                            if piece == 0:
                                if x[rb][cb] == player:
                                    planes[7][r][c] = 1  # 7: form bridge
                            else:
                                if x[rb][cb] == piece:
                                    planes[piece + 4][r][c] = 1  # 5/6: bridge endpoints
                                    cn = list(set(self.neighbors[(r, c)]).intersection(set(self.neighbors[(rb, cb)])))  # common neighbors
                                    r1, c1 = cn[0]
                                    r2, c2 = cn[1]
                                    if x[r1][c1] == 0 and x[r2][c2] == 3 - player:
                                        planes[8][r1][c1] = 1
                                    elif x[r2][c2] == 0 and x[r1][c1] == 3 - player:
                                        planes[8][r2][c2] = 1
            out.append(planes)
        return torch.FloatTensor(out)

    def save_model(self, iterations: int) -> None:
        print(f'\nSaved model ANET_{iterations}')
        torch.save(self.state_dict(), f'../models/{Config.model_dir}/ANET_{iterations}')

    def load_model(self, iterations: int):
        self.iterations = str(iterations)
        print(f'Loaded model ANET_{iterations}')
        self.load_state_dict(torch.load(f'../models/{Config.model_dir}/ANET_{iterations}'))

    @staticmethod
    def __to_tensor(data):
        return torch.FloatTensor(data)
