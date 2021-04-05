

from copy import deepcopy
from random import sample
import sys

import numpy as np
import torch
from agent.actor import Actor
from agent.mcts import MonteCarloTree
from environment.hexagonal_grid import HexagonalGrid
from environment.universal_action import UniversalAction
from environment.universal_state import UniversalState
from utils.config_parser import Config
from utils.loadingbar import print_progress
from utils.visualize_training import visualize_training


class ReinforcementLearning:

    def __init__(self, games: int, simulations: int, epochs: int, save_interval: int, epsilon: float, learning_rate: float, board_size: int, nn_dimensions: list, activation_functions: list, optimizer: str, exploration_constant: float) -> None:
        self.games = games
        self.simulations = simulations
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon / self.games
        self.save_interval = save_interval
        self.exploration_constant = exploration_constant
        self.board_size = board_size

        self.actor = Actor(self.epochs, self.learning_rate, self.save_interval, board_size, nn_dimensions, activation_functions, optimizer)
        self.replay_buffer = []

        self.losses = []
        self.accuracies = []

        self.heuristic_env = HexagonalGrid(visual=False)

    def train(self):
        env = HexagonalGrid(visual=True)

        init_state = env.get_state()
        tree = MonteCarloTree(deepcopy(init_state), self.actor, self.epsilon, self.exploration_constant)

        for game_index in range(self.games):
            if game_index % self.save_interval == 0:
                self.actor.save_model(game_index)

            env.reset()
            tree.reset(deepcopy(init_state))

            while True:
                if env.check_win_condition():
                    #env.visualize(False, 10)
                    break

                for sim_index in range(self.simulations):
                    tree.single_pass()

                distribution = tree.get_action_distribution(tree.root)
                self.append_replay_buffer(env.get_state(), distribution)

                chosen_key = max(distribution, key=distribution.get)
                action = UniversalAction(chosen_key)
                env.execute_action(action)

                tree.set_root(action, env.get_state())

                # Visualize last game
                # if game_index == self.games - 1:
                #     env.visualize(False, 1)

            self.train_actor(game_index)
            tree.epsilon -= self.epsilon_decay

            loss = 0 if len(self.losses) == 0 else self.losses[-1]
            acc = 0 if len(self.accuracies) == 0 else self.accuracies[-1]
            print_progress(game_index, self.games, length=20, suffix=f'Game: {game_index+1}/{self.games}, Epsilon: {round(tree.epsilon,2):0.2f}, Loss: {round(loss, 5):0.5f}, Acc: {round(acc,5):0.5f}')

        print()
        self.actor.save_model(self.games)
        visualize_training(self.losses, self.accuracies)

    def append_replay_buffer(self, state: UniversalState, distribution: dict) -> None:

        for (row, col) in state.nodes.keys():
            if (row, col) in distribution.keys():
                continue
            distribution[(row, col)] = 0

        assert len(distribution) == len(state.nodes.keys())

        distribution = self.apply_heuristics(distribution, state)

        input = state.generate_actor_input()

        target = np.zeros((self.board_size**2,))
        for (row, col), value in distribution.items():
            index = row * self.board_size + col
            target[index] = value

        assert distribution[(1, 3)] == target[7]
        assert distribution[(3, 3)] == target[15]

        if len(self.replay_buffer) > Config.buffer_limit:
            self.replay_buffer.pop(0)
        self.replay_buffer.append((input, target))

    def train_actor(self, game_index: int):
        batch_size = min(Config.batch_size, len(self.replay_buffer))

        samples = sample(self.replay_buffer, batch_size)

        x_train, y_train = list(zip(*samples))

        loss, accuracy = self.actor.fit(x_train, y_train)

        self.losses.append(loss)
        self.accuracies.append(accuracy)

    def apply_heuristics(self, distribution: dict, state: UniversalState) -> torch.Tensor:
        player = state.player

        values = list(state.nodes.values())

        # If first move in game, select center
        if sum(values) == 0:
            #print('first move')
            distribution = dict.fromkeys(distribution.keys(), 0.0)
            distribution[(self.board_size // 2, self.board_size // 2)] = 1.0
            return distribution

        # Check for obvious wins in next move
        for key, value in distribution.items():
            # If probability is higher than 50%, check if it could be a winning state
            if value > 0.5:
                self.heuristic_env.state.nodes = deepcopy(state.nodes)

                self.heuristic_env.state.nodes[key] = player.value + 1
                self.heuristic_env.state.player = player
                if self.heuristic_env.check_win_condition():
                    #print('found winning action')
                    distribution = dict.fromkeys(distribution.keys(), 0.0)
                    distribution[key] = 1.0
                    return distribution

        return distribution
