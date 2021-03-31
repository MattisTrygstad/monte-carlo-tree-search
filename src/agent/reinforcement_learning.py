

from copy import deepcopy
from random import sample

import numpy as np
from agent.actor import Actor
from agent.mcts import MonteCarloTree
from environment.hexagonal_grid import HexagonalGrid
from environment.universal_action import UniversalAction
from environment.universal_state import UniversalState
from utils.loadingbar import print_progress
from utils.visualize_training import visualize_training


class ReinforcementLearning:

    def __init__(self, games: int, simulations: int, epochs: int, save_interval: int, epsilon: float, epsilon_decay: float, learning_rate: float, board_size: int, nn_dimensions: list, activation_functions: list, optimizer: str, exploration_constant: float) -> None:
        self.games = games
        self.simulations = simulations
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.save_interval = save_interval
        self.exploration_constant = exploration_constant

        self.actor = Actor(self.epochs, self.learning_rate, self.save_interval, board_size, nn_dimensions, activation_functions, optimizer)
        self.replay_buffer = []

        self.losses = []
        self.accuracies = []

    def train(self):
        env = HexagonalGrid(visual=True)

        init_state = UniversalState(deepcopy(env.state.nodes), env.get_player_turn())
        tree = MonteCarloTree(deepcopy(init_state), self.actor, self.epsilon, self.exploration_constant)

        for game_index in range(self.games):
            env.reset()
            tree.reset(deepcopy(init_state))

            while True:
                # Check win condition
                if env.check_win_condition():
                    #env.visualize(False, 10)
                    break

                for sim_index in range(self.simulations):
                    tree.single_pass()

                distribution = tree.get_action_distribution(tree.root)
                self.append_replay_buffer(UniversalState(deepcopy(env.state.nodes), env.get_player_turn()), distribution)

                chosen_key = max(distribution, key=distribution.get)
                action = UniversalAction(chosen_key)
                env.execute_action(action)

                tree.set_root(action, UniversalState(deepcopy(env.state.nodes), env.get_player_turn()))

                # Visualize last game
                # if game_index == self.games - 1:
                #     env.visualize(False, 1)

            self.train_actor(game_index)
            tree.epsilon *= self.epsilon_decay

            loss = 0 if len(self.losses) == 0 else self.losses[-1]
            acc = 0 if len(self.accuracies) == 0 else self.accuracies[-1]
            print_progress(game_index, self.games, length=20, suffix=f'Epsilon: {round(tree.epsilon,5)}, Loss: {round(loss, 5)}, Acc: {round(acc,5)}')

        print()
        self.actor.save_model(0)

        visualize_training(self.losses, self.accuracies)

    def append_replay_buffer(self, state: UniversalState, distribution: dict) -> None:

        for (row, col) in state.nodes.keys():
            if (row, col) in distribution.keys():
                continue
            distribution[(row, col)] = 0

        input = state.to_numpy()

        visit_count_list = np.asarray(list(distribution.values()))
        target = visit_count_list / np.sum(visit_count_list)
        self.replay_buffer.append((input, target))

    def train_actor(self, game_index: int):
        batch_size = len(self.replay_buffer) // 2

        samples = sample(self.replay_buffer, batch_size)

        x_train, y_train = list(zip(*samples))

        loss, accuracy = self.actor.train(x_train, y_train)

        self.losses.append(loss)
        self.accuracies.append(accuracy)
