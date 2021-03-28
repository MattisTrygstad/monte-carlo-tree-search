

from copy import deepcopy
from random import sample
import sys
from agent.actor import Actor
from agent.mcts import MonteCarloTree
from environment.hexagonal_grid import HexagonalGrid
from environment.universal_action import UniversalAction
from environment.universal_state import UniversalState


class ReinforcementLearning:

    def __init__(self, games: int, simulations: int, epochs: int, save_interval: int, learning_rate: float, input_size: int, nn_dimensions: list, activation_functions: list) -> None:
        self.games = games
        self.simulations = simulations
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.save_interval = save_interval

        self.actor = Actor(self.epochs, self.learning_rate, self.save_interval, input_size, nn_dimensions, activation_functions)
        self.replay_buffer = []

        self.losses = []
        self.accuracies = []

    def train(self):
        env = HexagonalGrid(visual=True)

        env.reset()

        init_state = UniversalState(deepcopy(env.state.nodes), env.get_player_turn())
        tree = MonteCarloTree(env.get_player_turn(), init_state)

        for game_index in range(self.games):

            counter = 0
            while True:
                # Check win condition
                if env.check_win_condition():
                    print(counter)
                    env.visualize(False, 10)
                    break

                for _ in range(self.simulations):
                    tree.single_pass()

                visit_counts = tree.get_children_visit_count(tree.root)

                self.append_replay_buffer(UniversalState(deepcopy(env.state.nodes), env.get_player_turn()), visit_counts)

                chosen_key = max(visit_counts, key=visit_counts.get)

                action = UniversalAction(chosen_key)

                env.execute_action(action)

                tree.set_root(action, UniversalState(deepcopy(env.state.nodes), env.get_player_turn()))

                counter += 1

            self.train_actor(game_index)

        self.actor.save_model(0)

    def append_replay_buffer(self, state: UniversalState, visit_counts: dict) -> None:
        self.replay_buffer.append((state.to_numpy(), visit_counts))

    def train_actor(self, game_index: int):
        batch_size = len(self.replay_buffer) // 2

        samples = sample(self.replay_buffer, batch_size)

        x_train, y_train = list(zip(*samples))

        loss, accuracy = self.actor.train(x_train, y_train)

        self.losses.append(loss)
        self.accuracies.append(accuracy)
