

from copy import deepcopy
import sys
from agent.actor import Actor
from agent.mcts import MonteCarloTree
from environment.hexagonal_grid import HexagonalGrid
from environment.universal_action import UniversalAction
from environment.universal_state import UniversalState
from utils.config_parser import Config


class ReinforcementLearning:

    def __init__(self, games: int, simulations: int, save_interval: int) -> None:
        self.games = games
        self.simulations = simulations
        self.save_interval = save_interval

    def train(self):
        env = HexagonalGrid(visual=True)

        #actor = Actor(Config.actor_learning_rate, self.save_interval, 5, Config.nn_dimentions, Config.nn_activation_functions)

        env.reset()

        init_state = UniversalState(deepcopy(env.state.nodes))
        tree = MonteCarloTree(env.get_player_turn(), init_state)

        replay_buffer = []

        for game_index in range(self.games):

            counter = 0
            while True:
                # Check win condition
                if env.check_win_condition():
                    print(replay_buffer)
                    print(len(replay_buffer))
                    print(counter)
                    env.visualize(False, 10)
                    break

                for sim_index in range(self.simulations):
                    tree.single_pass()

                visit_counts = tree.get_children_visit_count(tree.root)

                replay_buffer.append((tree.root, visit_counts))

                chosen_key = max(visit_counts, key=visit_counts.get)

                action = UniversalAction(chosen_key)

                env.execute_action(action)

                tree.set_root(action, UniversalState(deepcopy(env.state.nodes)))

                counter += 1
