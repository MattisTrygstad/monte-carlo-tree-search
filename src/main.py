
from copy import deepcopy
from datetime import timedelta
from statistics import mean, median
import sys
import time

from matplotlib import pyplot as plt
from agent.actor import Actor
from agent.critic import Critic
from agent.mcts import MonteCarloTree, Node
from agent.table_approximator import TableApproximator
from environment.hexagonal_grid import HexagonalGrid
from environment.universal_action import UniversalAction
from environment.universal_state import UniversalState
from utils.config_parser import Config
from utils.normal_game import normal_game


#from agent.neural_network_approximator import NeuralNetworkApproximator


def main():
    rl_algorithm()

    # if Config.human_mode:
    #     normal_game()

    # elif Config.experiments:
    #     pass

    # else:
    #     pass


def rl_algorithm():
    env = HexagonalGrid()

    actor = Actor(Config.actor_learning_rate, Config.save_interval, 5, Config.nn_dimentions, Config.nn_activation_functions)

    for episode_index in range(Config.episodes):
        env.reset()

        init_state = UniversalState(deepcopy(env.state.nodes))

        tree = MonteCarloTree(env.get_player_turn(), init_state)

    while True:
        # Check win condition
        if env.check_win_condition():
            env.visualize(False, 10)
            break

        for sim_index in range(Config.simulations):
            tree.single_pass()

        sys.exit()
        # D = tree.get_visit_count(tree.root)


if __name__ == "__main__":
    main()
