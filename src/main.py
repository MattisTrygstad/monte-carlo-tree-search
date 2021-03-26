
from copy import deepcopy
from datetime import timedelta
from statistics import mean, median
import sys
import time

from matplotlib import pyplot as plt
from agent.actor import Actor
from agent.critic import Critic
from agent.mcts import MonteCarloTree, Node
from agent.reinforcement_learning import ReinforcementLearning
from agent.table_approximator import TableApproximator
from enums import Player
from environment.hexagonal_grid import HexagonalGrid
from environment.universal_action import UniversalAction
from environment.universal_state import UniversalState
from utils.config_parser import Config
from utils.normal_game import normal_game


def main():
    rl = ReinforcementLearning(Config.episodes, Config.simulations, Config.save_interval)

    rl.train()

    # if Config.human_mode:
    #     normal_game()

    # elif Config.experiments:
    #     pass

    # else:
    #     pass


if __name__ == "__main__":
    main()
