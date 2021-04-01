

import sys
from agent.reinforcement_learning import ReinforcementLearning
from enums import Player
from environment.hexagonal_grid import HexagonalGrid
from environment.universal_state import UniversalState
from utils.config_parser import Config
from utils.normal_game import normal_game


def main():
    win_test = False
    if win_test:
        nodes = {(0, 0): 2, (0, 1): 1, (0, 2): 1, (0, 3): 1, (1, 0): 2, (1, 1): 1, (1, 2): 2, (1, 3): 2, (2, 0): 2, (2, 1): 2, (2, 2): 1, (2, 3): 1, (3, 0): 2, (3, 1): 1, (3, 2): 2, (3, 3): 1}

        env = HexagonalGrid(UniversalState(nodes, Player.TWO), True)

        print(env.check_win_condition())

        env.visualize(True)
        sys.exit()

    rl = ReinforcementLearning(Config.episodes, Config.simulations, Config.epochs, Config.save_interval, Config.epsilon, Config.epsilon_decay, Config.actor_learning_rate, Config.board_size, Config.nn_dimentions, Config.nn_activation_functions, Config.optimizer, Config.exploration_constant)

    rl.train()

    # if Config.human_mode:
    #     normal_game()


if __name__ == "__main__":
    main()
