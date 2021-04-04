

import sys
from agent.reinforcement_learning import ReinforcementLearning
from enums import Player
from environment.hexagonal_grid import HexagonalGrid
from environment.universal_state import UniversalState
from tournament import Tournament
from utils.config_parser import Config
from utils.normal_game import normal_game


def main():

    if Config.train:
        rl = ReinforcementLearning(Config.episodes, Config.simulations, Config.epochs, Config.save_interval, Config.epsilon, Config.actor_learning_rate, Config.board_size, Config.nn_dimentions, Config.nn_activation_functions, Config.optimizer, Config.exploration_constant)
        rl.train()

    if Config.tournament:
        tournament = Tournament(Config.tournament_games, Config.episodes, Config.save_interval)
        tournament.run()

    # if Config.human_mode:
    #     normal_game()


if __name__ == "__main__":
    main()
