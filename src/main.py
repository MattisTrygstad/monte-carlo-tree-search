

from agent.reinforcement_learning import ReinforcementLearning
from utils.config_parser import Config
from utils.normal_game import normal_game


def main():
    rl = ReinforcementLearning(Config.episodes, Config.simulations, Config.epochs, Config.save_interval, Config.epsilon, Config.epsilon_decay, Config.actor_learning_rate, Config.board_size, Config.nn_dimentions, Config.nn_activation_functions, Config.optimizer, Config.exploration_constant)

    rl.train()

    # if Config.human_mode:
    #     normal_game()


if __name__ == "__main__":
    main()
