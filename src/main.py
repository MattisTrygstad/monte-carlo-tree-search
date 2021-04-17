from agent.reinforcement_learning import ReinforcementLearning
from BasicClientActor import BasicClientActor
from tournament import Tournament
from utils.config_parser import Config


def main():
    if Config.online_tournament:
        client = BasicClientActor(verbose=False)
        client.connect_to_server()
    else:

        if Config.train:
            rl = ReinforcementLearning(Config.episodes, Config.simulations, Config.epochs, Config.save_interval, Config.epsilon, Config.actor_learning_rate, Config.board_size, Config.nn_dimentions, Config.nn_activation_functions, Config.optimizer, Config.exploration_constant)
            rl.train()

        if Config.tournament:
            tournament = Tournament(Config.tournament_games, Config.episodes, Config.save_interval)
            tournament.run()


if __name__ == "__main__":
    main()
