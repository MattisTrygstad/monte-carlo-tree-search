

import matplotlib.pyplot as plt
from agent.actor_conv import Actor
from enums import Player
from environment.state_manager import StateManager
from utils.config_parser import Config

import numpy as np


class Tournament:

    def __init__(self, games: int, training_games: int, save_interval: int) -> None:
        self.games = games

        self.agents = []
        self.series_results = {}  # Key: str(iteraton_num), value: wins
        self.game_results = {}  # Key: str(iteraton_num), value: wins

        self.env = StateManager(visual=True)
        self.load_agents(training_games, save_interval)

        plt.close('all')

    def load_agents(self, training_games: int, save_interval: int) -> None:

        number_of_agents = int(training_games // save_interval) + 1

        for agent_index in range(number_of_agents):
            agent_iterations = int(agent_index * save_interval)
            self.series_results[str(agent_iterations)] = 0
            self.game_results[str(agent_iterations)] = 0

            actor = Actor(Config.epochs, Config.actor_learning_rate, Config.save_interval, Config.board_size, Config.nn_dimentions, Config.nn_activation_functions, Config.optimizer)
            actor.load_model(agent_iterations)

            self.agents.append(actor)

    def run(self):

        # total series: n * (n-1) / 2
        while len(self.agents) > 1:
            p1: Actor = self.agents.pop()

            for p2_index in range(len(self.agents)):
                p2: Actor = self.agents[p2_index]
                print(f'Blue {p1.iterations} vs. Red {p2.iterations}')
                scores = [0, 0]
                for game_index in range(self.games):
                    winner: Player = self.game(game_index, p1, p2)
                    scores[winner.value] += 1

                print(f'Final scores: {scores[0]} - {scores[1]}\n')
                if scores[0] > scores[1]:
                    self.series_results[p1.iterations] += 1
                elif scores[0] < scores[1]:
                    self.series_results[p2.iterations] += 1

                self.game_results[p1.iterations] += scores[0]
                self.game_results[p2.iterations] += scores[1]

        print(f'-- Total games won during tournament --')
        for k, v in sorted(self.game_results.items(), key=lambda item: item[1], reverse=True):
            print(f'ANET_{k}: {v:5.0f} wins')

        plot(self.series_results)

    def game(self, game_index: int, p1: Actor, p2: Actor):

        self.env.reset(random=True)

        if game_index == 0 and Config.visualize:
            print(f'Start: {self.env.get_player_turn()}')

        while True:
            if self.env.check_win_condition():
                return self.env.winner

            player = self.env.get_player_turn()
            state = self.env.get_state()
            actions = self.env.get_legal_actions()

            if player == Player.ONE:
                action = p1.generate_action(state, actions)

            elif player == Player.TWO:
                action = p2.generate_action(state, actions)

            self.env.execute_action(action)

            if game_index == 0 and Config.visualize:
                self.env.visualize(False, 1)


def plot(results: dict):
    plt.close()
    plt.figure(figsize=(12, 6))

    x_axis_labels = []
    y_axis_values = []

    for key, value in results.items():
        x_axis_labels.append(key)
        y_axis_values.append(value)

    y_pos = np.arange(len(x_axis_labels))

    plt.bar(y_pos + 0, y_axis_values, width=0.5, color='c', label='legend title')
    plt.xticks(y_pos, x_axis_labels)
    plt.legend(loc='best')
    plt.ylabel('Wins')
    plt.xlabel('Episodes trained')

    plt.title("Tournament results")

    plt.show()

    plt.close()
