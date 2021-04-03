

from random import choice
import sys
from agent.actor import Actor
from enums import Player
from environment.hexagonal_grid import HexagonalGrid
from utils.config_parser import Config


class Tournament:

    def __init__(self, games: int, training_games: int, save_interval: int) -> None:
        self.games = games

        self.agents = []
        self.results = {}  # Key: str(iteraton_num), value: wins

        self.env = HexagonalGrid(visual=True)
        self.load_agents(training_games, save_interval)

    def load_agents(self, training_games: int, save_interval: int) -> None:

        number_of_agents = int(training_games // save_interval) + 1

        print(number_of_agents)

        for agent_index in range(number_of_agents):
            agent_iterations = int(agent_index * save_interval)
            self.results[str(agent_iterations)] = 0

            actor = Actor(Config.epochs, Config.actor_learning_rate, Config.save_interval, Config.board_size, Config.nn_dimentions, Config.nn_activation_functions, Config.optimizer)
            actor.load_model(agent_iterations)

            self.agents.append(actor)

    def run(self):

        # total series: n * (n-1) / 2
        while len(self.agents) > 1:
            p1: Actor = self.agents.pop()

            for p2_index in range(len(self.agents)):
                p2: Actor = self.agents[p2_index]
                print(f'Player {p1.iterations} vs. Player {p2.iterations}')
                scores = [0, 0]
                for game_index in range(self.games):
                    winner: Player = self.game(game_index, p1, p2)
                    scores[winner.value] += 1

                print(f'Final scores: {scores[0]} - {scores[1]}\n')
                if scores[0] > scores[1]:
                    self.results[p1.iterations] += 1
                elif scores[0] < scores[1]:
                    self.results[p2.iterations] += 1

        print(self.results)

    def game(self, game_index: int, p1: Actor, p2: Actor):
        self.env.reset()

        start_player = choice(list(Player))
        end_player = Player.ONE if start_player == Player.TWO else Player.TWO

        while True:
            # Check win condition
            if self.env.check_win_condition():
                return self.env.winner

            player = self.env.get_player_turn()
            state = self.env.get_state()
            actions = self.env.get_legal_actions()

            if player == start_player:
                action = p1.generate_action(state, actions)

            elif player == end_player:
                action = p2.generate_action(state, actions)

            self.env.execute_action(action)

            if game_index == 0 and Config.visualize:
                self.env.visualize(False, 1)
