import ast
import sys

from matplotlib import pyplot as plt
from environment.hexagonal_grid import HexagonalGrid
from environment.universal_action import UniversalAction
from utils.config_parser import Config


def normal_game():
    env = HexagonalGrid(Config.win_multiplier)

    env.visualize(False)

    while True:
        # Check win condition
        if env.check_win_condition():
            env.visualize(False, 10)
            break

        legal_actions = env.get_legal_actions()

        print('-----\nLegal moves:')
        for action in legal_actions:
            print(f'{action}')
        print('-----')

        player = env.get_player_turn()

        user_input = input(f'Player {player.value+1}s turn: ')
        if user_input == 'q':
            break

        if user_input == 'undo':
            print('Action reversed')
            env.undo_action()
            env.visualize(False)
            continue

        try:
            node = tuple(ast.literal_eval(user_input))
        except:
            print('Invalid input, try again!')
            continue

        if node not in legal_actions:
            print('Illegal placement, try again!')
            continue

        action = UniversalAction()
        action.action = node
        env.execute_action(action)
        env.visualize(False)

    plt.close()
