
import ast
import configparser


class Config:
    config = configparser.ConfigParser()
    config.read('config.ini')

    train = bool(ast.literal_eval(config.get('PROGRAM_FLOW', 'train')))
    tournament = bool(ast.literal_eval(config.get('PROGRAM_FLOW', 'tournament')))
    online_tournament = bool(ast.literal_eval(config.get('PROGRAM_FLOW', 'online_tournament')))

    tournament_games = int(config.get('TOURNAMENT', 'tournament_games'))
    visualize = bool(ast.literal_eval(config.get('TOURNAMENT', 'visualize')))

    board_size = int(config.get('ENVIRONMENT', 'board_size'))

    simulations = int(config.get('LEARNING', 'simulations'))
    episodes = int(config.get('LEARNING', 'episodes'))
    epochs = int(config.get('LEARNING', 'epochs'))
    nn_dimentions = list(ast.literal_eval(config.get('LEARNING', 'nn_dimentions')))
    nn_activation_functions = list(ast.literal_eval(config.get('LEARNING', 'nn_activation_functions')))
    optimizer = str(ast.literal_eval(config.get('LEARNING', 'optimizer')))
    actor_learning_rate = float(config.get('LEARNING', 'actor_learning_rate'))
    save_interval = float(config.get('LEARNING', 'save_interval'))
    exploration_constant = float(config.get('LEARNING', 'exploration_constant'))
    batch_size = int(config.get('LEARNING', 'batch_size'))
    buffer_limit = int(config.get('LEARNING', 'buffer_limit'))

    epsilon = float(config.get('EPSILON', 'epsilon'))

    visualization_delay = float(config.get('VISUALIZATION', 'visualization_delay'))
