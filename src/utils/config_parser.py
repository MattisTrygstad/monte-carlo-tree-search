
import ast
import configparser


class Config:
    config = configparser.ConfigParser()
    config.read('config.ini')

    human_mode = bool(ast.literal_eval(config.get('PROGRAM_FLOW', 'human_mode')))
    experiments = bool(ast.literal_eval(config.get('PROGRAM_FLOW', 'experiments')))

    reinforcement = int(config.get('ENVIRONMENT', 'reinforcement'))
    board_size = int(config.get('ENVIRONMENT', 'board_size'))

    simulations = int(config.get('LEARNING', 'simulations'))
    episodes = int(config.get('LEARNING', 'episodes'))
    epochs = int(config.get('LEARNING', 'epochs'))
    nn_dimentions = list(ast.literal_eval(config.get('LEARNING', 'nn_dimentions')))
    nn_activation_functions = list(ast.literal_eval(config.get('LEARNING', 'nn_activation_functions')))
    optimizer = str(ast.literal_eval(config.get('LEARNING', 'optimizer')))
    actor_learning_rate = float(config.get('LEARNING', 'actor_learning_rate'))
    critic_learning_rate = float(config.get('LEARNING', 'critic_learning_rate'))
    save_interval = float(config.get('LEARNING', 'critic_learning_rate'))
    exploration_constant = float(config.get('LEARNING', 'exploration_constant'))
    rollout_bias = float(config.get('LEARNING', 'rollout_bias'))

    epsilon = float(config.get('EPSILON', 'epsilon'))
    epsilon_decay = float(config.get('EPSILON', 'epsilon_decay'))

    visualization_delay = float(config.get('VISUALIZATION', 'visualization_delay'))
