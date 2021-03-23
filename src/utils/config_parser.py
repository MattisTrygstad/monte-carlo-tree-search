
import ast
import sys
import configparser


class Config:
    config = configparser.ConfigParser()
    config.read('config.ini')

    human_mode = bool(ast.literal_eval(config.get('PROGRAM_FLOW', 'human_mode')))
    experiments = bool(ast.literal_eval(config.get('PROGRAM_FLOW', 'experiments')))

    reinforcement = int(config.get('ENVIRONMENT', 'reinforcement'))
    win_multiplier = int(config.get('ENVIRONMENT', 'win_multiplier'))
    board_size = int(config.get('ENVIRONMENT', 'board_size'))
    filled_nodes = list(ast.literal_eval(config.get('ENVIRONMENT', 'filled_nodes')))

    rollout_bias = float(config.get('LEARNING', 'rollout_bias'))
    exploitation_factor = float(config.get('LEARNING', 'exploitation_factor'))
    simulations = int(config.get('LEARNING', 'simulations'))
    episodes = int(config.get('LEARNING', 'episodes'))
    test_episodes = int(config.get('LEARNING', 'test_episodes'))
    nn_dimentions = list(ast.literal_eval(config.get('LEARNING', 'nn_dimentions')))
    nn_activation_functions = list(ast.literal_eval(config.get('LEARNING', 'nn_activation_functions')))
    actor_learning_rate = float(config.get('LEARNING', 'actor_learning_rate'))
    critic_learning_rate = float(config.get('LEARNING', 'critic_learning_rate'))
    save_interval = float(config.get('LEARNING', 'critic_learning_rate'))

    linear_epsilon = bool(ast.literal_eval(config.get('EPSILON', 'linear_epsilon')))
    exploitation_threshold = int(config.get('EPSILON', 'exploitation_threshold'))
    epsilon = float(config.get('EPSILON', 'epsilon'))
    epsilon_decay = float(config.get('EPSILON', 'epsilon_decay'))

    visualize_without_convergence = str(ast.literal_eval(config.get('VISUALIZATION', 'visualize_without_convergence')))
    visualization_delay = float(config.get('VISUALIZATION', 'visualization_delay'))
