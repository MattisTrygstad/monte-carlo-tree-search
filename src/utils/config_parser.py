
import ast
import sys
import configparser


class Config:
    config = configparser.ConfigParser()
    config.read('config.ini')

    human_mode = bool(ast.literal_eval(config.get('parameters', 'human_mode')))

    experiments = bool(ast.literal_eval(config.get('parameters', 'experiments')))
    actor_learning_rates = list(ast.literal_eval(config.get('parameters', 'actor_learning_rates')))
    critic_learning_rates = list(ast.literal_eval(config.get('parameters', 'critic_learning_rates')))
    decay_discount_values = list(ast.literal_eval(config.get('parameters', 'decay_discount_values')))
    iterations = int(config.get('parameters', 'iterations'))
    win_multipliers = list(ast.literal_eval(config.get('parameters', 'win_multipliers')))
    initial_epsilons = list(ast.literal_eval(config.get('parameters', 'initial_epsilons')))
    exploitation_thresholds = list(ast.literal_eval(config.get('parameters', 'exploitation_thresholds')))

    reinforcement = int(config.get('parameters', 'reinforcement'))
    win_multiplier = int(config.get('parameters', 'win_multiplier'))

    if str(config.get('parameters', 'board_type')) == 'triangle':
        board_type = 0
    elif str(config.get('parameters', 'board_type')) == 'diamond':
        board_type = 1
    else:
        print('Invalid board type')
        sys.exit()

    board_size = int(config.get('parameters', 'board_size'))
    filled_nodes = list(ast.literal_eval(config.get('parameters', 'filled_nodes')))
    episodes = int(config.get('parameters', 'episodes'))
    test_episodes = int(config.get('parameters', 'test_episodes'))

    nn_critic = bool(ast.literal_eval(config.get('parameters', 'nn_critic')))
    nn_dimentions = list(ast.literal_eval(config.get('parameters', 'nn_dimentions')))
    nn_activation_functions = list(ast.literal_eval(config.get('parameters', 'nn_activation_functions')))

    actor_learning_rate = float(config.get('parameters', 'actor_learning_rate'))
    critic_learning_rate = float(config.get('parameters', 'critic_learning_rate'))
    actor_decay_rate = float(config.get('parameters', 'actor_decay_rate'))
    critic_decay_rate = float(config.get('parameters', 'critic_decay_rate'))
    actor_discount_factor = float(config.get('parameters', 'actor_discount_factor'))
    critic_discount_factor = float(config.get('parameters', 'critic_discount_factor'))

    linear_epsilon = bool(ast.literal_eval(config.get('parameters', 'linear_epsilon')))
    exploitation_threshold = int(config.get('parameters', 'exploitation_threshold'))
    epsilon = float(config.get('parameters', 'epsilon'))
    epsilon_decay = float(config.get('parameters', 'epsilon_decay'))

    visualize_without_convergence = str(ast.literal_eval(config.get('parameters', 'visualize_without_convergence')))
    visualization_delay = float(config.get('parameters', 'visualization_delay'))
