
from datetime import timedelta
from statistics import mean, median
import sys
import time

from matplotlib import pyplot as plt
from agent.actor import Actor
from agent.critic import Critic
from agent.mcts import Node
from agent.table_approximator import TableApproximator
from environment.hexagonal_grid import HexagonalGrid
from environment.universal_action import UniversalAction
from environment.universal_state import UniversalState
from utils.config_parser import Config
from utils.normal_game import normal_game


#from agent.neural_network_approximator import NeuralNetworkApproximator


def main():
    env = HexagonalGrid(0)
    actions = env.get_legal_actions()
    player = env.get_player_turn()
    node = Node(actions, player)

    sys.exit()
    if Config.human_mode:
        normal_game()

    elif Config.experiments:
        run_experiments()

    else:
        actor_critic_game(Config.actor_learning_rate, Config.critic_learning_rate, Config.actor_decay_rate, Config.critic_decay_rate, Config.actor_discount_factor, Config.critic_discount_factor, Config.linear_epsilon, Config.win_multiplier, Config.epsilon, Config.exploitation_threshold, True)


def actor_critic_game(actor_learning_rate: float, critic_learning_rate: float, actor_decay_rate: float, critic_decay_rate: float, actor_discount_factor: float, critic_discount_factor: float, linear_epsilon: bool, win_multiplier: int, initial_epsilon: float, exploitation_start: int, visualize: bool) -> int:
    env = HexagonalGrid(win_multiplier)

    if Config.nn_critic:
        approximator = NeuralNetworkApproximator(len(str(env.get_state())), Config.nn_dimentions, Config.nn_activation_functions, critic_learning_rate, critic_discount_factor, critic_decay_rate)
    else:
        approximator = TableApproximator(critic_discount_factor, critic_decay_rate, critic_learning_rate)

    critic = Critic(approximator)
    actor = Actor(actor_discount_factor, actor_decay_rate, actor_learning_rate)

    # Exploration vs. exploitation configuration
    epsilon = initial_epsilon
    epsilon_linear = epsilon / Config.episodes

    # Statistics
    training_wins = 0
    test_wins = 0
    losses = 0
    remaining_nodes = []

    for episode in range(Config.episodes + Config.test_episodes):
        env.reset()
        critic.reset_eligibilities()
        actor.reset_eligibilities()

        if episode >= Config.episodes:
            # No exploration during final model test
            epsilon = 0
        elif linear_epsilon:
            if training_wins >= exploitation_start:
                epsilon = initial_epsilon - epsilon_linear * (episode + 1)
        else:
            if training_wins >= exploitation_start:
                epsilon *= Config.epsilon_decay

        state: UniversalState = env.get_state()
        action: UniversalAction = actor.generate_action(state, env.get_legal_actions(), epsilon)

        while True:
            reinforcement = env.execute_action(action)

            if env.check_win_condition():
                if episode < Config.episodes:
                    training_wins += 1
                else:
                    test_wins += 1
                break

            if len(env.get_legal_actions()) == 0:
                losses += 1
                break

            next_state: UniversalState = env.get_state()
            next_legal_actions = env.get_legal_actions()
            next_action: UniversalAction = actor.generate_action(next_state, next_legal_actions, epsilon)

            actor.set_eligibility(state, action, 1)
            td_error = critic.compute_temporal_difference_error(state, next_state, reinforcement)

            critic.set_eligibility(state, 1)

            # For all (s,a) pairs
            critic.compute_state_values(td_error, reinforcement, state, next_state)
            critic.decay_eligibilities()
            actor.compute_policies(td_error)
            actor.decay_eligibilities()

            state = next_state
            action = next_action

        if visualize:
            remaining_nodes.append(len(env.state.get_player_one_nodes()))

            if episode < Config.episodes:
                print(f'Episode: {episode}, wins: {training_wins}, losses: {losses}, epsilon: {round(epsilon, 5)}')
            if episode == Config.episodes:
                print(f'Testing final model...')

    plt.close()

    if visualize:
        print(f'Final model win rate: {test_wins}/{Config.test_episodes} = {round(test_wins/Config.test_episodes*100, 2)}% ')

        if Config.visualize_without_convergence or test_wins == Config.test_episodes:
            plt.plot(remaining_nodes)
            plt.ylabel('Remaining nodes')
            plt.show()

            plt.close()

        if test_wins > Config.test_episodes * 0.6:

            visualize_greedy_episode(actor, critic)

    return training_wins, test_wins


def visualize_greedy_episode(actor: Actor, critic: Critic):
    env = HexagonalGrid(Config.win_multiplier)

    epsilon = 0

    state: UniversalState = env.get_state()
    action = actor.generate_action(state, env.get_legal_actions(), epsilon)

    while True:
        reinforcement = env.execute_action(action)
        env.visualize(False, Config.visualization_delay)

        if env.check_win_condition():
            break

        if len(env.get_legal_actions()) == 0:
            break

        next_state = env.get_state()
        next_legal_actions = env.get_legal_actions()
        next_action = actor.generate_action(next_state, next_legal_actions, epsilon)

        actor.set_eligibility(state, action, 1)
        td_error = critic.compute_temporal_difference_error(state, next_state, reinforcement)

        critic.set_eligibility(state, 1)

        # For all (s,a) pairs
        critic.compute_state_values(td_error, reinforcement, state, next_state)
        critic.decay_eligibilities()
        actor.compute_policies(td_error)
        actor.decay_eligibilities()

        state = next_state
        action = next_action

    env.visualize(True)


def run_experiments() -> None:
    actor_learnig_rates = Config.actor_learning_rates
    critic_learnig_rates = Config.critic_learning_rates
    decay_discount_values = Config.decay_discount_values
    win_multipliers = Config.win_multipliers
    initial_epsilons = Config.initial_epsilons
    exploitation_thresholds = Config.exploitation_thresholds

    iterations = Config.iterations

    total = len(actor_learnig_rates) * len(critic_learnig_rates) * len(win_multipliers) * len(initial_epsilons) * len(exploitation_thresholds) * (len(decay_discount_values)**4) * iterations

    count = 1
    estimated_run_time = 0

    for actor_learning_rate in actor_learnig_rates:
        for critic_learning_rate in critic_learnig_rates:
            for critic_decay_rate in decay_discount_values:
                for actor_decay_rate in decay_discount_values:
                    for critic_discount_factor in decay_discount_values:
                        for actor_discount_factor in decay_discount_values:
                            for multiplier in win_multipliers:
                                for epsilon in initial_epsilons:
                                    for threshold in exploitation_thresholds:

                                        training_wins = []
                                        test_wins = []

                                        for x in range(iterations):
                                            start = time.time()
                                            print(f'Experiment progress: {count}/{total}, estimated run time: {str(timedelta(seconds=estimated_run_time))}')
                                            training, test = actor_critic_game(actor_learning_rate, critic_learning_rate, actor_decay_rate, critic_decay_rate, actor_discount_factor, critic_discount_factor, Config.linear_epsilon, multiplier, epsilon, threshold, False)

                                            training_wins.append(training)
                                            test_wins.append(test)

                                            count += 1

                                            end = time.time()
                                            estimated_run_time = (end - start) * (total - count)

                                        f = open('../experiment_results.txt', 'a')
                                        f.write(f'{critic_learning_rate},{actor_learning_rate},{critic_decay_rate},{actor_decay_rate},{critic_discount_factor},{actor_discount_factor},{multiplier},{Config.linear_epsilon},{epsilon},{threshold},{round(mean(training_wins), 2)},{round(median(training_wins),2)},{round(mean(test_wins), 2)},{round(median(test_wins),2)}\n')
                                        f.close()


if __name__ == "__main__":
    main()
