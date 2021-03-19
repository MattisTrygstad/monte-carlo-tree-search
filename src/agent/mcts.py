

import numpy as np
from enums import Player
from environment.universal_action import UniversalAction


class Mcts:

    def __init__(self) -> None:
        pass

    def single_pass(self) -> None:
        pass

    def tree_search(self) -> None:
        pass

    def expand_node(self) -> None:
        pass

    def evaluate_leaf(self) -> None:
        # Using rollout or critic
        pass

    def backprop(self) -> None:
        pass


class Node:

    def __init__(self, actions: list, player: Player) -> None:
        # Key: action, value: (N, Q)
        self.actions = {action: (0, 0) for action in actions}
        self.N = 0
        self.previous_action = None
        self.player = player

    def update(self, value):
        self.N += 1
        self.actions[self.previous_action][0] += 1

        prev_N, prev_Q = self.actions[self.previous_action]

        self.actions[self.previous_action][1] += (value - prev_Q) / prev_N

    def set_previous_action(self, action: UniversalAction):
        self.previous_action = action

    def compute_uct(self, action: UniversalAction, exploitation_factor: float):
        N, Q = self.actions[action]
        exploitation_factor *= -1 if self.player == Player.TWO else 1

        utc = exploitation_factor * np.sqrt(np.log(self.N if self.N > 0 else 1) / (N + 1))
        return utc + Q
