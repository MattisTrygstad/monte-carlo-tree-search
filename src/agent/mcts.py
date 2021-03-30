

from copy import deepcopy
from random import choice, choices
import random
from typing import Tuple
import numpy as np
from agent.actor import Actor
from agent.node import Node
from enums import Player
from environment.hexagonal_grid import HexagonalGrid
from environment.universal_action import UniversalAction
from environment.universal_state import UniversalState


class MonteCarloTree:

    def __init__(self, state: UniversalState, actor: Actor, epsilon: float) -> None:
        self.root = Node(None, None, Player.ONE)
        self.init_state = state
        self.env = HexagonalGrid(self.init_state, False)
        self.epsilon = epsilon
        self.actor = actor

    def reset(self, state: UniversalState):
        self.env.reset()
        self.init_state = state
        self.root = Node(None, None, Player.ONE)

    def set_root(self, action: UniversalAction, state: UniversalState):
        self.init_state = state
        self.env.reset(state)

        # TODO: discard unused children in tree
        self.root = self.root.children[action.coordinates]

    @staticmethod
    def get_children_visit_count(node: Node) -> dict:
        # Key: node_coordinate, value: visit_count
        return {key: child.visit_count for key, child in node.children.items()}

    def single_pass(self) -> None:
        node = self.tree_search()
        player = self.env.get_player_turn()
        winner = self.evaluate_leaf()
        self.backprop(node, player, winner)

    def tree_search(self) -> Node:
        """
        Traversing the tree from the root to a leaf node by using the tree policy.
        """
        node = self.root
        self.env.reset(UniversalState(deepcopy(self.init_state.nodes)))

        while len(node.children) != 0:
            visit_counts = self.get_children_visit_count(node)

            max_value = max(visit_counts.values())
            max_keys = [k for k, v in visit_counts.items() if v == max_value]

            chosen_key = choice(max_keys)
            node = node.children[chosen_key]
            action = UniversalAction(chosen_key)
            self.env.execute_action(action)

            if visit_counts[chosen_key] == 0:
                return node

        if self.expand_node(node):
            node: Node = choice(list(node.children.values()))
            self.env.execute_action(node.previous_action)

        return node

    def expand_node(self, node: Node) -> bool:
        """
        Generating some or all child states of a parent state, and then connecting the tree node housing the parent state (a.k.a. parent node) to the nodes housing the child states (a.k.a. child nodes).
        Returns:
            False if the node is a leaf (game over)
        """
        actions = self.env.get_legal_actions()
        if self.env.check_win_condition() or len(actions) == 0:
            return False

        player = Player.TWO if node.player == Player.ONE else Player.ONE
        for action in actions:
            node.children[action] = Node(node.parent, UniversalAction(action), player)

        return True

    def evaluate_leaf(self) -> Player:
        """
        Estimating the value of a leaf node in the tree by doing a rollout simulation using the default policy from the leaf node’s state to a final state.
        """
        # Using rollout or critic
        actions = self.env.get_legal_actions()

        while not self.env.check_win_condition() and actions:
            if random.uniform(0, 1) < self.epsilon:
                action = UniversalAction(choice(actions))
            else:
                state = UniversalState(deepcopy(self.env.state.nodes), self.env.get_player_turn())
                action = self.actor.generate_action(state, actions)
            self.env.execute_action(action)
            actions.remove(action.coordinates)

        return self.env.winner

    @staticmethod
    def backprop(node: Node, player: Player, winner: Player) -> None:
        """
        Passing the evaluation of a final state back up the tree, updating relevant data at all nodes and edges on the path from the final state to the tree root.
        """

        reinforcement = 0 if winner == player else 1

        while node is not None:
            node.visit_count += 1
            node.Q_value += reinforcement
            node = node.parent
            reinforcement = 0 if reinforcement == 1 else 1
