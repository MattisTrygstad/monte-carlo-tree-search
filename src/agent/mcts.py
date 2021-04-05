

from copy import deepcopy
from random import choice, choices
import random
import sys
from typing import Tuple
from unittest.mock import seal
import numpy as np
from agent.actor_test import Actor
from agent.node import Node
from enums import Player
from environment.hexagonal_grid import HexagonalGrid
from environment.universal_action import UniversalAction
from environment.universal_state import UniversalState


class MonteCarloTree:

    def __init__(self, state: UniversalState, actor: Actor, epsilon: float, exploration_constant: float) -> None:
        self.root = Node(None, state.player)
        self.init_state = state
        self.env = HexagonalGrid(self.init_state, False)
        self.epsilon = epsilon
        self.exploration_constant = exploration_constant
        self.actor = actor

    def reset(self, state: UniversalState):
        self.env.reset()
        self.init_state = state
        self.root = Node(None, state.player)

    def set_root(self, action: UniversalAction, state: UniversalState):
        self.init_state = state
        self.env.reset(state)

        self.root = self.root.children[action.coordinates]

    @staticmethod
    def get_action_distribution(node: Node) -> dict:
        total_visits = node.visit_count

        distribution = {key: traversals / total_visits for key, [traversals, _] in node.edges.items()}

        return distribution

    def single_pass(self) -> None:
        node = self.tree_search()
        winner = self.evaluate_leaf()
        self.backprop(node, winner)

    def tree_policy(self, node: Node) -> UniversalAction:
        player = self.env.get_player_turn()

        assert player == node.player

        tree_policy_values = {}
        for row, col in node.children.keys():
            tree_policy_values[(row, col)] = node.evaluate_action((row, col), self.exploration_constant)

        best_action = max(tree_policy_values, key=tree_policy_values.get)
        worst_action = min(tree_policy_values, key=tree_policy_values.get)

        selected_action = best_action if player == Player.ONE else worst_action
        return UniversalAction(selected_action)

    def tree_search(self) -> Node:
        """
        Traversing the tree from the root to a leaf node by using the tree policy.
        """
        node = self.root

        self.env.reset(UniversalState(deepcopy(self.init_state.nodes), self.init_state.player))

        while len(node.children) != 0:
            action = self.tree_policy(node)
            node.cached_action = action
            self.env.execute_action(action)
            node: Node = node.children[action.coordinates]

            if node.visit_count == 0:
                return node

        if self.expand_node(node):
            action = UniversalAction(choice(self.env.get_legal_actions()))
            node.cached_action = action
            self.env.execute_action(action)
            node: Node = node.children[action.coordinates]

        return node

    def expand_node(self, node: Node) -> bool:
        """
        Generating some or all child states of a parent state, and then connecting the tree node housing the parent state (a.k.a. parent node) to the nodes housing the child states (a.k.a. child nodes).
        Returns:
            False if the node is a leaf (game over)
        """
        if self.env.check_win_condition():
            return False

        assert node.player == self.env.get_player_turn()

        player = Player.TWO if node.player == Player.ONE else Player.ONE
        actions = self.env.get_legal_actions()

        for action in actions:
            node.children[action] = Node(node, player)
            node.edges[action] = [0, 0]

        return True

    def evaluate_leaf(self) -> Player:
        """
        Estimating the value of a leaf node in the tree by doing a rollout simulation using the default policy from the leaf node’s state to a final state.
        """
        # Using rollout or critic
        actions = self.env.get_legal_actions()

        while not self.env.check_win_condition():
            if random.uniform(0, 1) < self.epsilon:
                action = UniversalAction(choice(actions))
            else:
                state = self.env.get_state()
                action = self.actor.generate_action(state, actions)
                if action.coordinates not in actions:
                    print(f'Actor chose an invalid action.\nactions: {actions}\nselected action:{action.coordinates}')
                    action = UniversalAction(choice(actions))
            self.env.execute_action(action)
            actions.remove(action.coordinates)

        return self.env.winner

    @staticmethod
    def backprop(node: Node, winner: Player) -> None:
        """
        Passing the evaluation of a final state back up the tree, updating relevant data at all nodes and edges on the path from the final state to the tree root.
        """

        # TODO: -1 or 0 ?
        reinforcement = 1 if winner == Player.ONE else -1

        node.visit_count += 1
        while node.parent:
            node.parent.visit_count += 1

            action = node.parent.cached_action.coordinates
            node.parent.edges[action][0] += 1
            edge_traversals, q_value = node.parent.edges[action]
            node.parent.edges[action][1] += (reinforcement - q_value) / edge_traversals

            node = node.parent
