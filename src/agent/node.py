import math
import numpy as np
from enums import Player
from environment.universal_action import UniversalAction
from environment.universal_state import UniversalState


class Node:

    def __init__(self, parent: 'Node', player: Player) -> None:
        self.parent = parent
        self.cached_action = None
        self.player = player
        self.children = {}  # Key: node coordinates, value: Node()
        self.edges = {}  # Key: node coordinates, value: [traversals, q_value]
        self.visit_count = 0

    def evaluate_action(self, action: tuple, exploration_constant: float) -> float:
        edge_traversals, q_value = self.edges[action]
        exploration_constant *= 1 if self.player == Player.ONE else -1

        return q_value + exploration_constant * np.sqrt(np.log(self.visit_count) / (1 + edge_traversals))

    def __str__(self) -> str:
        return f'cached_action: {self.cached_action}\nplayer: {self.player}\nedges: {len(self.edges)}\nvisit_count: {self.visit_count}\nparent: {True if self.parent else False}'
