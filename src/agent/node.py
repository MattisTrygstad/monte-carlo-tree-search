import math
import numpy as np
from enums import Player
from environment.universal_action import UniversalAction
from environment.universal_state import UniversalState


class Node:

    def __init__(self, parent: 'Node', action: UniversalAction, player: Player) -> None:
        self.parent = parent
        self.previous_action = action
        self.player = player
        self.children = {}  # Key: node coordinate, value: Node()
        self.visit_count = 0
        self.Q_value = 0

    def compute_uct(self, exploitation_factor: float) -> float:
        exploitation_factor *= -1 if self.player == Player.TWO else 1

        if self.visit_count == 0:
            return 0 if exploitation_factor == 0 else math.inf
        else:
            return self.Q_value / self.visit_count + exploitation_factor * math.sqrt(2 * math.log(self.parent.visit_count) / self.visit_count)
