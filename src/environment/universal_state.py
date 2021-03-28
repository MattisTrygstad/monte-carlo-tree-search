
import numpy as np
from enums import Player


class UniversalState:
    """
    Superclass for representing the state in any game. This contributes to decoupling of the environment and the agent.
    """

    def __init__(self, nodes: dict = {}, player: Player = None) -> None:
        super().__init__()
        self.nodes = nodes
        self.player = player

    def __str__(self) -> str:
        return str(list(self.nodes.values()))

    def to_numpy(self):
        return np.array([self.player.value] + list(self.nodes.values()))
