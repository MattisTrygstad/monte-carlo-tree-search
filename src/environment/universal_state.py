
from math import sqrt
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

    def generate_actor_input(self):
        state = self.to_numpy()
        array = np.concatenate([np.array([self.player.value]), state])

        return array

    def to_numpy(self):
        board_size = int(sqrt(len(self.nodes.keys())))
        state = np.zeros((board_size**2,), dtype=int)
        for (row, col), value in self.nodes.items():
            index = row * board_size + col
            state[index] = value

        return state

    def ordered_keys(self):
        board_size = int(sqrt(len(self.nodes.keys())))
        state = np.zeros((board_size**2,), dtype=object)
        for (row, col) in self.nodes.keys():
            index = row * board_size + col
            state[index] = (row, col)

        return state
