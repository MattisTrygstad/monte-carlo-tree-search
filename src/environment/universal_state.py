
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

    def to_numpy(self):
        board_size = int(sqrt(len(self.nodes.keys())))
        state = np.zeros((board_size**2,))
        for (row, col), value in self.nodes.items():
            index = row * board_size + col
            state[index] = value

        assert self.nodes[(1, 3)] == state[7]
        assert self.nodes[(3, 3)] == state[15]

        array = np.concatenate([np.array([self.player.value]), state])
        assert array.shape == (17,)

        return array
