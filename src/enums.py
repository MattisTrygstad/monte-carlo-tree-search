
from enum import Enum


class NodeState(Enum):
    EMPTY = 0
    PLAYER_1 = 1
    PLAYER_2 = 2


class Player(Enum):
    ONE = 0
    TWO = 1


class Color(Enum):
    BACKGROUND = '#F1FAEE'  # White
    PLAYER_1 = '#1D3557'  # Dark blue
    EMPTY_NODE = '#457B9D'  # Light blue
    PLAYER_2 = '#E63946'  # Red


class BoardType(Enum):
    TRIANGLE = 0
    DIAMOND = 1
