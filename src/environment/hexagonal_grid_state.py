

from math import pi
from enums import BoardType, NodeState, Player
from environment.universal_state import UniversalState
from utils.config_parser import Config
from utils.trigonometry import rotation_matrix


class HexagonalGridState(UniversalState):

    def __init__(self, state: UniversalState = None) -> None:
        super().__init__()

        self.neighbors = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
        self.size = Config.board_size

        self.edges = []  # [((x,y),(i,j)),...)]
        self.node_names = {}  # (row, col): str
        self.node_coordinates = {}  # (row, col): (x_value, y_value)

        if not state:
            self.__generate_nodes()
        else:
            self.nodes = state.nodes
            self.node_names = {key: f'{key[0]}, {key[1]}' for key in self.nodes.keys()}

        self.__generate_edges()
        self.__generate_coordinates()

    def __generate_nodes(self) -> None:
        for x in range(self.size):
            for y in range(self.size):
                self.nodes[(x, y)] = NodeState.EMPTY.value
                self.node_names[(x, y)] = f'{x},{y}'

    def __generate_edges(self) -> None:
        for (row, col) in self.nodes.keys():
            for (x, y) in self.neighbors:
                if (row + x, col + y) in self.nodes:
                    self.edges.append(((row, col), (row + x, col + y)))

    def __generate_coordinates(self) -> None:
        for (row, col) in self.nodes:
            # Rotate entire grid 45deg to match action offsets
            (x, y) = rotation_matrix(row, col, - 3 * pi / 4)
            # (x, y) = (row, col)
            self.node_coordinates[(row, col)] = (x, y)

    def get_empty_nodes(self) -> dict:
        return {key: value for (key, value) in self.nodes.items() if value == NodeState.EMPTY.value}

    def get_player_one_nodes(self) -> dict:
        return {key: value for (key, value) in self.nodes.items() if value == NodeState.PLAYER_1.value}

    def get_player_two_nodes(self) -> dict:
        return {key: value for (key, value) in self.nodes.items() if value == NodeState.PLAYER_2.value}
