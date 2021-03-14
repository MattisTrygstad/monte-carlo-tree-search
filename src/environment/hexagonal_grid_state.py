

from math import pi
from enums import BoardType, NodeState, Player
from environment.universal_state import UniversalState
from utils.config_parser import Config
from utils.trigonometry import rotation_matrix


class HexagonalGridState(UniversalState):

    def __init__(self) -> None:
        super().__init__()

        self.neighbors = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
        self.size = Config.board_size

        self.edges = []  # [((x,y),(i,j)),...)]
        self.node_names = {}  # (row, col): str
        self.node_coordinates = {}  # (row, col): (x_value, y_value)

        # Describes the last executed action
        self.start_pos = ()
        self.end_pos = ()

        self.__generate_nodes()
        self.__generate_edges()
        self.__generate_coordinates()

    def __generate_nodes(self) -> None:
        if Config.board_type == BoardType.TRIANGLE.value:
            for row in range(self.size):
                for col in range(row + 1):
                    self.nodes[(row, col)] = NodeState.EMPTY.value
                    self.node_names[(row, col)] = f'{row},{col}'
        elif Config.board_type == BoardType.DIAMOND.value:
            for x in range(self.size):
                for y in range(self.size):
                    self.nodes[(x, y)] = NodeState.EMPTY.value
                    self.node_names[(x, y)] = f'{x},{y}'

        # Set filled nodes
        for (row, col) in Config.filled_nodes:
            self.nodes[(row, col)] = NodeState.PLAYER_1.value

    def __generate_edges(self) -> None:
        for (row, col) in self.nodes.keys():
            for (x, y) in self.neighbors:
                if (row + x, col + y) in self.nodes:
                    self.edges.append(((row, col), (row + x, col + y)))

    def __generate_coordinates(self) -> None:
        for (row, col) in self.nodes:
            # Rotate entire grid 45deg to match action offsets
            (x, y) = rotation_matrix(row, col, - 3 * pi / 4)
            #(x, y) = (row, col)
            self.node_coordinates[(row, col)] = (x, y)

    def get_empty_nodes(self) -> dict:
        return {key: value for (key, value) in self.nodes.items() if value == NodeState.EMPTY.value}

    def get_player_one_nodes(self) -> dict:
        return {key: value for (key, value) in self.nodes.items() if value == NodeState.PLAYER_1.value}

    def get_player_two_nodes(self) -> dict:
        return {key: value for (key, value) in self.nodes.items() if value == NodeState.PLAYER_2.value}
