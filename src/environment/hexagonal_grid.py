
from copy import deepcopy
import math
import sys
from matplotlib import pyplot as plt
import numpy as np
from abstract_classes.environment import Environment
from enums import BoardType, Color, NodeState, Player
import networkx as nx
from typing import Dict, Tuple
from environment.hexagonal_grid_state import HexagonalGridState
from environment.universal_action import UniversalAction
from environment.universal_state import UniversalState

from utils.config_parser import Config


class HexagonalGrid(Environment):

    def __init__(self, win_multiplier: int):
        self.state = HexagonalGridState()
        self.history = []

        if Config.board_type == BoardType.DIAMOND.value:
            self.fig, self.ax = plt.subplots(figsize=(7, 8))
        else:
            self.fig, self.ax = plt.subplots(figsize=(9, 7))

        self.G = nx.Graph()
        self.G.add_nodes_from(self.state.nodes)
        self.G.add_edges_from(self.state.edges)

        self.game_counter = 0

    def get_player_turn(self) -> Player:
        return Player.ONE if self.game_counter % 2 == 0 else Player.TWO

    def execute_action(self, action: UniversalAction) -> int:
        self.history.append(deepcopy(self.state.nodes))

        node_pos = action.action

        player = self.get_player_turn()

        if player == Player.ONE:
            self.state.nodes[node_pos] = NodeState.PLAYER_1.value
        elif player == Player.TWO:
            self.state.nodes[node_pos] = NodeState.PLAYER_2.value

        self.game_counter += 1
        return 1

    def undo_action(self) -> None:
        if self.history:
            self.state.nodes = self.history.pop()

    def get_legal_actions(self) -> list:
        return list(self.state.get_empty_nodes().keys())

    def check_win_condition(self) -> bool:
        start_col = 0
        goal_col = self.state.size - 1

        nodes = self.state.get_player_one_nodes()
        path = {}
        for start_node in nodes.keys():
            if start_node[1] != start_col:
                continue

            distance = {}
            q = {}
            path = {}
            for n in nodes.keys():
                distance[n] = math.inf
                q[n] = math.inf
                path[n] = None

            distance[start_node] = 0

            while q:
                curr_node = min(q, key=q.get)

                del q[curr_node]

                for neighbor in self.state.neighbors:
                    next_node = (curr_node[0] + neighbor[0], curr_node[1] + neighbor[1])
                    if next_node not in nodes:
                        continue

                    new_distance = distance[curr_node] + 1
                    if new_distance < distance[next_node]:
                        distance[next_node] = new_distance
                        path[next_node] = curr_node

        print(path)

        shortest_path = []
        for (child, parent) in path.items():
            if child[1] != goal_col:
                continue

            shortest_path.append(child)

            while True:
                print('test')
                if not parent:
                    break
                child = path[parent]
                shortest_path.append(child)
                parent = child

        shortest_path.reverse()
        print(shortest_path)

    def get_state(self) -> UniversalState:
        universal_state = UniversalState()
        universal_state.nodes = deepcopy(self.state.nodes)

        return universal_state

    def reset(self) -> None:
        self.state = HexagonalGridState()

        self.G = nx.Graph()
        self.G.add_nodes_from(self.state.nodes)
        self.G.add_edges_from(self.state.edges)

    def visualize(self, block: bool, delay: int = None) -> None:
        plt.cla()
        empty_nodes = self.state.get_empty_nodes()
        player_one_nodes = self.state.get_player_one_nodes()
        player_two_nodes = self.state.get_player_two_nodes()
        node_names = self.state.node_names
        node_coordinates = self.state.node_coordinates

        nx.draw(self.G, pos=node_coordinates, nodelist=empty_nodes, node_color=Color.EMPTY_NODE.value, node_size=800)
        nx.draw(self.G, pos=node_coordinates, nodelist=player_one_nodes, node_color=Color.PLAYER_1.value, node_size=800, ax=self.ax, labels=node_names, font_color=Color.BACKGROUND.value)
        nx.draw(self.G, pos=node_coordinates, nodelist=player_two_nodes, node_color=Color.PLAYER_2.value, node_size=800, ax=self.ax, labels=node_names, font_color=Color.BACKGROUND.value)

        """ if self.history:
            nx.draw(self.G, pos=node_coordinates, nodelist=[self.state.start_pos], node_color=Color.RED.value, node_size=1200)
            nx.draw(self.G, pos=node_coordinates, nodelist=[self.state.start_pos], node_color=Color.LIGHT_BLUE.value, node_size=800)
            nx.draw(self.G, pos=node_coordinates, nodelist=[self.state.end_pos], node_color=Color.RED.value, node_size=1200)
            nx.draw(self.G, pos=node_coordinates, nodelist=[self.state.end_pos], node_color=Color.DARK_BLUE.value, node_size=800) """

        """ self.ax.set_axis_on()
        self.ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.title('Peg Solitaire') """

        x1 = 1.2
        x2 = -1.8
        y1 = -0.8
        y2 = -3.5

        self.ax.text(x2, y1, 'Player 1', fontsize=15, color=Color.PLAYER_1.value)
        self.ax.text(x2, y2, 'Player 2', fontsize=15, color=Color.PLAYER_2.value)
        self.ax.text(x1, y1, 'Player 2', fontsize=15, color=Color.PLAYER_2.value)
        self.ax.text(x1, y2, 'Player 1', fontsize=15, color=Color.PLAYER_1.value)

        self.fig.tight_layout()
        plt.show(block=block)
        self.fig.patch.set_facecolor(Color.BACKGROUND.value)
        plt.pause(0.1)

        if delay:
            plt.pause(delay)
