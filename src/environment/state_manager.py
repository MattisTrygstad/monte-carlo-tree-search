
import math
import sys
from copy import deepcopy
from random import choice

import networkx as nx
from abstract_classes.environment import Environment
from enums import Color, NodeState, Player
from matplotlib import pyplot as plt

from environment.hexagonal_grid_state import HexagonalGridState
from environment.universal_action import UniversalAction
from environment.universal_state import UniversalState


class StateManager(Environment):

    def __init__(self, visual: bool = False):
        self.visual = visual
        if visual:
            self.fig, self.ax = plt.subplots(figsize=(7, 8))

        self.state = HexagonalGridState()

        self.G = nx.Graph()
        self.G.add_nodes_from(self.state.nodes)
        self.G.add_edges_from(self.state.edges, color='black', weight=1)

    def get_player_turn(self) -> Player:
        return Player.ONE if self.game_counter % 2 == 0 else Player.TWO

    def get_player_win_condition(self) -> Player:
        return Player.TWO if self.game_counter % 2 == 0 else Player.ONE

    def execute_action(self, action: UniversalAction) -> int:

        assert action.coordinates in self.get_legal_actions()

        node_pos = action.coordinates

        player = self.get_player_turn()

        if player == Player.ONE:
            self.state.nodes[node_pos] = NodeState.PLAYER_1.value
        elif player == Player.TWO:
            self.state.nodes[node_pos] = NodeState.PLAYER_2.value

        self.game_counter += 1
        return 1

    def get_legal_actions(self) -> list:
        return list(self.state.get_empty_nodes().keys())

    def check_win_condition(self) -> bool:
        start_index = 0
        goal_index = self.state.size - 1

        player = self.get_player_win_condition()

        if player == Player.ONE:
            nodes = self.state.get_player_one_nodes().keys()
            node_index = 1
        elif player == Player.TWO:
            nodes = self.state.get_player_two_nodes().keys()
            node_index = 0

        # Find shortest path between the two sides using BFS
        for start_node in nodes:
            if start_node[node_index] != start_index:
                continue

            queue = [[start_node]]
            visited = []

            while queue:
                path = queue.pop(0)
                curr_node = path[-1]

                if curr_node in visited:
                    continue

                for neighbor in self.state.neighbors:
                    next_node = (curr_node[0] + neighbor[0], curr_node[1] + neighbor[1])
                    if next_node not in nodes or next_node[node_index] == start_index:
                        continue

                    new_path = list(path)
                    new_path.append(next_node)
                    queue.append(new_path)

                    if next_node[node_index] == goal_index:
                        # print('Shortest path:', *new_path)
                        # print(f'Player {player.value+1} won')
                        self.shortest_path = new_path
                        self.winner = player
                        return True

                visited.append(curr_node)

        return False

    def get_state(self) -> UniversalState:
        return UniversalState(deepcopy(self.state.nodes), self.get_player_turn())

    def reset(self, state: UniversalState = None, random=False) -> None:

        if hasattr(self, 'winner'):
            del self.winner

        if hasattr(self, 'shortest_path'):
            del self.shortest_path

        if state:
            self.state = HexagonalGridState(deepcopy(state))
            self.game_counter = 0 if state.player == Player.ONE else 1
        elif random:
            self.state = HexagonalGridState()
            self.game_counter = choice([0, 1])
        else:
            print('WARNING: Player 1 always start')
            self.game_counter = 0

    def visualize(self, block: bool, delay: int = None) -> None:

        plt.cla()
        empty_nodes = self.state.get_empty_nodes()
        player_one_nodes = self.state.get_player_one_nodes()
        player_two_nodes = self.state.get_player_two_nodes()
        node_names = self.state.node_names
        node_coordinates = self.state.node_coordinates

        if hasattr(self, 'shortest_path'):
            color = Color.PLAYER_1.value if self.winner == Player.ONE else Color.PLAYER_2.value
            for x in range(1, len(self.shortest_path), 1):
                start = self.shortest_path[x - 1]
                end = self.shortest_path[x]
                self.G.remove_edge(start, end)
                self.G.add_edge(start, end, color=color, weight=5)

        edges = self.G.edges()

        colors = [self.G[u][v]['color'] for u, v in edges]
        weights = [self.G[u][v]['weight'] for u, v in edges]

        nx.draw(self.G, pos=node_coordinates, nodelist=player_one_nodes, node_color=Color.PLAYER_1.value, node_size=800, ax=self.ax, labels=node_names, font_color=Color.BACKGROUND.value)
        nx.draw(self.G, pos=node_coordinates, nodelist=player_two_nodes, node_color=Color.PLAYER_2.value, node_size=800, ax=self.ax, labels=node_names, font_color=Color.BACKGROUND.value)
        nx.draw(self.G, pos=node_coordinates, edge_color=colors, width=weights, nodelist=empty_nodes, node_color=Color.EMPTY_NODE.value, node_size=800)

        if self.state.size == 4:
            x1 = 1.2
            x2 = -1.8
            y1 = -0.8
            y2 = -3.5

            self.ax.text(x2, y1, 'Player 2', fontsize=15, color=Color.PLAYER_1.value)
            self.ax.text(x2, y2, 'Player 1', fontsize=15, color=Color.PLAYER_2.value)
            self.ax.text(x1, y1, 'Player 1', fontsize=15, color=Color.PLAYER_2.value)
            self.ax.text(x1, y2, 'Player 2', fontsize=15, color=Color.PLAYER_1.value)

        self.fig.tight_layout()
        plt.show(block=block)
        self.fig.patch.set_facecolor(Color.BACKGROUND.value)
        plt.pause(0.1)

        if delay:
            plt.pause(delay)
