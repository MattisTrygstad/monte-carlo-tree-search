from enums import NodeState, Player


def check_win_condition(board_size: int, player: Player, nodes: dict) -> bool:
    start_index = 0
    goal_index = board_size - 1
    neighbors = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

    if player == Player.ONE:
        nodes = get_player_one_nodes(nodes).keys()
        node_index = 1
    elif player == Player.TWO:
        nodes = get_player_two_nodes(nodes).keys()
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

            for neighbor in neighbors:
                next_node = (curr_node[0] + neighbor[0], curr_node[1] + neighbor[1])
                if next_node not in nodes or next_node[node_index] == start_index:
                    continue

                new_path = list(path)
                new_path.append(next_node)
                queue.append(new_path)

                if next_node[node_index] == goal_index:
                    # print('Shortest path:', *new_path)
                    # print(f'Player {player.value+1} won')
                    return True

            visited.append(curr_node)

    return False


def get_player_one_nodes(nodes: dict) -> dict:
    return {key: value for (key, value) in nodes.items() if value == NodeState.PLAYER_1.value}


def get_player_two_nodes(nodes: dict) -> dict:
    return {key: value for (key, value) in nodes.items() if value == NodeState.PLAYER_2.value}
