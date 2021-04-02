from collections import defaultdict
import numpy as np


def generate_bridge_neighbors(size: int, neighbors: dict):
    bridge_neighbors = defaultdict(list)
    n_mat = np.zeros(size**4).reshape(size**2, size**2)
    for f in range(size**2):
        for t in range(size**2):
            if f != t:
                c_from = (f // size, f % size)
                c_to = (t // size, t % size)
                if c_to in neighbors[c_from]:
                    n_mat[f][t] = 1
    two_step_mat = np.matmul(n_mat, n_mat)
    np.fill_diagonal(two_step_mat, 0)
    for f in range(size**2):
        for t in range(size**2):
            if f != t:
                c_from = (f // size, f % size)
                c_to = (t // size, t % size)
                if two_step_mat[f][t] == 2 and c_to not in neighbors[c_from]:
                    bridge_neighbors[c_from].append(c_to)

    return bridge_neighbors
