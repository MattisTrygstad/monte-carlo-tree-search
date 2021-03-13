from math import cos, sin


def rotation_matrix(x: int, y: int, rad: float) -> tuple:
    return (x * cos(rad) - y * sin(rad), x * sin(rad) + y * cos(rad))
