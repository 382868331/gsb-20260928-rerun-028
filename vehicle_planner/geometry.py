from __future__ import annotations

from typing import Iterable

from .models import RectangleObstacle


def wrap_angle(angle: float) -> float:
    import math

    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def angle_distance(a: float, b: float) -> float:
    import math

    return abs(wrap_angle(a - b))


def rotate(x: float, y: float, angle: float) -> tuple[float, float]:
    import math

    c = math.cos(angle)
    s = math.sin(angle)
    return x * c - y * s, x * s + y * c


def interval_overlap(min_a: float, max_a: float, min_b: float, max_b: float) -> bool:
    return max(min_a, min_b) < min(max_a, max_b)


def project_vertices(vertices, axis_x, axis_y):
    projections = [x * axis_x + y * axis_y for x, y in vertices]
    return min(projections), max(projections)


def rectangles_intersect(vertices_a, vertices_b) -> bool:
    for vertices in (vertices_a, vertices_b):
        for index, (x1, y1) in enumerate(vertices):
            x2, y2 = vertices[(index + 1) % len(vertices)]
            axis_x = y2 - y1
            axis_y = x1 - x2
            min_a, max_a = project_vertices(vertices_a, axis_x, axis_y)
            min_b, max_b = project_vertices(vertices_b, axis_x, axis_y)
            if not interval_overlap(min_a, max_a, min_b, max_b):
                return False
    return True


def obstacle_vertices(obstacle: RectangleObstacle):
    return [
        (obstacle.min_x, obstacle.min_y),
        (obstacle.max_x, obstacle.min_y),
        (obstacle.max_x, obstacle.max_y),
        (obstacle.min_x, obstacle.max_y),
    ]
